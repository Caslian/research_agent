"""
InnoCore AI 智能体控制器 - LangGraph 多智能体编排
负责四大智能体的协同调度、任务编排和执行日志
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agents.base import BaseAgent
from agents.coach import CoachAgent
from agents.hunter import HunterAgent
from agents.miner import MinerAgent
from agents.validator import ValidatorAgent
from core.config import get_config
from core.exceptions import AgentException

logger = logging.getLogger(__name__)


class TaskType(Enum):
    PAPER_HUNTING = "paper_hunting"
    PAPER_ANALYSIS = "paper_analysis"
    WRITING_ASSISTANCE = "writing_assistance"
    CITATION_VALIDATION = "citation_validation"
    FULL_WORKFLOW = "full_workflow"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentController:
    """智能体控制器 - Redis 任务队列 + PostgreSQL 执行日志"""

    def __init__(self):
        self.config = get_config()

        self.agents = {
            "hunter": HunterAgent(),
            "miner": MinerAgent(),
            "coach": CoachAgent(),
            "validator": ValidatorAgent(),
        }

        # 内存任务管理（降级方案）
        self.active_tasks: Dict[str, Dict] = {}
        self.task_history: List[Dict] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # 并发控制
        self.semaphore = asyncio.Semaphore(self.config.concurrent_agents)

        # Redis 是否可用
        self._redis_available = False

        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "agent_status_changed": [],
        }

    async def initialize(self):
        """初始化控制器"""
        logger.info("初始化 Agent Controller...")
        try:
            from core.redis_manager import redis_manager
            await redis_manager.initialize()
            self._redis_available = True
            logger.info("Agent Controller 使用 Redis 任务队列")
        except Exception as e:
            logger.warning(f"Redis 不可用，使用内存任务队列: {str(e)}")
            self._redis_available = False
        logger.info("Agent Controller 初始化完成")

    async def submit_task(
        self, task_type: TaskType, input_data: Dict[str, Any],
        priority: int = 0, callback: Callable = None,
    ) -> str:
        """提交任务到队列"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_tasks)}"

        task = {
            "id": task_id,
            "type": task_type,
            "input_data": input_data,
            "status": TaskStatus.PENDING,
            "priority": priority,
            "callback": callback,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "agent_results": {},
        }

        self.active_tasks[task_id] = task

        if self._redis_available:
            try:
                from core.redis_manager import redis_manager
                await redis_manager.push_task("task_queue", task_id, priority)
                await redis_manager.set_active_task(task_id, task)
            except Exception:
                pass

        await self.task_queue.put((priority, task))
        logger.info(f"任务已提交: {task_id}, 类型: {task_type.value}")
        return task_id

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """执行单个任务"""
        if task_id not in self.active_tasks:
            raise AgentException(f"任务不存在: {task_id}")

        task = self.active_tasks[task_id]

        # 防止重复执行：如果任务已在执行或已完成，直接返回
        if task["status"] != TaskStatus.PENDING:
            logger.warning(f"任务 {task_id} 已在执行或已完成 (状态: {task['status'].value})，跳过重复执行")
            if task["status"] == TaskStatus.COMPLETED:
                return task.get("result", {})
            raise AgentException(f"任务 {task_id} 已在执行中")

        async with self.semaphore:
            task["status"] = TaskStatus.RUNNING
            start_time = datetime.now()
            task["started_at"] = start_time
            await self._trigger_event("task_started", task)

            # 记录 Agent 执行开始
            exec_id = None
            try:
                from core.database import db_manager
                exec_id = await db_manager.log_agent_execution(
                    agent_name="controller",
                    task_type=task["type"].value,
                    task_id=task_id,
                    input_summary=json.dumps(task["input_data"], ensure_ascii=False)[:500],
                )
            except Exception:
                pass

            try:
                result = await self._dispatch_task(task)
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                task["status"] = TaskStatus.COMPLETED
                task["completed_at"] = datetime.now()
                task["result"] = result

                if exec_id:
                    try:
                        from core.database import db_manager
                        await db_manager.update_agent_execution(
                            exec_id, "completed",
                            output_summary=json.dumps(result, ensure_ascii=False)[:500],
                            duration_ms=duration_ms,
                        )
                    except Exception:
                        pass

                await self._trigger_event("task_completed", task)
                if task.get("callback"):
                    await task["callback"](task)

                return result

            except Exception as e:
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                task["status"] = TaskStatus.FAILED
                task["completed_at"] = datetime.now()
                task["error"] = str(e)

                if exec_id:
                    try:
                        from core.database import db_manager
                        await db_manager.update_agent_execution(
                            exec_id, "failed", duration_ms=duration_ms, error_message=str(e),
                        )
                    except Exception:
                        pass

                await self._trigger_event("task_failed", task)
                logger.error(f"任务执行失败 {task_id}: {str(e)}")
                raise AgentException(f"任务执行失败: {str(e)}")

            finally:
                self.task_history.append(task.copy())
                del self.active_tasks[task_id]
                if self._redis_available:
                    try:
                        from core.redis_manager import redis_manager
                        await redis_manager.remove_active_task(task_id)
                        await redis_manager.push_task_history(task)
                    except Exception:
                        pass

    async def _dispatch_task(self, task: Dict) -> Dict[str, Any]:
        """分发任务到对应 Agent"""
        dispatch_map = {
            TaskType.PAPER_HUNTING: self._execute_paper_hunting,
            TaskType.PAPER_ANALYSIS: self._execute_paper_analysis,
            TaskType.WRITING_ASSISTANCE: self._execute_writing_assistance,
            TaskType.CITATION_VALIDATION: self._execute_citation_validation,
            TaskType.FULL_WORKFLOW: self._execute_full_workflow,
        }
        handler = dispatch_map.get(task["type"])
        if not handler:
            raise AgentException(f"不支持的任务类型: {task['type']}")
        return await handler(task)

    async def _execute_paper_hunting(self, task: Dict) -> Dict[str, Any]:
        result = await self.agents["hunter"].run(task["input_data"])
        task["agent_results"]["hunter"] = result
        return {
            "task_type": "paper_hunting",
            "papers_found": result.get("papers", []),
            "statistics": {
                "total_found": result.get("total_found", 0),
                "downloaded": result.get("downloaded_papers", 0),
            },
        }

    async def _execute_paper_analysis(self, task: Dict) -> Dict[str, Any]:
        result = await self.agents["miner"].run(task["input_data"])
        task["agent_results"]["miner"] = result
        return {
            "task_type": "paper_analysis",
            "analysis_report": result,
            "paper_id": task["input_data"].get("paper_id"),
        }

    async def _execute_writing_assistance(self, task: Dict) -> Dict[str, Any]:
        result = await self.agents["coach"].run(task["input_data"])
        task["agent_results"]["coach"] = result
        return {
            "task_type": "writing_assistance",
            # "assistance_result": result['result']['polished_text'],
            "assistance_result": result,
            "user_id": task["input_data"].get("user_id"),
        }

    async def _execute_citation_validation(self, task: Dict) -> Dict[str, Any]:
        result = await self.agents["validator"].run(task["input_data"])
        task["agent_results"]["validator"] = result
        return {
            "task_type": "citation_validation",
            "validation_result": result,
            "paper_info": task["input_data"].get("paper_info"),
        }

    async def _execute_full_workflow(self, task: Dict) -> Dict[str, Any]:
        """执行完整工作流: Hunter -> Miner(xN) -> Validator -> Coach"""
        input_data = task["input_data"]
        user_id = input_data.get("user_id")
        keywords = input_data.get("keywords", [])

        workflow_result = {
            "task_type": "full_workflow",
            "stages": {},
            "final_papers": [],
            "analysis_reports": [],
        }

        # 记录工作流
        wf_id = None
        try:
            from core.database import db_manager
            wf_id = await db_manager.create_workflow(
                user_id or "anonymous", "full",
                steps=["hunting", "analysis", "validation", "coach"],
            )
        except Exception:
            pass

        try:
            # Stage 1: Hunter - 论文搜索
            logger.info("工作流 Stage 1: 论文搜索")
            hunting_result = await self.agents["hunter"].run({
                "keywords": keywords,
                "max_papers": input_data.get("max_papers", 10),
                "sources": input_data.get("sources", ["arxiv"]),
            })
            workflow_result["stages"]["hunting"] = hunting_result
            task["agent_results"]["hunter"] = hunting_result
            papers = hunting_result.get("papers", [])

            # Stage 2: Miner - 并行分析
            logger.info(f"[Workflow] Stage 2: 并行分析 {len(papers)} 篇论文")
            analysis_tasks = []
            for paper in papers[:5]:
                miner_input = None
                if paper.get("db_id"):
                    miner_input = {"paper_id": paper["db_id"], "user_id": user_id, "analysis_type": "full"}
                elif paper.get("pdf_url"):
                    miner_input = {"paper_url": paper["pdf_url"], "user_id": user_id, "analysis_type": "full"}
                elif paper.get("title") and paper.get("abstract"):
                    miner_input = {
                        "title": paper["title"], "abstract": paper["abstract"],
                        "authors": paper.get("authors", []), "user_id": user_id, "analysis_type": "full",
                    }
                if miner_input:
                    logger.info(f"[Workflow] 提交 MinerAgent 分析: %s", paper.get("title", "")[:60])
                    analysis_tasks.append(self.agents["miner"].run(miner_input))
                else:
                    logger.warning(f"[Workflow] 跳过论文（缺少可用的标识符）: %s", paper.get("title", "")[:60])

            if analysis_tasks:
                analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                for i, analysis in enumerate(analyses):
                    if isinstance(analysis, Exception):
                        logger.warning(f"[Workflow] Miner 分析失败: {str(analysis)}")
                    else:
                        workflow_result["analysis_reports"].append(analysis)
            else:
                logger.warning("[Workflow] Stage 2: 没有论文需要分析（所有论文缺少 db_id/url/title）")

            # Stage 3: Validator - 引用生成
            if input_data.get("validate_citations", False):
                logger.info("工作流 Stage 3: 引用校验")
                for paper in papers:
                    try:
                        v_result = await self.agents["validator"].run({
                            "paper_info": {
                                "title": paper.get("title", ""),
                                "authors": paper.get("authors", []),
                                "doi": paper.get("doi", ""),
                                "year": datetime.now().year,
                            },
                            "formats": ["bibtex", "apa"],
                            "verify_external": True,
                        })
                        paper["citations"] = v_result.get("citations", {})
                    except Exception as e:
                        logger.warning(f"引用校验失败: {str(e)}")

            workflow_result["final_papers"] = papers

            if wf_id:
                try:
                    from core.database import db_manager
                    await db_manager.update_workflow(wf_id, "completed", workflow_result)
                except Exception:
                    pass

            logger.info("完整工作流执行完成")
            return workflow_result

        except Exception as e:
            if wf_id:
                try:
                    from core.database import db_manager
                    await db_manager.update_workflow(wf_id, "failed", {"error": str(e)})
                except Exception:
                    pass
            raise

    async def start_task_processor(self):
        """启动后台任务处理器"""
        logger.info("任务处理器已启动")
        while True:
            try:
                priority, task = await self.task_queue.get()
                task_id = task["id"]
                # 跳过已被同步执行器处理完的任务
                if task_id not in self.active_tasks or self.active_tasks[task_id]["status"] != TaskStatus.PENDING:
                    continue
                asyncio.create_task(self.execute_task(task_id))
            except Exception as e:
                logger.error(f"任务处理器异常: {str(e)}")
                await asyncio.sleep(1)

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self.active_tasks.get(task_id)
        if not task:
            for t in self.task_history:
                if t["id"] == task_id:
                    task = t
                    break
        if not task:
            return None
        return {
            "id": task["id"],
            "type": task["type"].value,
            "status": task["status"].value,
            "created_at": task["created_at"].isoformat(),
            "started_at": task["started_at"].isoformat() if task["started_at"] else None,
            "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None,
            "priority": task["priority"],
        }

    async def cancel_task(self, task_id: str) -> bool:
        if task_id in self.active_tasks and self.active_tasks[task_id]["status"] == TaskStatus.PENDING:
            task = self.active_tasks[task_id]
            task["status"] = TaskStatus.CANCELLED
            task["completed_at"] = datetime.now()
            self.task_history.append(task.copy())
            del self.active_tasks[task_id]
            logger.info(f"任务已取消: {task_id}")
            return True
        return False

    async def get_agent_status(self) -> Dict[str, Any]:
        agent_status = {name: agent.get_status() for name, agent in self.agents.items()}
        return {
            "agents": agent_status,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.task_history),
            "max_concurrent": self.config.concurrent_agents,
            "redis_available": self._redis_available,
        }

    def add_event_callback(self, event_type: str, callback: Callable):
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)

    async def _trigger_event(self, event_type: str, data: Any):
        for cb in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)
            except Exception as e:
                logger.error(f"事件回调失败 {event_type}: {str(e)}")

    async def shutdown(self):
        """关闭控制器"""
        logger.info("关闭 Agent Controller...")
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        try:
            from core.redis_manager import redis_manager
            await redis_manager.close()
        except Exception:
            pass
        logger.info("Agent Controller 已关闭")


agent_controller = AgentController()
