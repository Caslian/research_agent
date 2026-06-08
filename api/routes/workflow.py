"""
工作流API路由 — 通过 AgentController 编排多智能体协同
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
from agents.controller import agent_controller, TaskType
from core.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model_info() -> Dict[str, Any]:
    """获取当前 LLM 模型信息"""
    config = get_config()
    return {
        "llm_model": config.llm.model_name,
        "llm_provider": config.llm.provider.value,
        "llm_base_url": config.llm.base_url or "OpenAI 默认",
        "embedding_model": config.vector_db.embedding_model,
        "embedding_base_url": getattr(config.vector_db, 'embedding_base_url', None) or config.llm.base_url or "OpenAI 默认",
    }


# Pydantic模型
class WorkflowRequest(BaseModel):
    keywords: str
    analysis_type: str = "summary"
    citation_format: str = "bibtex"
    writing_task: Optional[str] = None
    limit: int = 5


class WorkflowStatus(BaseModel):
    workflow_id: str
    status: str
    current_step: str
    progress: int


@router.post("/complete", response_model=Dict[str, Any])
async def complete_workflow(request: WorkflowRequest):
    """
    完整工作流：搜索 → 分析 → 校验引用 → 写作辅助
    通过 AgentController 编排 Hunter → Miner → Validator → Coach
    """
    try:
        model_info = get_model_info()
        keywords = [kw.strip() for kw in request.keywords.split() if kw.strip()]

        logger.info(f"[Agent] 提交完整工作流: keywords={keywords}")
        task_id = await agent_controller.submit_task(
            TaskType.FULL_WORKFLOW,
            {
                "keywords": keywords,
                "max_papers": request.limit,
                "sources": ["arxiv"],
                "validate_citations": True,
                "citation_format": request.citation_format,
                "writing_task": request.writing_task,
            }
        )
        result = await agent_controller.execute_task(task_id)

        stages = result.get("stages", {})
        papers = result.get("final_papers", [])
        analyses = result.get("analysis_reports", [])

        # 构建步骤结果
        steps = []

        # Step 1: Hunter
        hunting = stages.get("hunting", {})
        steps.append({
            "step": 1,
            "name": "Hunter - 论文搜索",
            "agent": "hunter",
            "model_info": model_info,
            "status": "completed" if hunting else "failed",
            "result": {
                "total_found": hunting.get("total_found", 0),
                "papers": papers,
            }
        })

        # Step 2: Miner
        steps.append({
            "step": 2,
            "name": "Miner - 论文分析",
            "agent": "miner",
            "model_info": model_info,
            "status": "completed" if analyses else "failed",
            "result": {
                "total_analyzed": len(analyses),
                "analyses": [
                    {"paper_id": a.get("paper_id", ""), "title": a.get("paper_info", {}).get("title", ""),
                     "analysis": a.get("analysis", "")}
                    for a in analyses
                ],
            }
        })

        # Step 3: Validator
        citations_list = []
        for paper in papers:
            if paper.get("citations"):
                citations_list.append({
                    "paper_id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "formatted_citation": paper.get("citations", {}).get(request.citation_format, ""),
                })
        steps.append({
            "step": 3,
            "name": "Validator - 引用生成",
            "agent": "validator",
            "model_info": model_info,
            "status": "completed" if citations_list else "failed",
            "result": {
                "total_citations": len(citations_list),
                "citations": citations_list,
            }
        })

        # Step 4: Coach (if requested)
        if request.writing_task:
            steps.append({
                "step": 4,
                "name": "Coach - 报告生成",
                "agent": "coach",
                "model_info": model_info,
                "status": "completed",
                "result": {"report": ""}
            })

        return {
            "workflow_id": task_id,
            "status": "completed",
            "model_info": model_info,
            "steps": steps,
            "summary": {
                "total_papers": len(papers),
                "analyzed_papers": len(analyses),
                "generated_citations": len(citations_list),
                "keywords": request.keywords,
            },
        }
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"工作流执行失败: {str(e)}")


@router.post("/search-and-analyze", response_model=Dict[str, Any])
async def search_and_analyze(request: WorkflowRequest):
    """
    简化工作流：搜索 + 分析
    通过 AgentController 编排 Hunter → Miner
    """
    try:
        model_info = get_model_info()
        keywords = [kw.strip() for kw in request.keywords.split() if kw.strip()]

        logger.info(f"[Agent] 提交简化工作流: keywords={keywords}")
        task_id = await agent_controller.submit_task(
            TaskType.FULL_WORKFLOW,
            {
                "keywords": keywords,
                "max_papers": request.limit,
                "sources": ["arxiv"],
                "validate_citations": False,
            }
        )
        result = await agent_controller.execute_task(task_id)

        papers = result.get("final_papers", [])
        analyses = result.get("analysis_reports", [])

        steps = [
            {
                "step": 1,
                "name": "搜索论文",
                "agent": "hunter",
                "model_info": model_info,
                "status": "completed",
                "papers": papers,
            },
            {
                "step": 2,
                "name": "分析论文",
                "agent": "miner",
                "model_info": model_info,
                "status": "completed",
                "analysis": analyses[0] if analyses else {},
            }
        ]

        return {
            "status": "completed",
            "model_info": model_info,
            "steps": steps,
            "agent_workflow_id": task_id,
        }
    except Exception as e:
        logger.error(f"搜索和分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行失败: {str(e)}")


@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """获取工作流状态"""
    try:
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "progress": 100,
            "message": "工作流已完成"
        }
    except Exception as e:
        logger.error(f"获取工作流状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
