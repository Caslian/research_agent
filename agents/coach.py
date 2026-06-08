"""
InnoCore AI 写作助教 (Coach Agent) - 基于 LangChain 框架
负责风格迁移、实时润色、解释复杂概念
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field

from agents.base import BaseAgent
from core.database import db_manager
from core.vector_store import vector_store_manager
from core.exceptions import AgentException, TimeoutException

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务配置数据类"""
    task_type: str
    prompt_template: str
    default_result: Dict[str, Any]
    context_builder: Callable
    result_validator: Callable
    success_message: str


class CoachAgent(BaseAgent):
    """写作助教智能体"""
    
    # 常量定义
    MAX_CONTENT_LENGTH = 10000  # 最大内容长度
    DEFAULT_TIMEOUT = 60  # 默认超时时间（秒）
    
    def __init__(self, llm=None):
        super().__init__("Coach", llm)
        
        # 添加工具
        self.add_tool("explain_concept", self._explain_concept, "解释复杂概念")
        self.add_tool("polish_text", self._polish_text, "润色文本")
        self.add_tool("mimic_style", self._mimic_style, "模仿写作风格")
        self.add_tool("get_user_style", self._get_user_style, "获取用户写作风格")
        self.add_tool("suggest_improvements", self._suggest_improvements, "建议改进")
        
        # 缓存
        self._user_context_cache = {}
        self._cache_ttl = 300  # 缓存有效期 5 分钟
        
        # 初始化任务配置
        self._task_configs = self._initialize_task_configs()
    
    def _initialize_task_configs(self) -> Dict[str, TaskConfig]:
        """初始化任务配置"""
        return {
            "explain": TaskConfig(
                task_type="explain",
                prompt_template=self._build_explain_prompt,
                default_result={
                    "explanation": "Explanation generation failed",
                    "examples": [],
                    "importance": "",
                    "applications": []
                },
                context_builder=self._build_explain_context,
                result_validator=self._ensure_explanation_fields,
                success_message="解释任务完成"
            ),
            "polish": TaskConfig(
                task_type="polish",
                prompt_template=self._build_polish_prompt,
                default_result={
                    "polished_text": "",
                    "modifications": ["Polish failed: using original text"],
                    "style_suggestions": [],
                    "references": []
                },
                context_builder=self._build_polish_context,
                result_validator=self._ensure_polish_fields,
                success_message="润色任务完成"
            ),
            "mimic": TaskConfig(
                task_type="mimic",
                prompt_template=self._build_mimic_prompt,
                default_result={
                    "rewritten_text": "",
                    "style_analysis": "Mimic failed: using original text",
                    "mimic_techniques": [],
                    "reference_structures": []
                },
                context_builder=self._build_mimic_context,
                result_validator=self._ensure_mimic_fields,
                success_message="模仿任务完成"
            ),
            "suggest": TaskConfig(
                task_type="suggest",
                prompt_template=self._build_suggest_prompt,
                default_result={
                    "overall_evaluation": "Analysis failed",
                    "improvement_suggestions": [],
                    "grammar_issues": [],
                    "structure_suggestions": [],
                    "academic_improvements": []
                },
                context_builder=self._build_suggest_context,
                result_validator=self._ensure_suggest_fields,
                success_message="建议任务完成"
            )
        }
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行写作助教任务"""
        await self.validate_input(input_data)
        
        self.set_state("running")
        start_time = datetime.now()
        
        try:
            user_id = input_data["user_id"]
            task_type = input_data["task_type"]
            content = input_data["content"]
            context = input_data.get("context", {})
            
            # 输入验证
            self._validate_content(content)
            
            # 验证任务类型
            if task_type not in self._task_configs:
                raise AgentException(f"不支持的任务类型: {task_type}")
            
            logger.info(f"Coach Agent 开始执行任务: user_id={user_id}, task_type={task_type}")
            
            # 使用统一的任务处理框架
            result = await self._execute_task(user_id, content, context, task_type)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Coach Agent 任务完成: task_type={task_type}, elapsed={elapsed:.2f}s")
            
            self.set_state("completed")
            
            return {
                "status": "success",
                "task_type": task_type,
                "user_id": user_id,
                "result": result,
                "processing_time_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except (AgentException, TimeoutException):
            raise
        except Exception as e:
            self.set_state("error")
            logger.error(f"Coach Agent 执行失败: user_id={input_data.get('user_id')}, error={str(e)}", exc_info=True)
            raise AgentException(f"Coach Agent执行失败: {str(e)}")
    
    def get_required_fields(self) -> List[str]:
        """获取必需的输入字段"""
        return ["user_id", "task_type", "content"]
    
    def _validate_content(self, content: str):
        """验证内容输入"""
        if not content or not content.strip():
            raise AgentException("内容不能为空")
        
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise AgentException(
                f"内容长度超过限制: {len(content)} > {self.MAX_CONTENT_LENGTH}"
            )
    
    async def _execute_task(self, user_id: str, content: str, context: Dict, task_type: str) -> Dict[str, Any]:
        """
        统一的任务执行框架（模板方法）
        
        Args:
            user_id: 用户ID
            content: 待处理内容
            context: 上下文信息
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        config = self._task_configs[task_type]
        
        try:
            # 1. 构建上下文
            task_context = await config.context_builder(user_id, content, context)
            
            # 2. 构建 prompt
            prompt = config.prompt_template(content, task_context)
            
            # 3. 调用 LLM
            response = await self.think(prompt)
            
            # 4. 解析响应
            result = self._parse_llm_json_response(response, config.default_result.copy())
            
            # 5. 验证和补全结果字段
            result = config.result_validator(result, config.default_result)
            
            # 6. 记录成功日志
            logger.info(f"{config.success_message}: user_id={user_id}, content_length={len(content)}")
            self._add_to_history(f"完成{config.task_type}任务: {content[:50]}...")
            
            return result
            
        except TimeoutException:
            logger.warning(f"{config.task_type}任务超时: user_id={user_id}")
            default_result = config.default_result.copy()
            # 根据任务类型设置超时消息
            timeout_messages = {
                "explain": "Explanation generation timed out, please retry",
                "polish": "Polish timed out, please retry",
                "mimic": "Mimic timed out, please retry",
                "suggest": "Analysis timed out, please retry"
            }
            first_key = next(iter(default_result.keys()))
            default_result[first_key] = timeout_messages.get(config.task_type, "任务超时")
            return default_result
            
        except Exception as e:
            logger.error(f"{config.task_type}任务失败: user_id={user_id}, error={str(e)}", exc_info=True)
            default_result = config.default_result.copy()
            first_key = next(iter(default_result.keys()))
            default_result[first_key] = f"任务执行失败: {str(e)}"
            return default_result
    
    def _parse_llm_json_response(self, response: str, default_result: Dict) -> Dict:
        """
        解析 LLM 返回的 JSON 响应
        
        Args:
            response: LLM 返回的原始文本
            default_result: 解析失败时的默认返回值
            
        Returns:
            解析后的字典
        """
        if not response:
            logger.warning("LLM 返回空响应")
            return default_result

        # 阶段1: 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 阶段2: 提取 outermost { ... } 块（比代码块 regex 更可靠）
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 阶段3: 提取 markdown 代码块
        import re
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(
            f"无法解析 LLM JSON 响应，使用默认值。"
            f"响应前200字符: {response[:200]}"
        )
        return default_result
    
    async def _get_cached_user_context(self, user_id: str) -> Dict[str, Any]:
        """获取缓存的用户上下文"""
        import time
        current_time = time.time()
        
        if user_id in self._user_context_cache:
            cache_data = self._user_context_cache[user_id]
            if current_time - cache_data['timestamp'] < self._cache_ttl:
                return cache_data['data']
        
        # 缓存失效或未找到，重新获取
        user_context = await self._get_user_context(user_id)
        self._user_context_cache[user_id] = {
            'data': user_context,
            'timestamp': current_time
        }
        
        return user_context
    
    # ==================== 上下文构建器 ====================
    
    async def _build_explain_context(self, user_id: str, content: str, context: Dict) -> Dict:
        """构建解释任务的上下文"""
        user_context = await self._get_cached_user_context(user_id)
        return {
            "user_context": user_context,
            "additional_context": context
        }
    
    async def _build_polish_context(self, user_id: str, content: str, context: Dict) -> Dict:
        """构建润色任务的上下文"""
        user_style = await self._get_user_writing_style(user_id)
        style_references = await self._get_style_references(user_id, content)
        return {
            "user_style": user_style,
            "style_references": style_references,
            "additional_context": context
        }
    
    async def _build_mimic_context(self, user_id: str, content: str, context: Dict) -> Dict:
        """构建模仿任务的上下文"""
        target_style = context.get("target_style", "formal_academic")
        reference_papers = context.get("reference_papers", [])
        
        if not reference_papers:
            reference_papers = await self._get_user_top_papers(user_id, limit=3)
        
        return {
            "target_style": target_style,
            "reference_papers": reference_papers,
            "additional_context": context
        }
    
    async def _build_suggest_context(self, user_id: str, content: str, context: Dict) -> Dict:
        """构建建议任务的上下文"""
        user_writing_history = await self._get_user_writing_history(user_id)
        return {
            "user_writing_history": user_writing_history,
            "additional_context": context
        }
    
    # ==================== Prompt 构建器 ====================
    
    def _build_explain_prompt(self, content: str, task_context: Dict) -> str:
        """构建解释任务的 prompt"""
        user_context = task_context["user_context"]
        additional_context = task_context["additional_context"]
        
        return f"""Explain the following concept in simple, accessible terms. Return ONLY a JSON object (no markdown, no explanation).

Concept to explain:
{content}

Additional context:
{json.dumps(additional_context, ensure_ascii=False, indent=2)}

User research background:
{json.dumps(user_context, ensure_ascii=False, indent=2)}

IMPORTANT: Return a JSON object with exactly these keys:
- "explanation": clear explanation in simple terms
- "examples": list of concrete examples or analogies
- "importance": why this matters in the field
- "applications": list of potential applications

Example: {{"explanation": "This is...", "examples": ["Think of it as..."], "importance": "Critical for...", "applications": ["Used in..."]}}"""
    
    def _build_polish_prompt(self, content: str, task_context: Dict) -> str:
        """构建润色任务的 prompt"""
        user_style = task_context["user_style"]
        style_references = task_context["style_references"]
        additional_context = task_context["additional_context"]
        
        return f"""Polish the following text into academic English. Return ONLY a JSON object (no markdown, no explanation).

Original text:
{content}

User style preferences:
{json.dumps(user_style, ensure_ascii=False, indent=2)}

Context:
{json.dumps(additional_context, ensure_ascii=False, indent=2)}

IMPORTANT: Return a JSON object with exactly these keys:
- "polished_text": the polished English academic text
- "modifications": list of strings describing each change made
- "style_suggestions": list of style improvement suggestions
- "references": list of reference sources cited

Example: {{"polished_text": "This study examines...", "modifications": ["Replaced X with Y"], "style_suggestions": ["Use passive voice"], "references": ["Nature 2022"]}}"""
    
    def _build_mimic_prompt(self, content: str, task_context: Dict) -> str:
        """构建模仿任务的 prompt"""
        target_style = task_context["target_style"]
        reference_papers = task_context["reference_papers"]
        additional_context = task_context["additional_context"]
        
        return f"""Rewrite the following content in the target writing style. Return ONLY a JSON object (no markdown, no explanation).

Original:
{content}

Target style:
{target_style}

Reference papers:
{json.dumps(reference_papers, ensure_ascii=False, indent=2)}

Context:
{json.dumps(additional_context, ensure_ascii=False, indent=2)}

IMPORTANT: Return a JSON object with exactly these keys:
- "rewritten_text": the text rewritten in the target style
- "style_analysis": explanation of how the target style was achieved
- "mimic_techniques": list of specific techniques used
- "reference_structures": list of sentence structures borrowed from references

Example: {{"rewritten_text": "...", "style_analysis": "...", "mimic_techniques": ["..."], "reference_structures": ["..."]}}"""
    
    def _build_suggest_prompt(self, content: str, task_context: Dict) -> str:
        """构建建议任务的 prompt"""
        user_writing_history = task_context["user_writing_history"]
        additional_context = task_context["additional_context"]
        
        return f"""Review the following text and suggest improvements. Return ONLY a JSON object (no markdown, no explanation).

Text:
{content}

User writing history:
{json.dumps(user_writing_history, ensure_ascii=False, indent=2)}

Context:
{json.dumps(additional_context, ensure_ascii=False, indent=2)}

IMPORTANT: Return a JSON object with exactly these keys:
- "overall_evaluation": brief overall assessment
- "improvement_suggestions": list of specific improvement suggestions ranked by importance
- "grammar_issues": list of grammar or expression issues found
- "structure_suggestions": list of structure optimization suggestions
- "academic_improvements": list of academic expression improvements

Example: {{"overall_evaluation": "Well written but...", "improvement_suggestions": ["Add more examples..."], "grammar_issues": ["Passive voice overuse..."], "structure_suggestions": ["Reorder section 2..."], "academic_improvements": ["Use more precise terminology..."]}}"""
    
    # ==================== 结果验证器 ====================
    
    def _ensure_explanation_fields(self, result: Dict, default: Dict) -> Dict:
        """确保解释结果包含所有必需字段"""
        for key in default.keys():
            if key not in result:
                result[key] = default[key]
        return result
    
    def _ensure_polish_fields(self, result: Dict, default: Dict) -> Dict:
        """确保润色结果包含所有必需字段"""
        for key in default.keys():
            if key not in result:
                result[key] = default[key]
        return result
    
    def _ensure_mimic_fields(self, result: Dict, default: Dict) -> Dict:
        """确保模仿结果包含所有必需字段"""
        for key in default.keys():
            if key not in result:
                result[key] = default[key]
        return result
    
    def _ensure_suggest_fields(self, result: Dict, default: Dict) -> Dict:
        """确保建议结果包含所有必需字段"""
        for key in default.keys():
            if key not in result:
                result[key] = default[key]
        return result
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """获取用户的研究背景"""
        # 非 UUID 格式的 user_id（如 'default'）不查询数据库
        import re as _re
        uuid_pattern = _re.compile(
            r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        )
        if not user_id or not uuid_pattern.match(user_id):
            return {}
        try:
            user = await db_manager.get_user(user_id)
            if user:
                return user.get("profile", {})
            return {}
        except Exception as e:
            logger.warning(f"获取用户上下文失败: user_id={user_id}, error={str(e)}")
            return {}
    
    async def _get_user_writing_style(self, user_id: str) -> Dict[str, Any]:
        """获取用户写作风格偏好"""
        user_context = await self._get_cached_user_context(user_id)
        return user_context.get("writing_style", {
            "tone": "formal",
            "complexity": "medium",
            "preferred_journals": ["Nature", "Science"],
            "language": "english"
        })
    
    async def _get_style_references(self, user_id: str, content: str) -> List[Dict[str, Any]]:
        """获取风格参考"""
        try:
            # 搜索用户库中的相关论文
            search_results = await vector_store_manager.hybrid_search(
                query=content,
                user_id=user_id,
                top_k=3,
                include_l2=True,
                include_l1=False
            )
            
            references = []
            for result in search_results:
                payload = result["payload"]
                references.append({
                    "title": payload.get("title", ""),
                    "abstract": payload.get("abstract", "")[:200],
                    "similarity": result["score"]
                })
            
            return references
            
        except Exception as e:
            logger.warning(f"获取风格参考失败: user_id={user_id}, error={str(e)}")
            return []
    
    async def _get_user_top_papers(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """获取用户评分最高的论文"""
        try:
            user_papers = await db_manager.get_user_papers(user_id, limit=limit)
            
            top_papers = []
            for paper in user_papers:
                top_papers.append({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "")[:300],
                    "rating": paper.get("rating", 0),
                    "authors": paper.get("authors", [])
                })
            
            return top_papers
            
        except Exception as e:
            logger.warning(f"获取用户论文失败: user_id={user_id}, error={str(e)}")
            return []
    
    async def _get_user_writing_history(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户写作历史"""
        try:
            # TODO: 从数据库获取真实的写作历史记录
            # 暂时返回模拟数据
            return [
                {
                    "date": "2024-01-01",
                    "content_type": "abstract",
                    "word_count": 200,
                    "feedback_score": 4.5
                }
            ]
        except Exception as e:
            logger.warning(f"获取写作历史失败: user_id={user_id}, error={str(e)}")
            return []
    
    # 工具方法
    async def _explain_concept(self, concept: str, context: Dict = None) -> Dict:
        """解释概念工具"""
        ctx = context or {}
        return await self._handle_legacy_task("explain", ctx.get("user_id", ""), concept, ctx)
    
    async def _polish_text(self, text: str, context: Dict = None) -> Dict:
        """润色文本工具"""
        ctx = context or {}
        return await self._handle_legacy_task("polish", ctx.get("user_id", ""), text, ctx)
    
    async def _mimic_style(self, text: str, target_style: str, context: Dict = None) -> Dict:
        """模仿风格工具"""
        ctx = context or {}
        ctx["target_style"] = target_style
        return await self._handle_legacy_task("mimic", ctx.get("user_id", ""), text, ctx)
    
    async def _get_user_style(self, user_id: str) -> Dict:
        """获取用户风格工具"""
        return await self._get_user_writing_style(user_id)
    
    async def _suggest_improvements(self, text: str, context: Dict = None) -> Dict:
        """建议改进工具"""
        ctx = context or {}
        return await self._handle_legacy_task("suggest", ctx.get("user_id", ""), text, ctx)
    
    async def _handle_legacy_task(self, task_type: str, user_id: str, content: str, context: Dict) -> Dict:
        """
        兼容旧的工具调用方式
        直接调用统一的任务执行框架
        """
        if task_type not in self._task_configs:
            raise AgentException(f"不支持的任务类型: {task_type}")
        
        return await self._execute_task(user_id, content, context, task_type)
    
    def clear_cache(self):
        """清除缓存"""
        self._user_context_cache.clear()
        logger.info("Coach Agent 缓存已清除")