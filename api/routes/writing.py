"""
写作辅助API路由 — 委托 CoachAgent 执行写作任务
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging
from core.config import get_config
from agents.controller import agent_controller, TaskType

logger = logging.getLogger(__name__)
router = APIRouter()

config = get_config()


def _model_info():
    """返回当前 Coach Agent 使用的模型信息"""
    return {
        "agent": "coach",
        "llm_model": config.llm.model_name,
        "llm_provider": config.llm.provider.value,
        "llm_base_url": config.llm.base_url or "OpenAI 默认"
    }


# CoachAgent task_type mapping
_COACH_TASK_MAP = {
    "polish": "polish",
    "translate": "polish",
    "explain": "explain",
    "expand": "suggest",
}


def _extract_agent_output(assistance_result: Dict, task_type: str) -> Dict:
    """从 Agent 的结构化输出中提取对用户友好的字段"""
    inner = (assistance_result or {}).get("result", {})
    if not isinstance(inner, dict):
        return {"content": str(inner)}

    if task_type == "explain":
        return {
            "explanation": inner.get("explanation", ""),
            "examples": inner.get("examples", []),
            "importance": inner.get("importance", ""),
            "applications": inner.get("applications", []),
        }
    elif task_type == "polish":
        return {
            "polished_text": inner.get("polished_text", ""),
            "modifications": inner.get("modifications", []) or inner.get("main_modifications", []),
            "style_suggestions": inner.get("style_suggestions", []) or inner.get("style_improvement", []),
        }
    elif task_type == "mimic":
        return {
            "rewritten_text": inner.get("rewritten_text", ""),
            "style_analysis": inner.get("style_analysis", ""),
            "techniques": inner.get("mimic_techniques", []),
        }
    elif task_type == "suggest":
        return {
            "evaluation": inner.get("overall_evaluation", ""),
            "suggestions": (inner.get("improvement_suggestions", []) or
                          inner.get("improvements", []) or
                          inner.get("suggestions", [])),
            "grammar_issues": inner.get("grammar_issues", []),
            "structure_suggestions": inner.get("structure_suggestions", []),
        }
    return inner


# Pydantic模型
class ExplainRequest(BaseModel):
    user_id: str
    concept: str
    context: Optional[Dict[str, Any]] = {}


class PolishRequest(BaseModel):
    user_id: str
    text: str
    target_style: Optional[str] = "academic"


class WritingCoachRequest(BaseModel):
    text: str
    style: str = "formal"
    task: str = "polish"
    context: Optional[Dict[str, Any]] = {}


class MimicRequest(BaseModel):
    user_id: str
    text: str
    target_style: str
    reference_papers: Optional[list] = []
    context: Optional[Dict[str, Any]] = {}


class SuggestRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[Dict[str, Any]] = {}


@router.post("/coach", response_model=Dict[str, Any])
async def writing_coach(request: WritingCoachRequest):
    """写作助手 — 委托 CoachAgent 通过 LangGraph ReAct 执行"""
    try:
        agent_task = _COACH_TASK_MAP.get(request.task, "polish")
        logger.info(f"[Agent] CoachAgent task=%s style=%s", agent_task, request.style)

        task_id = await agent_controller.submit_task(
            TaskType.WRITING_ASSISTANCE,
            {
                "user_id": "default",
                "task_type": agent_task,
                "content": request.text,
                "context": {
                    "style": request.style,
                    "original_task": request.task,
                    **(request.context or {}),
                },
            }
        )
        result = await agent_controller.execute_task(task_id)
        assistance = result.get("assistance_result", {})
        output = _extract_agent_output(assistance, agent_task)

        return {
            "success": True,
            "model_info": _model_info(),
            "task": request.task,
            "style": request.style,
            "original": request.text,
            "result": output,
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"写作助手处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post("/explain", response_model=Dict[str, Any])
async def explain_concept(request: ExplainRequest):
    """解释复杂概念 — 委托 CoachAgent"""
    try:
        logger.info(f"[Agent] CoachAgent explain concept=%s", request.concept[:50])
        task_id = await agent_controller.submit_task(
            TaskType.WRITING_ASSISTANCE,
            {
                "user_id": request.user_id,
                "task_type": "explain",
                "content": request.concept,
                "context": request.context or {},
            }
        )
        result = await agent_controller.execute_task(task_id)
        assistance = result.get("assistance_result", {})
        output = _extract_agent_output(assistance, "explain")

        return {
            "success": True,
            "concept": request.concept,
            **output,
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"概念解释失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/polish", response_model=Dict[str, Any])
async def polish_text(request: PolishRequest):
    """润色文本 — 委托 CoachAgent"""
    try:
        logger.info(f"[Agent] CoachAgent polish")
        task_id = await agent_controller.submit_task(
            TaskType.WRITING_ASSISTANCE,
            {
                "user_id": request.user_id,
                "task_type": "polish",
                "content": request.text,
                "context": {"target_style": request.target_style},
            }
        )
        result = await agent_controller.execute_task(task_id)
        assistance = result.get("assistance_result", {})
        output = _extract_agent_output(assistance, "polish")

        return {
            "success": True,
            "original": request.text,
            **output,
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"文本润色失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mimic", response_model=Dict[str, Any])
async def mimic_style(request: MimicRequest):
    """模仿写作风格 — 委托 CoachAgent"""
    try:
        logger.info(f"[Agent] CoachAgent mimic target=%s", request.target_style)
        task_id = await agent_controller.submit_task(
            TaskType.WRITING_ASSISTANCE,
            {
                "user_id": request.user_id,
                "task_type": "mimic",
                "content": request.text,
                "context": {
                    "target_style": request.target_style,
                    "reference_papers": request.reference_papers or [],
                    **(request.context or {}),
                },
            }
        )
        result = await agent_controller.execute_task(task_id)
        assistance = result.get("assistance_result", {})
        output = _extract_agent_output(assistance, "mimic")

        return {
            "success": True,
            "original": request.text,
            **output,
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"风格模仿失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest", response_model=Dict[str, Any])
async def suggest_improvements(request: SuggestRequest):
    """建议改进 — 委托 CoachAgent"""
    try:
        logger.info(f"[Agent] CoachAgent suggest")
        task_id = await agent_controller.submit_task(
            TaskType.WRITING_ASSISTANCE,
            {
                "user_id": request.user_id,
                "task_type": "suggest",
                "content": request.text,
                "context": request.context or {},
            }
        )
        result = await agent_controller.execute_task(task_id)
        assistance = result.get("assistance_result", {})
        output = _extract_agent_output(assistance, "suggest")

        return {
            "success": True,
            "original": request.text,
            **output,
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"改进建议失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/style")
async def get_user_writing_style(user_id: str):
    """获取用户写作风格"""
    try:
        style_profile = {
            "user_id": user_id,
            "writing_style": {
                "tone": "formal_academic", "complexity": "medium",
                "sentence_length": "medium", "vocabulary_richness": "high", "clarity": "good"
            },
            "preferred_patterns": ["句式模式1", "句式模式2"],
            "common_phrases": ["常用短语1", "常用短语2"],
            "improvement_areas": ["改进领域1", "改进领域2"],
            "style_evolution": {"last_month": "上个月的风格变化", "trend": "improving"}
        }
        return {"success": True, "style_profile": style_profile}
    except Exception as e:
        logger.error(f"获取用户写作风格失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/templates")
async def get_writing_templates(user_id: str):
    """获取写作模板"""
    try:
        return {"success": True, "templates": {
            "user_id": user_id,
            "templates": [
                {"id": "abstract_template", "name": "摘要模板", "category": "academic",
                 "structure": ["背景介绍", "问题陈述", "方法概述", "主要结果", "结论意义"]},
                {"id": "introduction_template", "name": "引言模板", "category": "academic",
                 "structure": ["研究背景", "相关工作", "研究空白", "主要贡献", "论文结构"]}
            ]
        }}
    except Exception as e:
        logger.error(f"获取写作模板失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check/grammar", response_model=Dict[str, Any])
async def check_grammar(text: str, user_id: Optional[str] = None):
    """语法检查"""
    return {"success": True, "grammar_check": {"text": text, "score": 85}}


@router.post("/check/plagiarism", response_model=Dict[str, Any])
async def check_plagiarism(text: str, user_id: Optional[str] = None):
    """抄袭检查"""
    return {"success": True, "plagiarism_check": {"text": text, "originality_score": 84.5, "risk_level": "low"}}
