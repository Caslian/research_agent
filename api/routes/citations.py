"""
引用校验API路由 — 委托 ValidatorAgent 执行引用校验与生成
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
    """返回当前 Validator Agent 使用的模型信息"""
    return {
        "agent": "validator",
        "llm_model": config.llm.model_name,
        "llm_provider": config.llm.provider.value,
        "llm_base_url": config.llm.base_url or "OpenAI 默认"
    }


# Pydantic模型
class CitationValidationRequest(BaseModel):
    citation: str
    format: str = "bibtex"


class CitationGenerateRequest(BaseModel):
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    format: str = "bibtex"


@router.post("/validate", response_model=Dict[str, Any])
async def validate_citation(request: CitationValidationRequest):
    """校验引用格式 — 委托 ValidatorAgent 通过 LangGraph ReAct 执行"""
    try:
        logger.info(f"[Agent] 提交 ValidatorAgent 校验任务: citation={request.citation[:80]}...")

        task_id = await agent_controller.submit_task(
            TaskType.CITATION_VALIDATION,
            {
                "citation_text": request.citation,
                "formats": [request.format, "bibtex", "apa", "ieee", "mla"],
                "verify_external": True,
            }
        )
        result = await agent_controller.execute_task(task_id)
        validation = result.get("validation_result", {})
        citations = validation.get("citations", {})
        verification = validation.get("verification", {})

        formatted = citations.get(request.format, request.citation)

        return {
            "success": True,
            "model_info": _model_info(),
            "original_citation": request.citation,
            "formatted_citation": formatted,
            "format": request.format,
            "verified": verification.get("status") == "verified",
            "metadata": validation.get("paper_info"),
            "all_formats": {k: v for k, v in citations.items() if k != "metadata"},
            "verification_status": validation.get("verification_status", "unknown"),
            "warnings": [] if verification.get("status") == "verified"
                else ["无法自动验证引用，已返回原始格式。建议提供包含 DOI 的引用信息以获得更准确的结果。"],
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"引用校验失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"校验失败: {str(e)}")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_citation(request: CitationGenerateRequest):
    """生成引用格式 — 委托 ValidatorAgent"""
    try:
        logger.info(f"[Agent] 提交 ValidatorAgent 生成任务: format={request.format}")

        paper_info = {}
        if request.title:
            paper_info["title"] = request.title
        if request.authors:
            paper_info["authors"] = [a.strip() for a in request.authors.split(",")]
        if request.year:
            paper_info["year"] = request.year
        if request.journal:
            paper_info["journal"] = request.journal
        if request.doi:
            paper_info["doi"] = request.doi

        task_id = await agent_controller.submit_task(
            TaskType.CITATION_VALIDATION,
            {
                "paper_info": paper_info,
                "formats": [request.format, "bibtex", "apa", "ieee", "mla"],
                "verify_external": bool(request.doi),
            }
        )
        result = await agent_controller.execute_task(task_id)
        validation = result.get("validation_result", {})
        citations = validation.get("citations", {})

        citation = citations.get(request.format, "")

        return {
            "success": True,
            "citation": citation,
            "format": request.format,
            "all_formats": {k: v for k, v in citations.items() if k != "metadata"},
            "metadata": validation.get("paper_info", paper_info),
            "agent_task_id": task_id,
        }
    except Exception as e:
        logger.error(f"引用生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.get("/formats", response_model=Dict[str, Any])
async def get_citation_formats():
    """获取支持的引用格式"""
    try:
        formats = {
            "bibtex": {"name": "BibTeX", "description": "常用于LaTeX文档的引用格式",
                       "example": "@article{key, title={Title}, author={Author}, year={2024}}"},
            "apa": {"name": "APA", "description": "美国心理学会格式，常用于社会科学",
                    "example": "Author, A. (2024). Title. *Journal*, 1(1), 1-10."},
            "ieee": {"name": "IEEE", "description": "电气电子工程师学会格式，常用于工程技术",
                     "example": "[1] A. Author, \"Title,\" *Journal*, vol. 1, no. 1, pp. 1-10, 2024."},
            "mla": {"name": "MLA", "description": "现代语言学会格式，常用于人文学科",
                    "example": "Author. \"Title.\" *Journal*, vol. 1, no. 1, 2024, pp. 1-10."}
        }
        return {"success": True, "formats": formats, "total": len(formats)}
    except Exception as e:
        logger.error(f"获取引用格式失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")
