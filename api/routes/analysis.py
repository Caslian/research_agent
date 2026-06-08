"""
分析相关API路由 — 委托 MinerAgent 执行论文分析
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
import os
from core.config import get_config
from utils.pdf_parser import pdf_parser
from agents.controller import agent_controller, TaskType

logger = logging.getLogger(__name__)
router = APIRouter()

config = get_config()


def _model_info():
    """返回当前 Miner Agent 使用的模型信息"""
    return {
        "agent": "miner",
        "llm_model": config.llm.model_name,
        "llm_provider": config.llm.provider.value,
        "llm_base_url": config.llm.base_url or "OpenAI 默认"
    }


# Pydantic模型
class AnalysisRequest(BaseModel):
    paper_id: str
    user_id: Optional[str] = None
    analysis_type: str = "full"


class ComparisonRequest(BaseModel):
    paper_ids: List[str]
    user_id: Optional[str] = None
    comparison_aspects: List[str] = ["method", "results", "innovation"]


class InnovationSearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    search_scope: str = "both"
    top_k: int = 10


class PaperAnalysisRequest(BaseModel):
    paper_url: str
    analysis_type: str = "summary"


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_paper(request: PaperAnalysisRequest):
    """分析论文 — 委托 MinerAgent 通过 LangGraph ReAct 执行"""
    try:
        paper_url = request.paper_url.strip()

        # 本地 PDF：先解析，再委托 MinerAgent（title + abstract 模式）
        if paper_url.startswith('/uploads/') or paper_url.endswith('.pdf'):
            logger.info(f"检测到本地 PDF 文件: {paper_url}")

            if paper_url.startswith('/uploads/'):
                file_path = os.path.join('downloads', paper_url.replace('/uploads/', ''))
            else:
                file_path = paper_url

            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"PDF 文件不存在: {paper_url}")

            logger.info(f"开始解析 PDF 文件: {file_path}")
            pdf_result = await pdf_parser.parse_pdf(file_path)

            if not pdf_result.get("success"):
                raise HTTPException(status_code=500, detail=pdf_result.get("error", "PDF 解析失败"))

            title = pdf_result.get("title", "未知标题")
            authors = pdf_result.get("authors", ["未知作者"])
            abstract = pdf_result.get("abstract", "")
            full_text = pdf_result.get("full_text", "")
            text_for_analysis = full_text[:8000] if len(full_text) > 8000 else full_text

            logger.info(f"[Agent] 提交 MinerAgent 分析任务: title={title[:50]}...")
            task_id = await agent_controller.submit_task(
                TaskType.PAPER_ANALYSIS,
                {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "full_text": text_for_analysis,
                    "analysis_type": request.analysis_type,
                }
            )
            result = await agent_controller.execute_task(task_id)
            analysis_report = result.get("analysis_report", {})

            return {
                "success": True,
                "model_info": _model_info(),
                "paper_info": {
                    "id": "local_pdf",
                    "title": title,
                    "authors": authors,
                    "published_date": "N/A",
                    "url": paper_url,
                    "categories": ["本地文件"],
                    "page_count": pdf_result.get("page_count", 0),
                    "word_count": pdf_result.get("word_count", 0)
                },
                "analysis_type": request.analysis_type,
                "analysis": analysis_report.get("analysis", ""),
                "abstract": abstract,
                "agent_task_id": task_id,
            }

        # ArXiv URL：直接委托 MinerAgent（paper_url 模式）
        logger.info(f"[Agent] 提交 MinerAgent 分析任务: url={paper_url}")
        task_id = await agent_controller.submit_task(
            TaskType.PAPER_ANALYSIS,
            {
                "paper_url": paper_url,
                "analysis_type": request.analysis_type,
            }
        )
        result = await agent_controller.execute_task(task_id)
        analysis_report = result.get("analysis_report", {})
        paper_info = analysis_report.get("paper_info", {})

        return {
            "success": True,
            "model_info": _model_info(),
            "paper_info": {
                "id": analysis_report.get("paper_id", ""),
                "title": paper_info.get("title", ""),
                "authors": paper_info.get("authors", []),
                "published_date": "N/A",
                "url": paper_url,
                "categories": []
            },
            "analysis_type": request.analysis_type,
            "analysis": analysis_report.get("analysis", ""),
            "abstract": "",
            "agent_task_id": task_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"论文分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/compare", response_model=Dict[str, Any])
async def compare_papers(request: ComparisonRequest):
    """对比多篇论文"""
    try:
        comparison_result = {
            "paper_ids": request.paper_ids,
            "comparison_aspects": request.comparison_aspects,
            "similarities": ["相似点1", "相似点2"],
            "differences": ["差异点1", "差异点2"],
            "innovation_gaps": ["创新空白1", "创新空白2"],
            "recommendations": ["建议1", "建议2"]
        }
        return {"success": True, "result": comparison_result}
    except Exception as e:
        logger.error(f"论文对比失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/innovation/search", response_model=Dict[str, Any])
async def search_innovation_opportunities(request: InnovationSearchRequest):
    """搜索创新机会"""
    try:
        innovation_results = {
            "query": request.query,
            "opportunities": [
                {"title": "创新机会1", "description": "基于当前研究的创新方向", "related_papers": ["paper1", "paper2"], "confidence": 0.85},
                {"title": "创新机会2", "description": "另一个潜在的研究方向", "related_papers": ["paper3", "paper4"], "confidence": 0.72}
            ],
            "research_gaps": ["研究空白1", "研究空白2"],
            "future_directions": ["未来方向1", "未来方向2"]
        }
        return {"success": True, "result": innovation_results}
    except Exception as e:
        logger.error(f"创新机会搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{paper_id}/summary")
async def get_paper_summary(paper_id: str, user_id: Optional[str] = None):
    """获取论文摘要"""
    try:
        summary = {
            "paper_id": paper_id,
            "summary": "这是一篇关于...的论文，主要贡献包括...",
            "key_contributions": ["贡献1", "贡献2", "贡献3"],
            "methodology": "论文采用的方法是...",
            "results": "实验结果表明...",
            "limitations": "研究的局限性包括...",
            "future_work": "未来工作方向..."
        }
        return {"success": True, "summary": summary}
    except Exception as e:
        logger.error(f"获取论文摘要失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{paper_id}/innovations")
async def get_paper_innovations(paper_id: str, user_id: Optional[str] = None):
    """获取论文创新点"""
    try:
        innovations = {
            "paper_id": paper_id,
            "innovations": [
                {"aspect": "方法创新", "description": "提出了新的方法...", "novelty": "high", "impact": "significant"},
                {"aspect": "理论创新", "description": "在理论上有所突破...", "novelty": "medium", "impact": "moderate"}
            ],
            "comparison_with_prior_work": "与之前的工作相比...",
            "potential_applications": ["应用1", "应用2"]
        }
        return {"success": True, "innovations": innovations}
    except Exception as e:
        logger.error(f"获取论文创新点失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/insights")
async def get_user_insights(user_id: str):
    """获取用户研究洞察"""
    try:
        insights = {
            "user_id": user_id,
            "research_interests": ["兴趣1", "兴趣2"],
            "reading_patterns": {"papers_read": 50, "favorite_topics": ["主题1", "主题2"], "reading_frequency": "daily"},
            "knowledge_gaps": ["知识空白1", "知识空白2"],
            "research_suggestions": [{"topic": "建议研究方向1", "reason": "基于您的阅读历史...", "related_papers": ["paper1", "paper2"]}],
            "skill_assessment": {"technical_skills": ["技能1", "技能2"], "writing_skills": ["写作技能1", "写作技能2"], "improvement_areas": ["改进领域1", "改进领域2"]}
        }
        return {"success": True, "insights": insights}
    except Exception as e:
        logger.error(f"获取用户研究洞察失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, Any])
async def batch_analyze_papers(paper_ids: List[str], user_id: Optional[str] = None):
    """批量分析论文 — 委托 MinerAgent"""
    try:
        results = []
        for paper_id in paper_ids:
            try:
                task_id = await agent_controller.submit_task(
                    TaskType.PAPER_ANALYSIS,
                    {"paper_id": paper_id, "user_id": user_id, "analysis_type": "quick"}
                )
                result = await agent_controller.execute_task(task_id)
                results.append({"paper_id": paper_id, "task_id": task_id, "success": True, "result": result})
            except Exception as e:
                results.append({"paper_id": paper_id, "success": False, "error": str(e)})

        return {
            "success": True,
            "total_papers": len(paper_ids),
            "successful_analyses": sum(1 for r in results if r["success"]),
            "results": results
        }
    except Exception as e:
        logger.error(f"批量分析论文失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-pdf", response_model=Dict[str, Any])
async def upload_pdf_for_analysis(file: UploadFile = File(...)):
    """上传 PDF 文件并解析"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持 PDF 文件")

        logger.info(f"接收到 PDF 文件: {file.filename}")
        pdf_bytes = await file.read()
        pdf_result = await pdf_parser.parse_pdf_from_bytes(pdf_bytes, file.filename)

        if not pdf_result.get("success"):
            raise HTTPException(status_code=500, detail=pdf_result.get("error", "PDF 解析失败"))

        os.makedirs("downloads", exist_ok=True)
        file_path = os.path.join("downloads", file.filename)
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        logger.info(f"PDF 文件已保存: {file_path}")

        return {
            "success": True,
            "filename": file.filename,
            "file_path": f"/uploads/{file.filename}",
            "title": pdf_result.get("title", "未知标题"),
            "authors": pdf_result.get("authors", ["未知作者"]),
            "abstract": pdf_result.get("abstract", "")[:500],
            "page_count": pdf_result.get("page_count", 0),
            "word_count": pdf_result.get("word_count", 0),
            "message": "PDF 文件上传并解析成功，可以使用返回的 file_path 进行分析"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF 上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
