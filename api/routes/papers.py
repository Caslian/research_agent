"""
论文相关API路由 — 委托 HunterAgent 执行搜索
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from core.config import get_config
from agents.controller import agent_controller, TaskType

logger = logging.getLogger(__name__)
router = APIRouter()

# 初始化配置
config = get_config()


def _model_info():
    """返回当前 Hunter Agent 使用的模型信息"""
    return {
        "agent": "hunter",
        "llm_model": config.llm.model_name,
        "llm_provider": config.llm.provider.value,
        "llm_base_url": config.llm.base_url or "OpenAI 默认"
    }

# Pydantic模型
class PaperSearchRequest(BaseModel):
    keywords: str
    source: str = "arxiv"
    limit: int = 10

class PaperResponse(BaseModel):
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: str


def _build_arxiv_query(keywords: str) -> str:
    """
    构建 ArXiv 查询语句，支持多关键词 AND 组合
    
    Args:
        keywords: 用户输入的关键词，如 "wifi fingerprint" 或 "machine learning"
        
    Returns:
        ArXiv 查询语句，如 "all:wifi AND all:fingerprint"
        
    修复说明：
    - 原始版本直接把 keywords 当作短语查询，导致 "wifi fingerprint"
      被当成一个短语，匹配到大量无关论文（该短语在量子物理论文摘要中
      偶遇命中了 6333 篇，取最新 5 篇全是量子方向）
    - 现在改为：空格分隔关键词，生成 "all:词1 AND all:词2" 格式，
      强制要求每篇论文在任意字段中同时包含所有关键词
    """
    if not keywords or not keywords.strip():
        return ""
    
    # 清理关键词：去除多余空格、转小写
    cleaned = keywords.strip().lower()
    
    # 按空格拆分关键词
    parts = [p.strip() for p in cleaned.split() if p.strip()]
    
    if not parts:
        return ""
    
    if len(parts) == 1:
        # 只有一个词：用 all: 字段搜索
        return f"all:{parts[0]}"
    
    # 多个词：用 AND 组合，确保每篇论文同时包含所有关键词
    query_parts = [f"all:{p}" for p in parts]
    return " AND ".join(query_parts)


@router.post("/search", response_model=Dict[str, Any])
async def search_papers(request: PaperSearchRequest):
    """搜索论文 — 委托 HunterAgent 通过 LangGraph ReAct 执行"""
    try:
        keywords = [kw.strip() for kw in request.keywords.split() if kw.strip()]
        if not keywords:
            return {
                "success": True,
                "model_info": _model_info(),
                "papers": [],
                "total_found": 0,
                "keywords": request.keywords,
                "source": request.source,
                "message": "关键词不能为空"
            }

        sources = ["arxiv"] if request.source in ("arxiv", "all") else [request.source]

        logger.info(f"[Agent] 提交 HunterAgent 搜索任务: keywords={keywords}, sources={sources}")
        task_id = await agent_controller.submit_task(
            TaskType.PAPER_HUNTING,
            {
                "keywords": keywords,
                "max_papers": request.limit,
                "sources": sources,
            }
        )
        result = await agent_controller.execute_task(task_id)

        controller_papers = result.get("papers_found", [])
        papers = []
        for p in controller_papers:
            papers.append({
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "abstract": (p.get("abstract", "") or "").replace('\n', ' ').strip(),
                "url": p.get("pdf_url", "").replace('/pdf/', '/abs/') if p.get("pdf_url") else "",
                "published_date": p.get("published", ""),
                "pdf_url": p.get("pdf_url", ""),
                "categories": p.get("categories", []),
            })

        stats = result.get("statistics", {})
        return {
            "success": True,
            "model_info": _model_info(),
            "papers": papers,
            "total_found": stats.get("total_found", len(papers)),
            "keywords": request.keywords,
            "source": request.source,
            "agent_task_id": task_id,
        }

    except Exception as e:
        logger.error(f"论文搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.post("/upload", response_model=Dict[str, Any])
async def upload_paper(file: UploadFile = File(...)):
    """上传论文PDF"""
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件")
        
        # 模拟文件上传
        file_url = f"/uploads/{file.filename}"
        
        return {
            "success": True,
            "file_url": file_url,
            "filename": file.filename,
            "size": getattr(file, 'size', 0),
            "message": "文件上传成功"
        }
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
