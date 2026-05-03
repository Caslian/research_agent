"""
论文相关API路由
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
import arxiv
from datetime import datetime
from core.config import get_config

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
    """搜索论文 - 使用真实的 ArXiv API"""
    try:
        papers = []
        
        if request.source == "arxiv" or request.source == "all":
            # 使用 ArXiv API 搜索
            logger.info(f"正在搜索 ArXiv: {request.keywords}")
            
            # 构建搜索查询（修复：使用 AND 逻辑组合关键词）
            arxiv_query = _build_arxiv_query(request.keywords)
            
            if not arxiv_query:
                return {
                    "success": True,
                    "papers": [],
                    "total_found": 0,
                    "keywords": request.keywords,
                    "source": request.source,
                    "message": "关键词不能为空"
                }
            
            logger.info(f"ArXiv 查询语句: {arxiv_query}")
            
            search = arxiv.Search(
                query=arxiv_query,
                max_results=request.limit,
                # 修复：优先按相关性排序（Relevance），而非最新时间
                # 最新论文（SubmittedDate）≠ 最相关，用 Relevance 确保关键词匹配度最高的结果排前面
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # 获取搜索结果
            for result in search.results():
                paper = {
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary.replace('\n', ' ').strip(),
                    "url": result.entry_id,
                    "published_date": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "primary_category": result.primary_category
                }
                papers.append(paper)
            
            logger.info(f"找到 {len(papers)} 篇论文")
        
        # 如果没有找到结果，返回提示
        if not papers:
            return {
                "success": True,
                "model_info": _model_info(),
                "papers": [],
                "total_found": 0,
                "keywords": request.keywords,
                "source": request.source,
                "message": "未找到相关论文，请尝试其他关键词或缩短关键词"
            }
        
        return {
            "success": True,
            "model_info": _model_info(),
            "papers": papers,
            "total_found": len(papers),
            "keywords": request.keywords,
            "source": request.source
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
