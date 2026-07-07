"""
论文相关API路由 — 委托 HunterAgent 执行搜索

重构后流程（v2）：
  - POST /search              搜索 + 下载 PDF + LLM 生成笔记（不下载到正式目录，不入库）
  - POST /upload              上传 PDF（旧接口保留兼容）
  - POST /{paper_id}/download-to-kb   用户点击"下载到知识库"后调用，复用 cache
"""
import hashlib
import logging
import re
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from agents.controller import agent_controller, TaskType
from core.config import get_config
from core.database import db_manager
from core.exceptions import InnoCoreException
from core.hunter_cache import hunter_cache
from core.knowledge_base_manager import knowledge_base_manager
from utils.pdf_parser import pdf_parser

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
        "llm_base_url": config.llm.base_url or "OpenAI 默认",
    }


# ============================================================
# Pydantic 模型
# ============================================================

class PaperSearchRequest(BaseModel):
    keywords: str
    source: str = "arxiv"
    limit: int = 10
    kb_id: Optional[str] = None     # 新增：用于 KB 去重 + 后续入库目标
    user_id: Optional[str] = None   # 新增：用户标识
    days_back: int = 7


class PaperResponse(BaseModel):
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: str
    # ===== v2 新增字段 =====
    note: str = ""                  # LLM 笔记
    note_status: str = "pending"    # pending / ready / failed
    pdf_url: str = ""               # 论文 PDF 链接（供入库兜底下载）
    relevance_score: float = 0.0    # 相关性评分
    pdf_cached: bool = False        # PDF 是否已缓存（用户入库时复用）


class DownloadToKBRequest(BaseModel):
    user_id: str
    kb_id: str
    pdf_url: str = ""               # 来自 hunter 搜索结果（cache miss 时兜底下载）
    note: Optional[str] = None      # 笔记文本（cache miss 时兜底生成）
    title: Optional[str] = None
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None  # 论文 arxiv id（用于 cache key）


class MarkReadRequest(BaseModel):
    """标记论文已读请求。

    设计要点：hunter 阶段论文尚未入库（无 paper_id），所以主标识用
    paper_key = f"{source}:{id}"（arxiv/IEEE id 前缀化避免冲突）。
    若论文已入库，可同时传 paper_id；可选项用于前端展示。
    """
    user_id: str
    source: str = "arxiv"           # arxiv / ieee
    paper_id: Optional[str] = None  # 论文 uuid（可空，hunter 阶段没有）
    title: Optional[str] = None     # 冗余存标题
    # 二选一必填其一：
    arxiv_id: Optional[str] = None  # arxiv 时填（如 2401.01234）
    ieee_id: Optional[str] = None   # IEEE 时填（article_number）


class MarkReadListItem(BaseModel):
    paper_key: str
    source: str
    reason: str
    title: str
    kb_id: Optional[str] = None
    paper_id: Optional[str] = None
    created_at: str


class MarkReadListResponse(BaseModel):
    success: bool
    user_id: str
    total: int
    items: List[MarkReadListItem]


# ============================================================
# arxiv query 构造（保留）
# ============================================================

def _build_arxiv_query(keywords: str) -> str:
    """
    构建 ArXiv 查询语句，支持多关键词 AND 组合
    """
    if not keywords or not keywords.strip():
        return ""

    cleaned = keywords.strip().lower()
    parts = [p.strip() for p in cleaned.split() if p.strip()]

    if not parts:
        return ""
    if len(parts) == 1:
        return f"all:{parts[0]}"
    query_parts = [f"all:{p}" for p in parts]
    return " AND ".join(query_parts)


# ============================================================
# POST /search — 搜索 + 笔记生成
# ============================================================

@router.post("/search", response_model=Dict[str, Any])
async def search_papers(request: PaperSearchRequest):
    """搜索论文 — 委托 HunterAgent。

    v2 行为：
      - 不下载到正式目录，不入库 PG
      - Hunter 会异步下载 PDF 到 HunterCache + 解析 intro/methodology + LLM 生成笔记
      - 返回带 note / note_status / pdf_url / pdf_cached 的论文列表
      - 用户再点"下载到知识库"按钮触发入库
    """
    try:
        keywords = [kw.strip() for kw in re.split(r'[，,、]+', request.keywords) if kw.strip()]
        if not keywords:
            return {
                "success": True,
                "model_info": _model_info(),
                "papers": [],
                "total_found": 0,
                "keywords": request.keywords,
                "source": request.source,
                "kb_id": request.kb_id,
                "message": "关键词不能为空",
            }

        sources = ["arxiv"] if request.source in ("arxiv", "all") else [request.source]

        logger.info(
            f"[Hunter] 搜索任务: keywords={keywords}, sources={sources}, "
            f"kb_id={request.kb_id}, user_id={request.user_id}"
        )

        task_id = await agent_controller.submit_task(
            TaskType.PAPER_HUNTING,
            {
                "keywords": keywords,
                "max_papers": request.limit,
                "sources": sources,
                "days_back": request.days_back,
                "kb_id": request.kb_id,
                "user_id": request.user_id,
            },
        )
        result = await agent_controller.execute_task(task_id)

        controller_papers = result.get("papers_found", [])
        papers: List[Dict[str, Any]] = []
        for p in controller_papers:
            papers.append({
                "id": p.get("id", ""),
                # 把 source 透传到每篇 paper（前端标记已读按钮、列表过滤等都需要）
                "source": p.get("source", sources[0] if sources else "arxiv"),
                "title": p.get("title", ""),
                "authors": p.get("authors", []) or [],
                "abstract": (p.get("abstract", "") or "").replace("\n", " ").strip(),
                "url": p.get("pdf_url", "").replace("/pdf/", "/abs/") if p.get("pdf_url") else "",
                "published_date": p.get("published", ""),
                "pdf_url": p.get("pdf_url", ""),
                "categories": p.get("categories", []),
                # v2 新字段
                "note": p.get("note", "") or "",
                "note_status": p.get("note_status", "pending"),
                "relevance_score": p.get("relevance_score", 0.0),
                "pdf_cached": p.get("pdf_cached", False),
            })

        return {
            "success": True,
            "model_info": _model_info(),
            "papers": papers,
            "total_found": result.get("total_found", len(papers)),
            "keywords": request.keywords,
            "source": request.source,
            "kb_id": request.kb_id,
            "agent_task_id": task_id,
        }

    except Exception as e:
        logger.error(f"论文搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


# ============================================================
# POST /{paper_id}/download-to-kb — 用户点击入库
# ============================================================

@router.post("/{paper_id}/download-to-kb", response_model=Dict[str, Any])
async def download_paper_to_kb(paper_id: str, req: DownloadToKBRequest):
    """把 Hunter 搜索结果中的论文下载到指定知识库。

    流程：
      1. PDF 来源选择：先查 HunterCache，未命中则按 pdf_url 重新下载
      2. SHA-256 去重：查 papers 表，已存在则复用 paper_id
      3. 写 PG papers 表（如未存在），落到 downloads/papers/
      4. 关联 kb_paper_relations
      5. 解析分块 → Qdrant 写入
      6. 注入笔记 chunk（来自 cache 或请求体）→ Qdrant 写入

    Args:
        paper_id: arxiv id（也用作 cache key 的一部分）
        req: {user_id, kb_id, pdf_url, note?, title?, abstract?, arxiv_id?}

    Returns:
        {success, paper_id, chunk_count, note_chunk_id, kb_id, cached}
    """
    arxiv_id = req.arxiv_id or paper_id
    user_id = req.user_id
    kb_id = req.kb_id

    if not user_id or not kb_id:
        raise HTTPException(status_code=400, detail="user_id 和 kb_id 必填")

    try:
        # 0. ownership 校验（不存在/不属于 user 一律 404）
        await knowledge_base_manager.get_kb(kb_id, user_id)

        # 1. 获取 PDF 字节（cache 优先）
        cache_hit = False
        pdf_bytes: Optional[bytes] = None
        cached_pdf_path = await hunter_cache.get_pdf_path(arxiv_id)
        if cached_pdf_path and cached_pdf_path.exists():
            pdf_bytes = cached_pdf_path.read_bytes()
            cache_hit = True
            logger.info(f"复用 cache PDF: arxiv_id={arxiv_id}")
        elif req.pdf_url:
            # 降级下载
            pdf_bytes = await _download_pdf_bytes(req.pdf_url)
            if pdf_bytes:
                await hunter_cache.save_pdf(arxiv_id, pdf_bytes)
                logger.info(f"按 pdf_url 重新下载 PDF: arxiv_id={arxiv_id}")
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"按 pdf_url 下载失败: {req.pdf_url}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="既无 cache PDF 也无 pdf_url，无法下载",
            )

        # 2. SHA-256 去重 + 入 PG
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        existing = await db_manager.get_paper_by_hash(content_hash)
        if existing and existing.get("id"):
            paper_db_id = str(existing["id"])
            logger.info(f"PDF 已存在（content_hash 命中）: {paper_db_id}")
        else:
            # 解析 PDF 取 title/authors/abstract
            pdf_result = await pdf_parser.parse_pdf_from_bytes(
                pdf_bytes, f"{arxiv_id}.pdf"
            )
            title = (
                req.title
                or (pdf_result.get("title") if pdf_result.get("success") else None)
                or f"arxiv:{arxiv_id}"
            )
            authors = (pdf_result.get("authors") if pdf_result.get("success") else []) or []
            abstract = (
                req.abstract
                or (pdf_result.get("abstract") if pdf_result.get("success") else "")
                or ""
            )
            # 写到正式存储区
            os.makedirs("downloads/papers", exist_ok=True)
            safe_title = "".join(c if c.isalnum() else "_" for c in (title or "paper"))[:50]
            file_path = os.path.join("downloads/papers", f"{arxiv_id}_{safe_title}.pdf")
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)

            paper_db_id = await db_manager.create_paper(
                title=title,
                authors=authors,
                abstract=abstract,
                doi=None,
                file_path=file_path,
                content_hash=content_hash,
                is_preset=False,
            )
            logger.info(
                f"PDF 入库成功: paper_id={paper_db_id}, file_path={file_path}"
            )

        # 2b. 自动写入"已读"集合（被动触发）
        # 注意：下载到 KB 等价于"用户已对这篇论文表态过"，
        # 后续即便换 KB 搜索也要跳过它，避免重复推荐。
        # 当前 download_paper_to_kb 只服务 arxiv 路径（paper_id 形参即 arxiv id）；
        # IEEE 走另一条路：传 req.ieee_id（未启用，预留）。
        try:
            if req.arxiv_id or arxiv_id:
                source = "arxiv"
                paper_key_id = req.arxiv_id or arxiv_id
            elif getattr(req, "ieee_id", None):
                source = "ieee"
                paper_key_id = req.ieee_id
            else:
                source = "arxiv"
                paper_key_id = arxiv_id
            await db_manager.mark_paper_read(
                user_id=user_id,
                paper_key=f"{source}:{paper_key_id}",
                source=source,
                reason="added_to_kb",
                kb_id=kb_id,
                paper_id=paper_db_id,
                title=title or req.title or f"{source}:{paper_key_id}",
            )
        except Exception as e:
            logger.warning(f"download-to-kb 自动 mark_paper_read 失败: {e}")

        # 3. 关联 KB（幂等）
        await knowledge_base_manager.add_paper_to_kb(
            kb_id=kb_id, paper_id=paper_db_id, user_id=user_id
        )

        # 3b. 回填 paper_id → hunter_notes（笔记关联）
        await db_manager.bind_paper_to_hunter_note(arxiv_id, paper_db_id)

        # 4. 解析分块 → Qdrant 写入
        proc = await knowledge_base_manager.process_paper_to_chunks(
            kb_id=kb_id, paper_id=paper_db_id, user_id=user_id
        )
        chunk_count = proc.get("chunk_count", 0)

        # 5. 注入笔记 chunk（先查 PG hunter_notes，再用请求体）
        note_text = req.note or ""
        if not note_text:
            cached_note = await db_manager.get_hunter_note(arxiv_id)
            if cached_note:
                note_text = cached_note.get("note", "")
        note_chunk_id = ""
        if note_text:
            # 拿到 paper 元数据传给 add_note_as_chunk
            paper_row = await db_manager.get_paper(paper_db_id)
            note_chunk_id = await knowledge_base_manager.add_note_as_chunk(
                kb_id=kb_id,
                paper_id=paper_db_id,
                user_id=user_id,
                note_text=note_text,
                paper_title=(paper_row or {}).get("title") or req.title or f"arxiv:{arxiv_id}",
                paper_meta={
                    "authors": (paper_row or {}).get("authors", []),
                },
            )

        return {
            "success": True,
            "paper_id": paper_db_id,
            "arxiv_id": arxiv_id,
            "kb_id": kb_id,
            "chunk_count": chunk_count,
            "note_chunk_id": note_chunk_id,
            "cached": cache_hit,
            "message": "已成功下载到知识库",
        }

    except InnoCoreException as e:
        logger.error(f"download-to-kb 业务异常: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"download-to-kb 失败: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")


# ============================================================
# POST /mark-read — 主动标记已读
# GET  /mark-read — 查看用户的已读清单
# ============================================================

def _resolve_paper_key(req: MarkReadRequest) -> tuple[str, str, str]:
    """从请求解析 (source, paper_id_for_key, title)。

    Raises:
        HTTPException: 当 arxiv_id / ieee_id 都缺失时返回 400。
    """
    if req.arxiv_id:
        return "arxiv", req.arxiv_id, req.title or f"arxiv:{req.arxiv_id}"
    if req.ieee_id:
        return "ieee", req.ieee_id, req.title or f"ieee:{req.ieee_id}"
    raise HTTPException(
        status_code=400,
        detail="arxiv_id 或 ieee_id 至少填一个",
    )


@router.post("/mark-read", response_model=Dict[str, Any])
async def mark_paper_as_read(req: MarkReadRequest):
    """用户主动将 hunter 搜索结果中的某篇论文标记为已读。

    与 /download-to-kb 的区别：
      - 不下载 PDF、不入库
      - 单纯写一条 user_paper_read_state(reason='marked_read')
      - 立即影响下次 Hunter 搜索的过滤结果
    """
    try:
        source, pid, title = _resolve_paper_key(req)
        paper_key = f"{source}:{pid}"
        ok = await db_manager.mark_paper_read(
            user_id=req.user_id,
            paper_key=paper_key,
            source=source,
            reason="marked_read",
            title=title,
            kb_id=None,
            paper_id=req.paper_id,
        )
        return {
            "success": ok,
            "user_id": req.user_id,
            "paper_key": paper_key,
            "source": source,
            "reason": "marked_read",
            "message": "已标记为已读" if ok else "标记失败",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"mark-read 失败: {e}")
        raise HTTPException(status_code=500, detail=f"标记失败: {str(e)}")


@router.get("/mark-read", response_model=MarkReadListResponse)
async def list_read_papers(user_id: str, limit: int = 200):
    """查看某用户的已读论文清单（前端展示用）。"""
    try:
        rows = await db_manager.list_read_papers_for_user(user_id, limit=limit)
        items = [
            MarkReadListItem(
                paper_key=r.get("paper_key", ""),
                source=r.get("source", ""),
                reason=r.get("reason", ""),
                title=r.get("title", ""),
                kb_id=str(r["kb_id"]) if r.get("kb_id") else None,
                paper_id=str(r["paper_id"]) if r.get("paper_id") else None,
                created_at=r["created_at"].isoformat() if r.get("created_at") else "",
            )
            for r in rows
        ]
        return MarkReadListResponse(
            success=True,
            user_id=user_id,
            total=len(items),
            items=items,
        )
    except Exception as e:
        logger.exception(f"list mark-read 失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


async def _download_pdf_bytes(pdf_url: str) -> Optional[bytes]:
    """按 URL 下载 PDF 字节（带简单重试）。"""
    import aiohttp

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_url,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"PDF 下载 HTTP {response.status}: {pdf_url}")
                        return None
        except Exception as e:
            logger.warning(f"PDF 下载异常 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 * (attempt + 1))
    return None


# ============================================================
# POST /upload — 上传本地 PDF（保留兼容）
# ============================================================

@router.post("/upload", response_model=Dict[str, Any])
async def upload_paper(file: UploadFile = File(...)):
    """上传论文PDF（兼容旧路径，新代码请走 /api/v1/analysis/upload-pdf）"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="只支持PDF文件")

        file_url = f"/uploads/{file.filename}"

        return {
            "success": True,
            "file_url": file_url,
            "filename": file.filename,
            "size": getattr(file, "size", 0),
            "message": "文件上传成功",
        }

    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")