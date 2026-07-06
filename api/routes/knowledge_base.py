"""
Knowledge Base 路由 - KB CRUD + chunk 入库 + RAG 问答

端点（前端 §七 所需）:
  GET    /api/v1/kb/list?user_id=XXX          列出用户的 KB（含默认 KB 自动创建）
  POST   /api/v1/kb/create                   创建 KB
  PATCH  /api/v1/kb/{kb_id}                  更新 KB 名称/描述
  DELETE /api/v1/kb/{kb_id}                  删除 KB（物理清理 Qdrant）
  GET    /api/v1/kb/{kb_id}/papers           列 KB 内的论文
  POST   /api/v1/kb/{kb_id}/papers           把论文加入 KB
  DELETE /api/v1/kb/{kb_id}/papers/{pid}     从 KB 移除论文（同步删 chunks）
  POST   /api/v1/kb/{kb_id}/papers/{pid}/process   分块+向量化入库

  POST   /api/v1/chat/ask                    RAG 问答（kb_id + user_id 必填）

user_id 在 MVP 阶段用 query 参数或 body 字段（auth 阶段改 JWT）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.exceptions import DatabaseException, InnoCoreException, VectorStoreException
from core.knowledge_base_manager import (
    KnowledgeBaseAccessDenied,
    KnowledgeBaseNotFound,
    knowledge_base_manager,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# Pydantic 模型
# ============================================================

class CreateKBRequest(BaseModel):
    user_id: str = Field(..., description="所属用户 ID（MVP 阶段从 query/body 传）")
    name: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    embedding_model: str = "text-embedding-v4"


class UpdateKBRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    description: Optional[str] = None


class AddPaperToKBRequest(BaseModel):
    user_id: str
    paper_id: str


# ============================================================
# 错误处理统一
# ============================================================

def _err(e: Exception) -> HTTPException:
    if isinstance(e, KnowledgeBaseNotFound):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, KnowledgeBaseAccessDenied):
        return HTTPException(status_code=403, detail=str(e))
    if isinstance(e, InnoCoreException):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, (DatabaseException, VectorStoreException)):
        return HTTPException(status_code=500, detail=str(e))
    return HTTPException(status_code=500, detail=f"unknown error: {e}")


# ============================================================
# 路由
# ============================================================

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_kbs(user_id: str = Query(..., description="用户 ID")):
    try:
        kbs = await knowledge_base_manager.list_kbs(user_id)
        return [kb.to_dict() for kb in kbs]
    except Exception as e:
        raise _err(e)


@router.post("/create", response_model=Dict[str, Any])
async def create_kb(req: CreateKBRequest):
    try:
        kb = await knowledge_base_manager.create_kb(
            user_id=req.user_id,
            name=req.name,
            description=req.description,
            embedding_model=req.embedding_model,
        )
        return kb.to_dict()
    except Exception as e:
        raise _err(e)


@router.patch("/{kb_id}", response_model=Dict[str, Any])
async def update_kb(kb_id: str, req: UpdateKBRequest):
    try:
        kb = await knowledge_base_manager.update_kb(
            kb_id=kb_id, user_id=req.user_id,
            name=req.name, description=req.description,
        )
        return kb.to_dict()
    except Exception as e:
        raise _err(e)


@router.delete("/{kb_id}", response_model=Dict[str, Any])
async def delete_kb(kb_id: str, user_id: str = Query(...)):
    try:
        ok = await knowledge_base_manager.delete_kb(kb_id, user_id)
        return {"success": ok, "kb_id": kb_id}
    except Exception as e:
        raise _err(e)


@router.get("/{kb_id}/papers", response_model=List[Dict[str, Any]])
async def list_kb_papers(kb_id: str, user_id: str = Query(...)):
    try:
        return await knowledge_base_manager.list_papers_in_kb(kb_id, user_id)
    except Exception as e:
        raise _err(e)


@router.post("/{kb_id}/papers", response_model=Dict[str, Any])
async def add_paper_to_kb(kb_id: str, req: AddPaperToKBRequest):
    """把已存在的 paper_id 加入 KB（仅 DB 关联）；下一步调 /papers/{pid}/process 做分块入库。"""
    try:
        inserted = await knowledge_base_manager.add_paper_to_kb(
            kb_id=kb_id, paper_id=req.paper_id, user_id=req.user_id,
        )
        return {"success": True, "inserted": inserted}
    except Exception as e:
        raise _err(e)


@router.delete("/{kb_id}/papers/{paper_id}", response_model=Dict[str, Any])
async def remove_paper_from_kb(
    kb_id: str, paper_id: str, user_id: str = Query(...),
    delete_vectors: bool = Query(True),
):
    try:
        ok = await knowledge_base_manager.remove_paper_from_kb(
            kb_id=kb_id, paper_id=paper_id, user_id=user_id,
            delete_vectors=delete_vectors,
        )
        return {"success": ok}
    except Exception as e:
        raise _err(e)


@router.post("/{kb_id}/papers/{paper_id}/process", response_model=Dict[str, Any])
async def process_paper_to_chunks(kb_id: str, paper_id: str, user_id: str = Query(...)):
    """对 KB 内论文执行 PDF 解析 → 分块 → 向量化入库。同步执行。"""
    try:
        result = await knowledge_base_manager.process_paper_to_chunks(
            kb_id=kb_id, paper_id=paper_id, user_id=user_id,
        )
        return result
    except Exception as e:
        raise _err(e)
