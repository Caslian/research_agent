"""
Chat / RAG 问答路由

端点（前端 §七 所需）:
  POST /api/v1/chat/ask  - RAG 问答
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.exceptions import DatabaseException, InnoCoreException, VectorStoreException
from core.knowledge_base_manager import (
    KnowledgeBaseAccessDenied,
    KnowledgeBaseNotFound,
    knowledge_base_manager,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatAskRequest(BaseModel):
    user_id: str = Field(..., description="MVP 阶段从请求体传入")
    kb_id: str = Field(..., description="必须是当前用户拥有的 KB")
    query: str = Field(..., min_length=1)
    top_k: int = 5
    paper_filter: Optional[str] = None


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


@router.post("/ask", response_model=Dict[str, Any])
async def chat_ask(req: ChatAskRequest):
    """RAG 问答。严格校验 user_id 对 kb 的所有权。"""
    try:
        return await knowledge_base_manager.ask_knowledge_base(
            kb_id=req.kb_id, user_id=req.user_id,
            query=req.query, top_k=req.top_k,
            paper_filter=req.paper_filter,
        )
    except Exception as e:
        raise _err(e)
