"""
KnowledgeBaseManager — KB 生命周期 + chunk 入库编排

封装:
  - KB CRUD（创建/列出/删除/重命名）
  - 论文加入 KB（DB 关联 + Qdrant 写 chunks）
  - KB 内 RAG 问答（search + LLM 生成）
  - 访问控制：每个 KB 必须校验 user_id ownership

依赖:
  - core.database.DatabaseManager（PG）
  - core.vector_store.VectorStoreManager（Qdrant）
  - utils.chunk_processor.process_paper_to_chunks
  - utils.research_paper_parser / utils.pdf_parser（解析）
  - core.llm_adapter.LLMAdapter（生成）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .database import db_manager
from .exceptions import DatabaseException, InnoCoreException, LLMException, VectorStoreException
from .llm_adapter import llm_adapter
from .vector_store import vector_store_manager

logger = logging.getLogger(__name__)


# ============================================================
# 自定义异常
# ============================================================

class KnowledgeBaseNotFound(InnoCoreException):
    """指定的知识库 ID 不存在或不属于当前用户。"""
    pass


class KnowledgeBaseAccessDenied(InnoCoreException):
    """用户越权访问他人 KB（kb_id 存在但 user_id 不匹配）。"""
    pass


# ============================================================
# 数据结构
# ============================================================

@dataclass
class KnowledgeBase:
    id: str
    user_id: str
    name: str
    description: str
    embedding_model: str
    paper_count: int
    chunk_count: int
    is_default: bool
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict) -> "KnowledgeBase":
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=row.get("name", ""),
            description=row.get("description", "") or "",
            embedding_model=row.get("embedding_model", "text-embedding-v4"),
            paper_count=int(row.get("paper_count", 0) or 0),
            chunk_count=int(row.get("chunk_count", 0) or 0),
            is_default=bool(row.get("is_default", False)),
            created_at=str(row.get("created_at", "") or ""),
        )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "embedding_model": self.embedding_model,
            "paper_count": self.paper_count,
            "chunk_count": self.chunk_count,
            "is_default": self.is_default,
            "created_at": self.created_at,
        }


# ============================================================
# Manager 类
# ============================================================

class KnowledgeBaseManager:
    """KB 生命周期与检索的主入口。所有方法异步。"""

    DEFAULT_KB_NAME = "默认知识库"

    # ---------- KB CRUD ----------

    async def list_kbs(self, user_id: str) -> List[KnowledgeBase]:
        """获取用户的所有 KB，确保每个用户至少有一个默认 KB。"""
        if not user_id:
            raise KnowledgeBaseAccessDenied("user_id 不能为空")
        try:
            await self._ensure_default_kb(user_id)
            rows = await db_manager.fetch(
                "SELECT * FROM knowledge_bases WHERE user_id = $1 ORDER BY is_default DESC, created_at ASC",
                user_id,
            )
            return [KnowledgeBase.from_row(r) for r in rows]
        except DatabaseException as e:
            logger.error(f"list_kbs 失败: {e}")
            raise

    async def get_kb(self, kb_id: str, user_id: str) -> KnowledgeBase:
        """获取指定 KB。校验 ownership，不属于当前用户则抛 AccessDenied。"""
        row = await db_manager.fetchrow(
            "SELECT * FROM knowledge_bases WHERE id = $1", kb_id,
        )
        if not row:
            raise KnowledgeBaseNotFound(f"知识库不存在: {kb_id}")
        if str(row["user_id"]) != str(user_id):
            raise KnowledgeBaseAccessDenied(f"无权访问知识库: {kb_id}")
        return KnowledgeBase.from_row(dict(row))

    async def create_kb(
        self,
        user_id: str,
        name: str,
        description: str = "",
        embedding_model: str = "text-embedding-v4",
    ) -> KnowledgeBase:
        name = (name or "").strip()
        if not name:
            raise InnoCoreException("KB 名称不能为空")
        if len(name) > 120:
            raise InnoCoreException("KB 名称长度不能超过 120 字符")

        # 同名检测：UNIQUE(user_id, name) 由建表约束
        try:
            new_id = await db_manager.fetchval(
                """
                INSERT INTO knowledge_bases (user_id, name, description, embedding_model)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                user_id, name, description, embedding_model,
            )
        except Exception as e:
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg:
                raise InnoCoreException(f"已存在同名 KB: {name}")
            raise DatabaseException(f"create_kb 失败: {e}")
        return await self.get_kb(str(new_id), user_id)

    async def update_kb(
        self,
        kb_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> KnowledgeBase:
        await self.get_kb(kb_id, user_id)  # ownership 校验
        sets, vals = [], []
        if name is not None:
            sets.append(f"name = ${len(vals)+1}")
            vals.append((name or "").strip()[:120])
        if description is not None:
            sets.append(f"description = ${len(vals)+1}")
            vals.append(description)
        if not sets:
            return await self.get_kb(kb_id, user_id)
        sets.append("updated_at = CURRENT_TIMESTAMP")
        vals.extend([kb_id])
        sql = f"UPDATE knowledge_bases SET {', '.join(sets)} WHERE id = ${len(vals)}"
        await db_manager.execute(sql, *vals)
        return await self.get_kb(kb_id, user_id)

    async def delete_kb(self, kb_id: str, user_id: str) -> bool:
        kb = await self.get_kb(kb_id, user_id)
        # 若 vector_store 已初始化，物理删除该 KB 在 Qdrant 的全部向量
        if getattr(vector_store_manager, "client", None) is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            try:
                await asyncio.to_thread(
                    vector_store_manager.client.delete,
                    collection_name=vector_store_manager.l2_collection,
                    points_selector=Filter(must=[
                        FieldCondition(key="kb_id", match=MatchValue(value=kb_id)),
                    ]),
                )
            except Exception as e:
                logger.warning(f"Qdrant 清理 kb_id={kb_id} 失败（继续删除 DB 记录）: {e}")
        await db_manager.execute("DELETE FROM knowledge_bases WHERE id = $1", kb_id)
        logger.info(f"KB 删除: kb_id={kb_id}, user_id={user_id}, name={kb.name}")
        return True

    async def _ensure_default_kb(self, user_id: str) -> str:
        """保证每个用户有一个默认 KB，没有则创建。返回 KB id。"""
        row = await db_manager.fetchrow(
            "SELECT id FROM knowledge_bases WHERE user_id = $1 AND is_default = TRUE LIMIT 1",
            user_id,
        )
        if row:
            return str(row["id"])
        # 同时检查 name = 默认知识库 的 KB
        row = await db_manager.fetchrow(
            "SELECT id FROM knowledge_bases WHERE user_id = $1 AND name = $2 LIMIT 1",
            user_id, self.DEFAULT_KB_NAME,
        )
        if row:
            # 标记为默认
            await db_manager.execute(
                "UPDATE knowledge_bases SET is_default = TRUE WHERE id = $1",
                row["id"],
            )
            return str(row["id"])
        # 创建默认 KB
        new_id = await db_manager.fetchval(
            """
            INSERT INTO knowledge_bases (user_id, name, description, is_default)
            VALUES ($1, $2, $3, TRUE) RETURNING id
            """,
            user_id, self.DEFAULT_KB_NAME, "自动创建的默认知识库",
        )
        logger.info(f"为新用户创建默认 KB: user_id={user_id}, kb_id={new_id}")
        return str(new_id)

    async def get_or_create_default_kb(self, user_id: str) -> KnowledgeBase:
        kb_id = await self._ensure_default_kb(user_id)
        return await self.get_kb(kb_id, user_id)

    # ---------- Paper ↔ KB 关联 + chunk 落库 ----------

    async def add_paper_to_kb(
        self,
        kb_id: str,
        paper_id: str,
        user_id: str,
    ) -> bool:
        """把论文加入 KB(仅 DB 关联)。幂等:重复加入直接返回 False。"""
        await self.get_kb(kb_id, user_id)
        row = await db_manager.fetchrow("SELECT id FROM papers WHERE id = $1", paper_id)
        if not row:
            raise InnoCoreException(f"论文不存在: {paper_id}")
        # 1. 插入或 no-op
        result = await db_manager.execute(
            """
            INSERT INTO kb_paper_relations (kb_id, paper_id) VALUES ($1, $2)
            ON CONFLICT (kb_id, paper_id) DO NOTHING
            """,
            kb_id, paper_id,
        )
        # result 形如 "INSERT 0 1" 或 "INSERT 0 0"
        inserted = result.endswith("1")
        if inserted:
            await db_manager.execute(
                """
                UPDATE knowledge_bases
                SET paper_count = paper_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                """,
                kb_id,
            )
        return inserted

    async def list_papers_in_kb(self, kb_id: str, user_id: str) -> List[Dict]:
        """列出 KB 内的论文（不含 chunk 详细）。"""
        await self.get_kb(kb_id, user_id)
        rows = await db_manager.fetch(
            """
            SELECT p.id, p.title, p.authors, p.abstract, p.publication_date,
                   rel.added_at, rel.chunk_count
            FROM kb_paper_relations rel
            JOIN papers p ON p.id = rel.paper_id
            WHERE rel.kb_id = $1
            ORDER BY rel.added_at DESC
            """,
            kb_id,
        )
        return [dict(r) for r in rows]

    async def remove_paper_from_kb(
        self,
        kb_id: str,
        paper_id: str,
        user_id: str,
        delete_vectors: bool = True,
    ) -> bool:
        """从 KB 移除论文。可选同步删除 Qdrant 的 chunks。"""
        await self.get_kb(kb_id, user_id)
        if delete_vectors:
            try:
                await vector_store_manager.delete_paper_chunks(kb_id, paper_id)
            except Exception as e:
                logger.warning(f"删除 Qdrant chunks 失败: {e}")
        await db_manager.execute(
            "DELETE FROM kb_paper_relations WHERE kb_id = $1 AND paper_id = $2",
            kb_id, paper_id,
        )
        await db_manager.execute(
            """
            UPDATE knowledge_bases
            SET paper_count = GREATEST(paper_count - 1, 0), updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            kb_id,
        )
        return True

    async def process_paper_to_chunks(
        self,
        kb_id: str,
        paper_id: str,
        user_id: str,
    ) -> Dict:
        """
        把已加入 KB 的论文解析 → 分块 → 向量化 → 写入 Qdrant。

        Returns:
            {"kb_id, paper_id, status, chunk_count, log_id}
        """
        await self.get_kb(kb_id, user_id)

        # 1. 读 papers 表
        prow = await db_manager.fetchrow("SELECT * FROM papers WHERE id = $1", paper_id)
        if not prow:
            raise InnoCoreException(f"论文不存在: {paper_id}")
        paper = dict(prow)

        file_path = paper.get("file_path")
        if not file_path or not os.path.exists(file_path):
            # 没有本地文件 → 需要 PDF URL（占位：留接口以后接）
            raise InnoCoreException(
                f"论文 {paper_id} 没有可解析的文件路径: {file_path!r}，"
                "MVP 阶段仅支持已有 file_path 的论文。"
            )

        # 2. 解析 PDF → ResearchPaper
        from utils.pdf_parser import pdf_parser  # 延迟导入避免循环
        from utils.research_paper_parser import ResearchPaperParser

        pdf_result = await pdf_parser.parse_pdf(file_path)
        if not pdf_result.get("success"):
            raise InnoCoreException(f"PDF 解析失败: {pdf_result.get('error', 'unknown')}")

        research_paper = ResearchPaperParser().parse(pdf_result.get("full_text", ""))

        # 3. 写 chunks 到 Qdrant
        log_id = await db_manager.fetchval(
            """
            INSERT INTO kb_paper_chunks_log (kb_id, paper_id, status)
            VALUES ($1, $2, 'processing') RETURNING id
            """,
            kb_id, paper_id,
        )
        try:
            chunks = _run_chunker(research_paper, paper_id=paper_id)
            ids = await vector_store_manager.add_paper_chunks_kb(
                kb_id=kb_id,
                user_id=user_id,
                paper_id=paper_id,
                chunks=chunks,
                paper_meta={
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []) or [],
                    "venue": paper.get("venue", ""),
                    "published_year": _year_from_date(paper.get("publication_date")),
                },
            )
            chunk_count = len(ids)
            # 4. 更新 log + relation + KB 计数
            await db_manager.execute(
                """
                UPDATE kb_paper_chunks_log
                SET status = 'ready', chunk_count = $1, completed_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                chunk_count, log_id,
            )
            await db_manager.execute(
                "UPDATE kb_paper_relations SET chunk_count = $1 WHERE kb_id = $2 AND paper_id = $3",
                chunk_count, kb_id, paper_id,
            )
            await db_manager.execute(
                """
                UPDATE knowledge_bases
                SET chunk_count = chunk_count + $1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                chunk_count, kb_id,
            )
            return {
                "kb_id": kb_id, "paper_id": paper_id,
                "status": "ready", "chunk_count": chunk_count, "log_id": str(log_id),
            }
        except Exception as e:
            logger.exception(f"process_paper_to_chunks 失败: {e}")
            try:
                await db_manager.execute(
                    "UPDATE kb_paper_chunks_log SET status = 'failed', error_message = $1, completed_at = CURRENT_TIMESTAMP WHERE id = $2",
                    str(e), log_id,
                )
            except Exception:
                pass
            raise

    # ---------- RAG 问答 ----------

    async def ask_knowledge_base(
        self,
        kb_id: str,
        user_id: str,
        query: str,
        top_k: int = 5,
        paper_filter: Optional[str] = None,
    ) -> Dict:
        """RAG 问答：search_kb + LLM 生成。

        Returns:
            {
              "kb_id, query, answer, sources": [{paper_id, title, section, content, score}, ...]
            }
        """
        await self.get_kb(kb_id, user_id)
        if not (query or "").strip():
            raise InnoCoreException("query 不能为空")

        # 1. 检索
        chunks = await vector_store_manager.search_kb(
            kb_id=kb_id, query=query, top_k=top_k, paper_filter=paper_filter,
        )

        if not chunks:
            return {
                "kb_id": kb_id,
                "query": query,
                "answer": "未在此知识库中找到相关内容。请先在此 KB 上传论文。",
                "sources": [],
            }

        # 2. 拼 prompt（带来源）
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[来源 {i}] 论文:{c['title']} | 章节:{c['section_name']} | "
                f"相关度:{c['score']:.2f}\n{c['content']}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            "你是一名严谨的研究助理，请仅根据【参考资料】回答用户问题。"
            "如果资料不足，请直接说明，不要编造。回答使用中文，先给结论再简短展开。\n\n"
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】\n{query}\n\n"
            "【回答】"
        )

        # 3. LLM 生成（兼容 LLMAdapter 同步 / async 两种调用）
        try:
            answer = await _call_llm_text(prompt)
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            raise LLMException(f"LLM 生成失败: {e}")

        sources = [
            {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "authors": c["authors"],
                "section_name": c["section_name"],
                "section_type": c["section_type"],
                "chunk_index": c["chunk_index"],
                "content": c["content"][:1500],
                "score": c["score"],
            }
            for c in chunks
        ]
        return {
            "kb_id": kb_id,
            "query": query,
            "answer": answer,
            "sources": sources,
        }


# ============================================================
# 模块级辅助函数
# ============================================================

def _run_chunker(paper, paper_id: str) -> List[Document]:
    """调用 ChunkProcessor。包装 try/except 让上层只看业务异常。"""
    from utils.chunk_processor import process_paper_to_chunks
    return process_paper_to_chunks(paper, paper_id=paper_id)


async def _call_llm_text(prompt: str) -> str:
    """调用 LLM 生成文本。统一用 LLMAdapter.ainvoke（异步）。"""
    try:
        return await llm_adapter.ainvoke(prompt)
    except Exception as e:
        logger.warning(f"ainvoke 失败,回退到 invoke: {e}")
        return llm_adapter.invoke(prompt)


def _year_from_date(date_str: Optional[str]) -> int:
    """把 'YYYY-MM-DD' / 'YYYY' 解析成年份整数；解析失败返回 0。"""
    if not date_str:
        return 0
    s = str(date_str).strip()
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else 0


# ============================================================
# 全局实例
# ============================================================

knowledge_base_manager = KnowledgeBaseManager()
