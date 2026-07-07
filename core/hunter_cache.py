"""
HunterCache — Hunter Agent 临时缓存管理。

设计目标：
- 全局共享：同一篇 arxiv 论文多用户访问只下载/解析/生成笔记一次
- 自动清理：默认 TTL=7 天，应用启动时扫描一次 + 每 24h 周期清理
- 数据隔离：
  * 临时缓存 downloads/cache/{arxiv_id}/   → 7 天后自动清理
  * 正式存储 downloads/papers/{arxiv_id}_{safe_title}.pdf  → 用户入库后落地
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HunterCache:
    """Hunter 阶段临时缓存的统一入口。"""

    CACHE_ROOT = "downloads/cache"
    DEFAULT_TTL_DAYS = 7

    def __init__(
        self,
        cache_root: Optional[str] = None,
        ttl_days: Optional[int] = None,
    ):
        self.cache_root = Path(cache_root or self.CACHE_ROOT)
        self.ttl_seconds = (ttl_days or self.DEFAULT_TTL_DAYS) * 86400
        # 确保根目录存在
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ---------- 路径工具 ----------

    def _cache_dir(self, arxiv_id: str) -> Path:
        """返回某篇论文的缓存目录。"""
        safe_id = self._sanitize_id(arxiv_id)
        return self.cache_root / safe_id

    def _is_expired(self, path: Path) -> bool:
        """判断目录是否超过 TTL。"""
        try:
            mtime = path.stat().st_mtime
            return (time.time() - mtime) > self.ttl_seconds
        except FileNotFoundError:
            return True

    @staticmethod
    def _sanitize_id(arxiv_id: str) -> str:
        """清洗 arxiv id（防止路径注入）。"""
        return "".join(c for c in (arxiv_id or "") if c.isalnum() or c in "-._")

    # ---------- 元数据 ----------

    async def save_meta(self, arxiv_id: str, meta: Dict[str, Any]) -> None:
        """保存论文元数据到 meta.json。"""
        d = self._cache_dir(arxiv_id)
        d.mkdir(parents=True, exist_ok=True)
        meta_file = d / "meta.json"
        meta["arxiv_id"] = arxiv_id
        async with asyncio.Lock():
            meta_file.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def load_meta(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """读取 meta.json；不存在返回 None。"""
        meta_file = self._cache_dir(arxiv_id) / "meta.json"
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"load_meta 解析失败 {arxiv_id}: {e}")
            return None

    # ---------- PDF ----------

    async def save_pdf(self, arxiv_id: str, pdf_bytes: bytes) -> Path:
        """保存 PDF 字节到 paper.pdf。"""
        d = self._cache_dir(arxiv_id)
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / "paper.pdf"
        pdf_path.write_bytes(pdf_bytes)
        # 更新 mtime，便于 TTL 判定
        os.utime(pdf_path, None)
        return pdf_path

    async def get_pdf_path(self, arxiv_id: str) -> Optional[Path]:
        """返回 paper.pdf 路径；不存在或过期返回 None。"""
        pdf_path = self._cache_dir(arxiv_id) / "paper.pdf"
        if not pdf_path.exists():
            return None
        if self._is_expired(pdf_path):
            return None
        return pdf_path

    async def has_pdf(self, arxiv_id: str) -> bool:
        return await self.get_pdf_path(arxiv_id) is not None

    # ---------- 解析结果 (intro/method) ----------

    async def save_parsed(
        self, arxiv_id: str, contribution: str, methodology: str
    ) -> None:
        d = self._cache_dir(arxiv_id)
        d.mkdir(parents=True, exist_ok=True)
        parsed_file = d / "parsed.json"
        async with asyncio.Lock():
            parsed_file.write_text(
                json.dumps(
                    {"contribution": contribution, "methodology": methodology},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    async def load_parsed(self, arxiv_id: str) -> Optional[Dict[str, str]]:
        parsed_file = self._cache_dir(arxiv_id) / "parsed.json"
        if not parsed_file.exists():
            return None
        try:
            return json.loads(parsed_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"load_parsed 解析失败 {arxiv_id}: {e}")
            return None

    # ---------- 清理 ----------

    async def cleanup_expired(self) -> int:
        """遍历 CACHE_ROOT，删除 mtime > TTL 的整个 arxiv_id 子目录。

        注意：笔记（hunter_notes）已迁移至 PG，不在此清理范围内。
        """
        cleaned = 0
        if not self.cache_root.exists():
            return 0
        for sub in self.cache_root.iterdir():
            if not sub.is_dir():
                continue
            if self._is_expired(sub):
                try:
                    shutil.rmtree(sub)
                    cleaned += 1
                    logger.info(f"Hunter cache 清理过期目录: {sub.name}")
                except Exception as e:
                    logger.warning(f"清理 Hunter cache 失败 {sub}: {e}")
        return cleaned

    async def clear_all(self) -> int:
        """清空整个缓存（调试/测试用）。"""
        cleaned = 0
        if not self.cache_root.exists():
            return 0
        for sub in self.cache_root.iterdir():
            if sub.is_dir():
                try:
                    shutil.rmtree(sub)
                    cleaned += 1
                except Exception:
                    pass
        return cleaned


# ============================================================
# 全局单例 + 周期清理任务
# ============================================================

hunter_cache = HunterCache()


async def cache_cleanup_loop(cache: HunterCache, interval_seconds: int = 86400):
    """后台周期清理任务；每 interval_seconds 清理一次。"""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            n = await cache.cleanup_expired()
            if n:
                logger.info(f"Hunter cache 周期清理完成：删除 {n} 个过期条目")
        except asyncio.CancelledError:
            logger.info("Hunter cache 周期清理任务被取消")
            raise
        except Exception as e:
            logger.warning(f"Hunter cache 周期清理失败: {e}")