"""
InnoCore AI 前哨探员 (Hunter Agent) - 基于 LangChain 框架
负责每日根据关键词监控ArXiv/IEEE，初筛并下载PDF

重构后职责（v2）：
1. 关键词搜索 arxiv → 按相关性排序
2. **不自动下载、不入库 PG** —— 由用户在前端点击"下载到知识库"才入库
3. 搜索阶段即异步下载 PDF + 解析 intro/methodology + LLM 生成中文笔记
4. PDF → HunterCache（7 天 TTL）；笔记 → PG hunter_notes 表（永久持久化）
"""
import asyncio
import aiohttp
import feedparser
import re
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from datetime import datetime, timedelta
import hashlib
import os
from urllib.parse import urljoin, quote

from agents.base import BaseAgent
from core.database import db_manager
from core.exceptions import AgentException, ExternalAPIException
from core.hunter_cache import hunter_cache, HunterCache
from utils.research_paper_parser import PaperIntroExtractor
from utils.note_prompts import HUNTER_NOTE_PROMPT, HUNTER_NOTE_FALLBACK_PROMPT
from core.llm_adapter import llm_adapter
from langchain_core.tools import tool
from typing import Optional
import logging
logger = logging.getLogger(__name__)
class HunterAgent(BaseAgent):
    """前哨探员智能体"""

    def __init__(self, llm=None):
        super().__init__("Hunter", llm)
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.ieee_base_url = "https://ieeexploreapi.ieee.org/api/v1"
        self.download_dir = "downloads/papers"
        self.cache: HunterCache = hunter_cache

        # 确保下载目录存在（最终正式存储区）
        os.makedirs(self.download_dir, exist_ok=True)

        # 添加工具
        self.add_tool("search_arxiv", self._search_arxiv, "搜索ArXiv论文")
        self.add_tool("search_ieee", self._search_ieee, "搜索IEEE论文")
        self.add_tool("download_pdf", self._download_pdf, "下载PDF文件")
        self.add_tool("extract_metadata", self._extract_metadata, "提取论文元数据")

    # ============================================================
    # 主入口（重构后）
    # ============================================================

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行论文搜索 + 笔记生成任务（不下载到正式目录，不入库 PG）。

        输入新增：
          - kb_id:    用于 KB 去重过滤 + 用户最终入库目标
          - user_id:  用户标识

        返回：
          - papers: 每篇带 note / note_status / pdf_url / relevance_score / pdf_cached
        """
        await self.validate_input(input_data)

        self.set_state("running")

        try:
            keywords_str = input_data["keywords"]
            logger.info(f"[Hunter] 原始 keywords 输入: {keywords_str!r}")
            import re as _re
            if isinstance(keywords_str, str):
                # 字符串：按分隔符拆分，保留空格
                raw = keywords_str.strip()
                tokens = _re.split(r'[，,、]+', raw)
                keywords = [t.strip() for t in tokens if t.strip()]
            else:
                # 列表：对每个元素再拆分后合并
                result = []
                for item in list(keywords_str):
                    parts = _re.split(r'[，,、]+', str(item).strip())
                    result.extend(t.strip() for t in parts if t.strip())
                keywords = result

            max_papers = input_data.get("max_papers", 10)
            sources = input_data.get("sources", ["arxiv"])
            days_back = input_data.get("days_back", 7)
            kb_id = input_data.get("kb_id")
            user_id = input_data.get("user_id", "anonymous")

            # 1. 搜索（不下载）
            papers = await self._search_only(
                keywords=keywords,
                max_papers=max_papers,
                days_back=days_back,
                sources=sources,
                kb_id=kb_id,
                user_id=user_id,
            )

            # 2. 并行触发"下载 PDF + 解析 + LLM 笔记"，带 cache 短路
            if papers:
                papers = await asyncio.gather(
                    *[self._fetch_and_generate_note(p) for p in papers],
                    return_exceptions=False,
                )

            self.set_state("completed")

            return {
                "status": "success",
                "total_found": len(papers),
                "papers": papers,
                "keywords": keywords,
                "kb_id": kb_id,
                "user_id": user_id,
            }

        except Exception as e:
            self.set_state("error")
            raise AgentException(f"Hunter Agent执行失败: {str(e)}")

    def get_required_fields(self) -> List[str]:
        return ["keywords"]

    # ============================================================
    # 1. 搜索（不下载）
    # ============================================================

    # 上游论文池拉取系数 + 单关键词最多拉的论文数（防止单次 arxiv query 超 API 上限）
    _FETCH_PADDING_FACTOR = 3       # 至少拉到 3×max_papers，给过滤留缓冲
    _FETCH_PADDING_FLOOR   = 5      # 已读量额外加 buffer
    _SINGLE_QUERY_CAP      = 100    # 单关键词最多拉 100 篇（arxiv 经验上限）

    @staticmethod
    def _filter_by_read_keys(papers: List[Dict], read_keys: set) -> List[Dict]:
        """过滤掉 paper_key 落在 read_keys 集合里的论文。

        paper_key = f"{paper['source']}:{paper['id']}"，与 user_paper_read_state 表的 paper_key 字段一致。

        与 _filter_by_read_state 的区别：本函数复用已经查过的 read_keys，
        不再单独打 DB。在 _search_only 里我们会在开头一次性查 read_keys，
        然后既用来计算 fetch_target，又用来过滤，节省一次 SQL。
        """
        if not read_keys:
            return papers
        filtered: List[Dict] = []
        for p in papers:
            source = p.get("source", "arxiv")
            pid = p.get("id", "")
            if not pid:
                # 没 id 的论文保留（无法判定为已读，宁可显示让用户决策）
                filtered.append(p)
                continue
            key = f"{source}:{pid}"
            if key in read_keys:
                continue
            filtered.append(p)
        return filtered

    async def _search_only(
        self,
        keywords: List[str],
        max_papers: int,
        days_back: int,
        sources: List[str],
        kb_id: Optional[str],
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """纯搜索：arxiv/ieee 搜索 + 去重 + KB 过滤 + 用户已读过滤 + 相关性排序。

        不下载 PDF、不入库 PG。

        v3 改进：动态 fetch_target
          - 如果用户已读集合很大，过滤后会"掏空"上游候选池，导致返回 0 篇
          - 解决：上游拉 max_papers + R（用户已读量） + buffer，且不超过单 query 上限
          - 截断发生在最后一步（ranked[:max_papers]），保证即便过滤很多仍能凑齐 max_papers

        过滤顺序：
          1. KB 去重（如果给了 kb_id）   → 排除已加入目标 KB 的论文
          2. 用户已读去重（如果给了 user_id）→ 排除该用户已读/已入库的论文
        """
        # ---- 1. 预先查用户已读集合（决定上游拉多少）----
        read_keys: set = set()
        if user_id:
            try:
                read_keys = await db_manager.list_read_paper_keys(user_id)
            except Exception as e:
                logger.warning(f"读取已读集合失败（按空集合处理）: {e}")
                read_keys = set()

        # ---- 2. 计算上游 fetch_target ----
        # 经验公式：上游至少要给 max_papers + R + buffer 才有保障
        # 注意：单 query 拉取量不超过 _SINGLE_QUERY_CAP（防 arxiv API 报错）
        padding = max(
            max_papers * self._FETCH_PADDING_FACTOR,
            max_papers + len(read_keys) + self._FETCH_PADDING_FLOOR,
        )
        fetch_target = min(padding, self._SINGLE_QUERY_CAP)

        all_papers: List[Dict] = []

        # ---- 3. 上游拉取 ----
        if "arxiv" in sources:
            arxiv_papers = await self._search_papers_from_arxiv(
                keywords, fetch_target, days_back
            )
            all_papers.extend(arxiv_papers)

        if "ieee" in sources:
            ieee_papers = await self._search_papers_from_ieee(
                keywords, fetch_target, days_back
            )
            all_papers.extend(ieee_papers)

        # 去重
        unique_papers = self._deduplicate_papers(all_papers)

        # 相关性评分 + 排序
        ranked = self._score_and_rank(unique_papers, keywords)

        # KB 去重（如果给了 kb_id）
        if kb_id:
            ranked = await self._filter_by_kb(ranked, kb_id)

        # 用户级已读去重（独立于 KB；只要有 user_id 就跑）
        # 复用上面预取的 read_keys，不再重复 DB 查询
        if user_id:
            ranked = self._filter_by_read_keys(ranked, read_keys)

        # 取 top N
        # 如果过滤后 < max_papers 篇，正常返回（候选池真没了，给用户清晰的"已找完"状态）
        return ranked[:max_papers]

    def _score_and_rank(self, papers: List[Dict], keywords: List[str]) -> List[Dict]:
        """计算 relevance_score 并排序。"""
        scored: List[Dict] = []
        for paper in papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            score = 0
            for kw in keywords:
                kwl = kw.lower()
                if kwl in title:
                    score += 2
                if kwl in abstract:
                    score += 1
            if score >= 1:
                paper["relevance_score"] = float(score)
                scored.append(paper)
        scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return scored

    async def _filter_by_kb(
        self, papers: List[Dict], kb_id: str
    ) -> List[Dict]:
        """过滤掉已经在目标 KB 中的论文。

        去重策略：
          1. 按 DOI 精确匹配 → 跳过
          2. 按标题模糊匹配 → 跳过
        Hunter 阶段还没有 content_hash，所以只能靠 DOI + title。
        """
        try:
            existing = await db_manager.list_paper_titles_and_dois_in_kb(kb_id)
        except Exception as e:
            logger.warning(f"KB 去重查询失败（跳过过滤）: {e}")
            return papers

        existing_dois = {
            (e.get("doi") or "").lower().strip()
            for e in existing
            if e.get("doi")
        }
        existing_titles_lower = [
            (e.get("title") or "").lower().strip() for e in existing
        ]

        filtered: List[Dict] = []
        for p in papers:
            p_doi = (p.get("doi") or "").lower().strip()
            p_title = (p.get("title") or "").lower().strip()

            if p_doi and p_doi in existing_dois:
                continue
            # 标题子串匹配（粗略）
            if any(
                p_title and p_title[:60] == t[:60]
                for t in existing_titles_lower if t
            ):
                continue

            filtered.append(p)

        logger.info(
            f"Hunter KB 去重: 输入 {len(papers)} 篇 → 输出 {len(filtered)} 篇"
        )
        return filtered

    async def _filter_by_read_state(
        self, papers: List[Dict], user_id: Optional[str]
    ) -> List[Dict]:
        """过滤掉用户已读的论文（KB 内或显式标记）。

        数据源：db_manager.list_read_paper_keys(user_id) → set[str]
        匹配规则：paper_key = f"{paper['source']}:{paper['id']}"

        异常处理：DB 不可用时降级（warn 后原样返回），不阻塞搜索主流程。

        实现细节：内部委托 _filter_by_read_keys 复用匹配逻辑，避免重复。
        """
        if not user_id:
            return papers
        try:
            read_keys = await db_manager.list_read_paper_keys(user_id)
        except Exception as e:
            logger.warning(f"读取已读集合失败（跳过过滤）: {e}")
            return papers

        filtered = self._filter_by_read_keys(papers, read_keys)
        logger.info(
            f"Hunter 用户已读去重 (user={user_id[:8]}…): "
            f"输入 {len(papers)} 篇 → 输出 {len(filtered)} 篇"
        )
        return filtered

    # ============================================================
    # 2. 下载 + 解析 + LLM 笔记生成
    # ============================================================

    async def _fetch_and_generate_note(self, paper: Dict) -> Dict:
        """下载 PDF → 解析 intro/methodology → LLM 生成笔记。

        笔记持久化到 PG hunter_notes 表（无 TTL）；
        PDF/parsed 继续走 HunterCache 文件 TTL。
        """
        arxiv_id = paper.get("id", "")
        if not arxiv_id:
            paper["note"] = ""
            paper["note_status"] = "failed"
            return paper

        # 1) 确保 PDF 已下载到 cache
        if not await self.cache.has_pdf(arxiv_id):
            pdf_url = paper.get("pdf_url")
            if pdf_url:
                try:
                    pdf_bytes = await self._download_pdf_bytes(pdf_url)
                    if pdf_bytes:
                        await self.cache.save_pdf(arxiv_id, pdf_bytes)
                        await self.cache.save_meta(
                            arxiv_id,
                            {
                                "title": paper.get("title", ""),
                                "authors": paper.get("authors", []),
                                "abstract": paper.get("abstract", ""),
                                "doi": paper.get("doi", ""),
                                "pdf_url": pdf_url,
                                "source": paper.get("source", "arxiv"),
                                "downloaded_at": datetime.now().isoformat(),
                            },
                        )
                except Exception as e:
                    logger.warning(f"下载 PDF 失败 {arxiv_id}: {e}")

        # 2) 解析 intro/methodology
        parsed = await self.cache.load_parsed(arxiv_id)
        contribution = ""
        methodology = ""
        if parsed:
            contribution = parsed.get("contribution", "")
            methodology = parsed.get("methodology", "")
        else:
            pdf_path = await self.cache.get_pdf_path(arxiv_id)
            if pdf_path and pdf_path.exists():
                try:
                    contribution, methodology = await self._parse_intro_method(
                        str(pdf_path)
                    )
                    await self.cache.save_parsed(
                        arxiv_id, contribution, methodology
                    )
                except Exception as e:
                    logger.warning(f"PDF 解析失败 {arxiv_id}: {e}")

        # 3) 生成笔记（优先从 PG hunter_notes 表读取；无则 LLM 生成并落库）
        note_text = ""
        try:
            # 3a) PG 命中短路
            cached = await db_manager.get_hunter_note(arxiv_id)
            if cached and cached.get("note"):
                note_text = cached["note"]
            else:
                # 3b) LLM 生成
                note_text = await self._generate_note_via_llm(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    contribution=contribution,
                    methodology=methodology,
                )
                if not note_text:
                    note_text = await self._generate_note_fallback(
                        title=paper.get("title", ""),
                        abstract=paper.get("abstract", ""),
                    )
                # 3c) 持久化落库（不设 TTL）
                if note_text:
                    await db_manager.save_hunter_note(
                        arxiv_id=arxiv_id,
                        note=note_text,
                        source=paper.get("source", "arxiv"),
                    )
        except Exception as e:
            logger.warning(f"笔记生成/存储失败 {arxiv_id}: {e}")
            note_text = ""

        paper["note"] = note_text
        paper["note_status"] = "ready" if note_text else "failed"
        paper["pdf_cached"] = await self.cache.has_pdf(arxiv_id)
        return paper

    async def _download_pdf_bytes(self, pdf_url: str) -> Optional[bytes]:
        """从 URL 下载 PDF 字节（带简单重试）。"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_url, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as response:
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
                    await asyncio.sleep(2 * (attempt + 1))
        return None

    async def _parse_intro_method(self, pdf_path: str) -> tuple[str, str]:
        """从 PDF 抽取 contribution + methodology 摘要。"""
        try:
            from utils.pdf_parser import pdf_parser

            pdf_result = await pdf_parser.parse_pdf(pdf_path)
            if not pdf_result.get("success"):
                return "", ""
            full_text = pdf_result.get("full_text", "")
            contribution = PaperIntroExtractor.extract_contribution(full_text)
            methodology = PaperIntroExtractor.extract_methodology(full_text)
            return contribution, methodology
        except Exception as e:
            logger.warning(f"_parse_intro_method 异常: {e}")
            return "", ""

    async def _generate_note_via_llm(
        self, title: str, abstract: str, contribution: str, methodology: str
    ) -> str:
        """主 prompt：基于 contribution + methodology 生成笔记。"""
        prompt = HUNTER_NOTE_PROMPT.format(
            title=title,
            abstract=abstract,
            contribution=contribution or "（未能抽取）",
            methodology=methodology or "（未能抽取）",
        )
        try:
            return await llm_adapter.ainvoke(prompt)
        except Exception as e:
            logger.warning(f"ainvoke 失败，回退 invoke: {e}")
            return llm_adapter.invoke(prompt)

    async def _generate_note_fallback(
        self, title: str, abstract: str
    ) -> str:
        """Fallback prompt：只基于 abstract 生成简短笔记。"""
        prompt = HUNTER_NOTE_FALLBACK_PROMPT.format(
            title=title, abstract=abstract or ""
        )
        try:
            return await llm_adapter.ainvoke(prompt)
        except Exception as e:
            logger.warning(f"ainvoke 失败，回退 invoke: {e}")
            return llm_adapter.invoke(prompt)

    # ============================================================
    # 原有搜索方法（保留）
    # ============================================================

    def _sanitize_keyword(self, kw: str) -> str:
        """清理关键词，移除 ArXiv 不接受的字符"""
        return re.sub(r'[，,、"\']+', ' ', kw).strip()

    async def _arxiv_single_query(self, query: str, start: int, max_results: int, days_back: int) -> List[Dict]:
        """执行单次 ArXiv 查询并解析结果"""
        date_filter = ""
        if days_back > 0:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")
            date_filter = f"submittedDate:[{start_date}0000 TO {end_date}2359]"

        full_query = f"({query}) AND ({date_filter})" if date_filter else query
        params = {
            "search_query": full_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.arxiv_base_url, params=params, ssl=False) as resp:
                        if resp.status == 429:
                            wait = 3 * (attempt + 1)
                            logger.warning(f"ArXiv 限流 (429)，第 {attempt+1}/{max_retries} 次重试，等待 {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            raise ExternalAPIException(f"ArXiv API请求失败: {resp.status}")
                        xml_content = await resp.text()
                        feed = feedparser.parse(xml_content)
                        papers = []
                        for entry in feed.entries:
                            papers.append({
                                "id": entry.id.split("/")[-1],
                                "title": entry.title,
                                "authors": [a.name for a in entry.authors],
                                "abstract": entry.summary,
                                "published": entry.published,
                                "pdf_url": entry.link.replace('/abs/', '/pdf/') + '.pdf',
                                "source": "arxiv",
                                "doi": entry.get('arxiv_doi', ''),
                                "categories": [t.term for t in entry.tags]
                            })
                        return papers
            except ExternalAPIException:
                raise
            except Exception as e:
                logger.warning(f"ArXiv 请求异常 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"ArXiv search failed after {max_retries} retries: {e}")
                    self._add_to_history(f"ArXiv search failed: {e}")
        return []

    async def _search_papers_from_arxiv(self, keywords: List[str], max_papers: int, days_back: int) -> List[Dict]:
        """从 ArXiv 搜索论文，按关键词独立加权评分排序。

        评分公式：
          score(paper) = Σ weight[i] × relevance(paper, keyword[i])

        其中：
          - weight[i] = (n - i) × 0.5     # 位置越靠前权重越高，间隔 0.5
          - relevance() = 在 abstract 得 1.0，在 title 得 2.0，完整 phrase 得 3.0
        """
        if not keywords:
            return []

        n = len(keywords)
        per_kw = max(max_papers * 2, 20)

        # ---------- 第一轮：每个关键词单独搜索 ----------
        raw_results: List[Tuple[List[Dict], int]] = []
        for idx, kw in enumerate(keywords):
            sanitized = self._sanitize_keyword(kw)
            if not sanitized:
                continue
            query = f'all:"{sanitized}"'
            papers = await self._arxiv_single_query(query, 0, per_kw, days_back)
            raw_results.append((papers, idx))

        # ---------- 去重 ----------
        seen_ids = set()
        unique: List[Dict] = []
        for papers, _ in raw_results:
            for p in papers:
                pid = p["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    unique.append(p)

        # ---------- 第二轮：每个关键词独立计算相关分 ----------
        kw_patterns = [
            (idx, re.compile(re.escape(self._sanitize_keyword(kw)), re.IGNORECASE))
            for idx, kw in enumerate(keywords)
            if self._sanitize_keyword(kw)
        ]

        for p in unique:
            title_lower = p.get("title", "").lower()
            abstract_lower = p.get("abstract", "").lower()

            kw_scores: List[float] = []
            for idx, pat in kw_patterns:
                # 独立相关分：abstract命中1.0，title命中2.0，phrase完整命中3.0
                title_hit = bool(pat.search(title_lower))
                abstract_hit = bool(pat.search(abstract_lower))

                if title_hit and abstract_hit:
                    score = 3.0
                elif title_hit:
                    score = 2.0
                elif abstract_hit:
                    score = 1.0
                else:
                    score = 0.0
                kw_scores.append(score)

            p["_kw_scores"] = kw_scores

        # ---------- 加权求和 ----------
        for p in unique:
            scores = p.get("_kw_scores", [])
            total = sum((n - idx) * 0.5 * s for idx, s in enumerate(scores))
            p["_weighted_score"] = total

        # ---------- 排序 ----------
        unique.sort(key=lambda p: p.get("_weighted_score", 0.0), reverse=True)

        # ---------- 清理临时字段 ----------
        for p in unique:
            p.pop("_kw_scores", None)
            p.pop("_weighted_score", None)

        return unique[:max_papers]

    def _kw_match_key(self, p: Dict) -> Tuple[int, List[int]]:
        return (p["_kw_match_count"], p["_kw_indices"])

    async def _search_papers_from_arxiv_orig(self, keywords: List[str], max_papers: int, days_back: int) -> List[Dict]:
        """原始单次 OR 查询，保留以备兼容"""
        papers = []
        query_parts = []
        for keyword in keywords:
            sanitized = self._sanitize_keyword(keyword)
            if sanitized:
                query_parts.append(f'all:"{sanitized}"')
        query = " OR ".join(query_parts)
        return await self._arxiv_single_query(query, 0, max_papers * 2, days_back)

    async def _search_papers_from_ieee(self, keywords: List[str], max_papers: int, days_back: int) -> List[Dict]:
        """从IEEE搜索论文"""
        papers = []
        config = self.config.external_apis
        if not config.ieee_base_url:
            logger.warning("IEEE API配置缺失，跳过IEEE搜索")
            self._add_to_history("IEEE API配置缺失，跳过IEEE搜索")
            return papers
        query = " OR ".join([f'"All Meta Data:{kw}"' for kw in keywords])
        params = {
            "apikey": config.ieee_api_key or "",
            "querytext": query,
            "max_records": max_papers * 2,
            "start_record": 1,
            "sort_order": "desc",
            "sort_field": "publication_date"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ieee_base_url, params=params, ssl=False) as response:
                    if response.status != 200:
                        raise ExternalAPIException(f"IEEE API请求失败: {response.status}")
                    
                    data = await response.json()
                    
                    for article in data.get("articles", []):
                        paper = {
                            "id": article.get("article_number", ""),
                            "title": article.get("title", ""),
                            "authors": [author.get("full_name", "") for author in article.get("authors", {}).get("authors", [])],
                            "abstract": article.get("abstract", ""),
                            "published": article.get("publication_date", ""),
                            "pdf_url": article.get("pdf_url", ""),
                            "source": "ieee",
                            "doi": article.get("doi", ""),
                            "categories": article.get("index_terms", {}).get("ieee_terms", {}).get("terms", [])
                        }
                        
                        papers.append(paper)
                        
        except Exception as e:
            self._add_to_history(f"IEEE搜索失败: {str(e)}")
        
        return papers
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """去重论文"""
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            title = paper.get("title", "").lower().strip()
            title_hash = hashlib.md5(title.encode()).hexdigest()

            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_papers.append(paper)

        return unique_papers
    
    # 工具方法
    async def _search_arxiv(self, query: str) -> List[Dict]:
        """搜索ArXiv工具"""
        import re as _re
        tokens = _re.split(r'[，,、]+', query.strip())
        keywords = [kw.strip() for kw in tokens if kw.strip()]
        return await self._search_papers_from_arxiv(keywords, 10, 7)
    
    async def _search_ieee(self, query: str) -> List[Dict]:
        """搜索IEEE工具"""
        import re as _ieee_re
        tokens = _ieee_re.split(r'[，,、]+', query.strip())
        keywords = [kw.strip() for kw in tokens if kw.strip()]
        return await self._search_papers_from_ieee(keywords, 10, 7)
    
    async def _download_pdf(self, pdf_url: str) -> str:
        """下载PDF工具"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, ssl=False) as response:
                    if response.status == 200:
                        content = await response.read()
                        filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        file_path = os.path.join(self.download_dir, filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        return file_path
                    else:
                        return f"下载失败，状态码: {response.status}"
        except Exception as e:
            return f"下载异常: {str(e)}"
    
    async def _extract_metadata(self, file_path: str) -> Dict:
        """提取论文元数据工具"""
        # 这里应该使用PDF解析库提取元数据
        # 暂时返回基础信息
        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "extracted_at": datetime.now().isoformat()
        }
