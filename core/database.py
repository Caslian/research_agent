"""
InnoCore AI 数据库管理模块
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    asyncpg = None
    HAS_ASYNCPG = False


def _serialize_user_row(row: Optional[Dict]) -> Optional[Dict]:
    """users 表里 profile 是 JSONB,asyncpg 取出来时仍是 str,反序列化。"""
    if not row:
        return row
    profile = row.get("profile")
    if isinstance(profile, str):
        try:
            row["profile"] = json.loads(profile) if profile else {}
        except Exception:
            row["profile"] = {}
    elif profile is None:
        row["profile"] = {}
    return row

from .config import get_config
from .exceptions import DatabaseException

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.config = get_config().database
        self.pool = None
    
    async def initialize(self):
        """初始化数据库连接池"""
        if not HAS_ASYNCPG:
            logger.warning("asyncpg 未安装，数据库功能不可用")
            return
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=1,
                max_size=self.config.pool_size
            )
            await self._create_tables()
            logger.info(f"PostgreSQL 初始化完成: {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.warning(f"数据库初始化失败（将以无数据库模式运行）: {str(e)}")
    
    async def _create_tables(self):
        """创建数据库表"""
        create_tables_sql = """
        -- ============================================================
        -- 用户表：账号信息 + JSONB profile（风格偏好、研究领域等）
        -- ============================================================
        CREATE TABLE IF NOT EXISTS users (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- 用户唯一 ID（UUID）
            email       VARCHAR(255) UNIQUE NOT NULL,                 -- 邮箱（登录凭证，唯一）
            profile     JSONB DEFAULT '{}',                          -- 用户配置（JSONB：风格偏好/研究领域等）
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP          -- 账号创建时间
        );

        -- ============================================================
        -- 论文表：Hunter 下载 / PDF 上传的论文元数据
        -- ============================================================
        CREATE TABLE IF NOT EXISTS papers (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- 论文唯一 ID（UUID）
            title         TEXT NOT NULL,                                -- 论文标题
            authors       TEXT[] DEFAULT '{}',                          -- 作者列表（TEXT 数组）
            abstract      TEXT,                                         -- 摘要全文
            doi           VARCHAR(255) UNIQUE,                          -- DOI 编号（唯一，可空）
            file_path     TEXT,                                         -- 本地 PDF 文件路径
            content_hash  VARCHAR(64) UNIQUE,                           -- PDF 内容 SHA-256（用于去重）
            is_preset     BOOLEAN DEFAULT FALSE,                        -- 是否预置到 L1 全局向量库
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP           -- 入库时间
        );

        -- ============================================================
        -- Hunter 笔记表：Hunter 阶段生成的论文技术路线笔记（持久化，不设 TTL）
        -- 备注：Hunter 阶段论文尚未入库 papers 表，所以用 arxiv_id 作主键。
        --        用户将论文入库时，通过 content_hash / doi 回填 paper_id。
        -- ============================================================
        CREATE TABLE IF NOT EXISTS hunter_notes (
            arxiv_id     VARCHAR(120) PRIMARY KEY,                      -- arxiv ID（主键，Hunter 阶段即有）
            paper_id     UUID REFERENCES papers(id) ON DELETE SET NULL, -- 论文入库后回填（可空，Hunter 阶段为空）
            source       VARCHAR(20)  DEFAULT 'arxiv',                  -- 来源：arxiv / ieee
            note         TEXT NOT NULL,                                 -- LLM 生成的中文技术路线笔记
            model        VARCHAR(60)  DEFAULT '',                       -- 生成笔记所用的模型名
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,           -- 笔记生成时间
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP            -- 更新时间（入库回填 paper_id 时更新）
        );

        -- ============================================================
        -- 用户-论文关系表：多对多 + 个性化标注
        -- ============================================================
        CREATE TABLE IF NOT EXISTS user_paper_relations (
            id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),                  -- 关系记录 ID
            user_id   UUID REFERENCES users(id) ON DELETE CASCADE,                  -- 所属用户 ID（外键，级联删除）
            paper_id  UUID REFERENCES papers(id) ON DELETE CASCADE,                 -- 论文 ID（外键，级联删除）
            tags      TEXT[] DEFAULT '{}',                                          -- 用户自定义标签
            rating    INTEGER DEFAULT 0,                                            -- 用户评分（0-5 等）
            is_read   BOOLEAN DEFAULT FALSE,                                        -- 是否已读
            added_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                          -- 加入时间
            UNIQUE(user_id, paper_id)                                               -- 同一用户同一论文只能存在一条
        );

        -- ============================================================
        -- 分析报告表：Miner 生成的报告（四段式）
        -- ============================================================
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- 报告唯一 ID
            paper_id               UUID REFERENCES papers(id) ON DELETE CASCADE,-- 关联论文 ID（级联删除）
            generated_for_user_id  UUID REFERENCES users(id) ON DELETE SET NULL,-- 为哪个用户生成（SET NULL：用户删了报告保留）
            summary                TEXT,                                       -- 论文摘要总结
            innovation_point       TEXT,                                       -- 创新点
            limitation             TEXT,                                       -- 局限性
            future_idea            TEXT,                                       -- 未来工作/改进方向
            vector_ids             JSONB DEFAULT '{}',                          -- 写入 Qdrant L1/L2 的点 ID（JSONB）
            created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP          -- 生成时间
        );

        -- ============================================================
        -- 引用缓存表：按 DOI 缓存 BibTeX
        -- ============================================================
        CREATE TABLE IF NOT EXISTS reference_cache (
            doi          VARCHAR(255) PRIMARY KEY,                       -- DOI（主键）
            bibtex_std   TEXT,                                          -- 标准 BibTeX 字符串
            is_verified  BOOLEAN DEFAULT FALSE,                         -- 是否经过 CrossRef/Scholar 联网核对
            last_check   TIMESTAMP DEFAULT CURRENT_TIMESTAMP            -- 上次校验时间
        );

        -- ============================================================
        -- Agent 执行日志表：每个 Agent 任务的执行记录
        -- ============================================================
        CREATE TABLE IF NOT EXISTS agent_execution_logs (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- 日志记录 ID
            agent_name      VARCHAR(50) NOT NULL,                        -- Agent 名（hunter/miner/coach/validator）
            task_type       VARCHAR(50),                                  -- 任务类型（full_workflow/search/analyze/...）
            task_id         VARCHAR(100),                                 -- Controller 分配的任务 ID
            input_summary   TEXT,                                         -- 输入摘要
            output_summary  TEXT,                                         -- 输出摘要
            tools_called    JSONB DEFAULT '[]',                           -- 调用过的工具列表（JSONB 数组）
            status          VARCHAR(20) DEFAULT 'running',                -- 任务状态（running/completed/failed）
            started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,          -- 开始时间
            completed_at    TIMESTAMP,                                   -- 完成时间
            duration_ms     INTEGER,                                      -- 任务耗时（毫秒）
            error_message   TEXT                                          -- 失败时的错误信息
        );

        -- ============================================================
        -- Agent 工具调用详情表：每次工具调用的输入/输出/耗时
        -- ============================================================
        CREATE TABLE IF NOT EXISTS agent_tool_calls (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),                  -- 调用记录 ID
            execution_id  UUID REFERENCES agent_execution_logs(id) ON DELETE CASCADE, -- 关联的 Agent 执行日志 ID（级联删除）
            tool_name     VARCHAR(100) NOT NULL,                                      -- 工具名
            tool_input    JSONB,                                                      -- 工具输入参数（JSONB）
            tool_output   JSONB,                                                      -- 工具返回结果（JSONB）
            called_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                        -- 调用时间
            duration_ms   INTEGER                                                      -- 工具调用耗时（毫秒）
        );

        -- ============================================================
        -- 工作流执行表：完整 workflow 的整体状态
        -- ============================================================
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- 工作流执行 ID
            user_id         UUID REFERENCES users(id),                   -- 发起用户（不级联删除，保留历史）
            workflow_type   VARCHAR(50) NOT NULL,                        -- 类型（full_workflow/search_only/analyze_only）
            status          VARCHAR(20) DEFAULT 'running',                -- 整体状态
            steps           JSONB DEFAULT '[]',                          -- 各阶段执行情况（JSONB 数组）
            result          JSONB,                                       -- 最终聚合结果（JSONB）
            started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,          -- 开始时间
            completed_at    TIMESTAMP                                    -- 完成时间
        );

        -- ============================================================
        -- 知识库表：每个用户可建多个 KB；RAG 检索的隔离边界
        -- ============================================================
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- KB ID（Qdrant payload.kb_id 用此）
            user_id           UUID REFERENCES users(id) ON DELETE CASCADE,  -- 所有者（级联删除 KB 及其向量）
            name              VARCHAR(120) NOT NULL,                       -- KB 名称
            description       TEXT DEFAULT '',                             -- KB 描述
            embedding_model   VARCHAR(120) DEFAULT 'text-embedding-v4',    -- 关联的 embedding 模型名（备用）
            paper_count       INTEGER DEFAULT 0,                           -- 论文计数（缓存，写入 chunks 后 +1）
            chunk_count       INTEGER DEFAULT 0,                           -- chunk 计数
            is_default        BOOLEAN DEFAULT FALSE,                       -- 用户是否标记为默认 KB
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, name)
        );

        -- ============================================================
        -- KB-论文关系表：等价于 user_paper_relations 的多对多升级
        -- 一篇 paper 可以同时属于多个 KB（关键设计点：避免 paper 表本身被打 kb_id）
        -- ============================================================
        CREATE TABLE IF NOT EXISTS kb_paper_relations (
            id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_id     UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            paper_id  UUID REFERENCES papers(id) ON DELETE CASCADE,
            added_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            chunk_count INTEGER DEFAULT 0,            -- 写入向量库的 chunk 数（与 Qdrant 物理 chunk 数对齐）
            UNIQUE(kb_id, paper_id)
        );

        -- ============================================================
        -- KB 同步日志表：每次 chunk 入库的尝试记录（成功/失败/重试次数）
        -- ============================================================
        CREATE TABLE IF NOT EXISTS kb_paper_chunks_log (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_id        UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            paper_id     UUID REFERENCES papers(id) ON DELETE CASCADE,
            chunk_count  INTEGER DEFAULT 0,
            status       VARCHAR(20) DEFAULT 'processing',  -- processing / ready / failed
            error_message TEXT,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );

        -- ============================================================
        -- 索引：加速常用查询
        --   - papers: content_hash / doi（去重 & 按 DOI 查找）
        --   - user_paper_relations: user_id / paper_id（多对多查询）
        --   - analysis_reports: paper_id / generated_for_user_id
        --   - agent_execution_logs: agent_name / task_id / status
        --   - workflow_executions: status
        -- ============================================================
        CREATE INDEX IF NOT EXISTS idx_papers_content_hash            ON papers(content_hash);                -- 论文按 SHA-256 去重查找
        CREATE INDEX IF NOT EXISTS idx_papers_doi                     ON papers(doi);                         -- 论文按 DOI 查找
        CREATE INDEX IF NOT EXISTS idx_user_paper_relations_user_id   ON user_paper_relations(user_id);       -- 用户-论文关系按用户过滤
        CREATE INDEX IF NOT EXISTS idx_user_paper_relations_paper_id  ON user_paper_relations(paper_id);      -- 用户-论文关系按论文过滤
        CREATE INDEX IF NOT EXISTS idx_analysis_reports_paper_id      ON analysis_reports(paper_id);          -- 分析报告按论文查找
        CREATE INDEX IF NOT EXISTS idx_analysis_reports_user_id       ON analysis_reports(generated_for_user_id); -- 分析报告按用户查找
        CREATE INDEX IF NOT EXISTS idx_agent_logs_agent               ON agent_execution_logs(agent_name);    -- Agent 日志按 Agent 名查找
        CREATE INDEX IF NOT EXISTS idx_agent_logs_task_id             ON agent_execution_logs(task_id);       -- Agent 日志按任务 ID 查找
        CREATE INDEX IF NOT EXISTS idx_agent_logs_status              ON agent_execution_logs(status);        -- Agent 日志按状态过滤
        CREATE INDEX IF NOT EXISTS idx_workflow_status                ON workflow_executions(status);         -- 工作流按状态过滤
        CREATE INDEX IF NOT EXISTS idx_kb_user_id                     ON knowledge_bases(user_id);            -- KB 按用户过滤
        CREATE INDEX IF NOT EXISTS idx_kb_paper_relations_kb          ON kb_paper_relations(kb_id);           -- KB 论文关系按 KB 过滤
        CREATE INDEX IF NOT EXISTS idx_kb_paper_relations_paper       ON kb_paper_relations(paper_id);        -- KB 论文关系按论文过滤
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_log_kb_paper         ON kb_paper_chunks_log(kb_id, paper_id);-- 同步日志查找
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_log_status           ON kb_paper_chunks_log(status);         -- 同步日志按状态过滤
        CREATE INDEX IF NOT EXISTS idx_hunter_notes_paper_id          ON hunter_notes(paper_id);             -- 按 paper_id 查找（入库后回填）

        -- ============================================================
        -- 用户论文已读状态表：hunter 搜索结果去重（用户级全局）
        --
        -- 触发场景：
        --   1. 用户将论文加入 KB  → reason='added_to_kb'（被动，写入时记录 kb_id/paper_id）
        --   2. 用户主动点击"标记已读" → reason='marked_read'（主动）
        -- 永久只插入（ON CONFLICT DO NOTHING），不做撤回。
        -- 标识键 paper_key = "{source}:{id}"（如 'arxiv:2401.01234' / 'ieee:9876543'），
        -- 因为 Hunter 阶段 paper 尚未入库，paper_id 不可用，需要用 source+id 自构造。
        -- ============================================================
        CREATE TABLE IF NOT EXISTS user_paper_read_state (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
            paper_key   VARCHAR(160) NOT NULL,                         -- 'arxiv:2401.01234' / 'ieee:9876543'
            source      VARCHAR(20)  NOT NULL,                         -- arxiv / ieee
            reason      VARCHAR(20)  NOT NULL,                         -- 'added_to_kb' / 'marked_read'
            kb_id       UUID REFERENCES knowledge_bases(id) ON DELETE SET NULL,  -- 来源 KB（可空）
            paper_id    UUID REFERENCES papers(id) ON DELETE SET NULL,           -- 关联 paper（可空）
            title       TEXT NOT NULL,                                 -- 冗余存储标题，避免 paper 清理后无据可查
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, paper_key)                                 -- 用户+论文键唯一
        );
        CREATE INDEX IF NOT EXISTS idx_read_state_user                 ON user_paper_read_state(user_id);
        CREATE INDEX IF NOT EXISTS idx_read_state_user_kb              ON user_paper_read_state(user_id, kb_id);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_tables_sql)
    
    @asynccontextmanager
    async def get_connection(self):
        """获取数据库连接"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                raise DatabaseException(f"数据库操作失败: {str(e)}")

    # ============================================================
    # 顶层便捷封装 - 让上层代码不必每次都拿 connection
    # ============================================================

    async def execute(self, sql: str, *args) -> str:
        """返回 PG 状态字符串（'INSERT 0 1' / 'UPDATE 1' 等）。args 用 $1,$2 风格。"""
        async with self.get_connection() as conn:
            return await conn.execute(sql, *args)

    async def fetchval(self, sql: str, *args):
        """单值返回，自动转 str(UUID)。args 用 $1,$2 风格。"""
        async with self.get_connection() as conn:
            v = await conn.fetchval(sql, *args)
            return str(v) if v is not None else None

    async def fetchrow(self, sql: str, *args) -> Optional[Dict]:
        """单行字典结果。"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(sql, *args)
            return dict(row) if row else None

    async def fetch(self, sql: str, *args) -> List[Dict]:
        """多行字典结果列表。"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(r) for r in rows]

    # 用户相关操作
    async def create_user(self, email: str, profile: Dict = None) -> str:
        """创建用户"""
        async with self.get_connection() as conn:
            user_id = await conn.fetchval(
                "INSERT INTO users (email, profile) VALUES ($1, $2) RETURNING id",
                email, json.dumps(profile or {})
            )
            return str(user_id)
    
    async def get_user(self, user_id: str) -> Optional[Dict]:
        """获取用户信息"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1", user_id
            )
            return _serialize_user_row(dict(row)) if row else None
    
    async def update_user_profile(self, user_id: str, profile: Dict) -> bool:
        """更新用户配置"""
        async with self.get_connection() as conn:
            result = await conn.execute(
                "UPDATE users SET profile = $1 WHERE id = $2",
                json.dumps(profile), user_id
            )
            return result == "UPDATE 1"
    
    # 论文相关操作
    async def create_paper(self, title: str, authors: List[str], 
                          abstract: str = None, doi: str = None,
                          file_path: str = None, content_hash: str = None,
                          is_preset: bool = False) -> str:
        """创建论文记录"""
        async with self.get_connection() as conn:
            paper_id = await conn.fetchval(
                """
                INSERT INTO papers (title, authors, abstract, doi, file_path, content_hash, is_preset)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                title, authors, abstract, doi, file_path, content_hash, is_preset
            )
            return str(paper_id)
    
    async def get_paper(self, paper_id: str) -> Optional[Dict]:
        """获取论文信息"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM papers WHERE id = $1", paper_id
            )
            return dict(row) if row else None
    
    async def get_paper_by_hash(self, content_hash: str) -> Optional[Dict]:
        """根据内容哈希获取论文"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM papers WHERE content_hash = $1", content_hash
            )
            return dict(row) if row else None
    
    async def search_papers(self, query: str, limit: int = 10, offset: int = 0) -> List[Dict]:
        """搜索论文"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM papers 
                WHERE title ILIKE $1 OR abstract ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                f"%{query}%", limit, offset
            )
            return [dict(row) for row in rows]

    # ---- Hunter 笔记 ----

    async def save_hunter_note(
        self,
        arxiv_id: str,
        note: str,
        source: str = "arxiv",
        model: str = "",
    ) -> bool:
        """保存或更新 Hunter 阶段生成的笔记（upsert）。

        注意：Hunter 阶段 paper_id 未知，所以暂不填，入库时再回填。
        """
        try:
            await self.execute(
                """
                INSERT INTO hunter_notes (arxiv_id, source, note, model, generated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    note = EXCLUDED.note,
                    model = EXCLUDED.model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                arxiv_id, source, note, model,
            )
            return True
        except Exception as e:
            logger.warning(f"save_hunter_note 失败 {arxiv_id}: {e}")
            return False

    async def get_hunter_note(self, arxiv_id: str) -> Optional[Dict]:
        """按 arxiv_id 读取笔记。"""
        return await self.fetchrow(
            "SELECT * FROM hunter_notes WHERE arxiv_id = $1", arxiv_id
        )

    async def get_hunter_notes(self, limit: int = 50) -> List[Dict]:
        """读取所有笔记（最近生成优先）。"""
        return await self.fetch(
            """
            SELECT * FROM hunter_notes
            ORDER BY generated_at DESC
            LIMIT $1
            """,
            limit,
        )

    async def bind_paper_to_hunter_note(self, arxiv_id: str, paper_id: str) -> bool:
        """用户将论文入库后，回填 paper_id 到 hunter_notes。"""
        try:
            r = await self.execute(
                """
                UPDATE hunter_notes SET paper_id = $1, updated_at = CURRENT_TIMESTAMP
                WHERE arxiv_id = $2 AND paper_id IS NULL
                """,
                paper_id, arxiv_id,
            )
            return r == "UPDATE 1"
        except Exception as e:
            logger.warning(f"bind_paper_to_hunter_note 失败: {e}")
            return False

    # ---- 用户论文已读状态（user-level dedup for Hunter 搜索）----

    async def mark_paper_read(
        self,
        user_id: str,
        paper_key: str,
        source: str,
        reason: str,
        title: str,
        kb_id: Optional[str] = None,
        paper_id: Optional[str] = None,
    ) -> bool:
        """将一篇论文标记为该用户的"已读"（INSERT-only，幂等）。

        Args:
            user_id:    用户 ID（UUID 字符串）
            paper_key:  '{source}:{id}' 形式，如 'arxiv:2401.01234' / 'ieee:9876543'
            source:     'arxiv' / 'ieee'
            reason:     'added_to_kb' / 'marked_read'
            title:      论文标题（冗余存储）
            kb_id:      若来自 KB 入库，记录 KB（可空）
            paper_id:   若已入库，回填 paper_id（可空）

        Returns:
            True 表示执行成功（不区分是否实际新增，幂等即可）
        """
        try:
            await self.execute(
                """
                INSERT INTO user_paper_read_state
                    (user_id, paper_key, source, reason, kb_id, paper_id, title)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (user_id, paper_key) DO NOTHING
                """,
                user_id, paper_key, source, reason, kb_id, paper_id, title or "",
            )
            return True
        except Exception as e:
            logger.warning(f"mark_paper_read 失败 user={user_id} key={paper_key}: {e}")
            return False

    async def list_read_paper_keys(self, user_id: str) -> set:
        """返回该用户所有已读论文的 paper_key 集合。

        Hunter 搜索阶段用：O(1) 查表过滤，避免对每篇 paper 单独 SQL。
        """
        try:
            rows = await self.fetch(
                "SELECT paper_key FROM user_paper_read_state WHERE user_id = $1",
                user_id,
            )
            return {r["paper_key"] for r in rows}
        except Exception as e:
            logger.warning(f"list_read_paper_keys 失败 user={user_id}: {e}")
            return set()

    async def list_read_papers_for_user(
        self, user_id: str, limit: int = 200
    ) -> List[Dict]:
        """前端展示该用户的已读论文清单（按时间倒序）。"""
        return await self.fetch(
            """
            SELECT paper_key, source, reason, title, kb_id, paper_id, created_at
            FROM user_paper_read_state
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )

    # 用户论文关系操作
    async def add_paper_to_user(self, user_id: str, paper_id: str, 
                               tags: List[str] = None, rating: int = 0) -> bool:
        """将论文添加到用户库"""
        async with self.get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO user_paper_relations (user_id, paper_id, tags, rating)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, paper_id) DO UPDATE SET
                        tags = EXCLUDED.tags,
                        rating = EXCLUDED.rating,
                        added_at = CURRENT_TIMESTAMP
                    """,
                    user_id, paper_id, tags or [], rating
                )
                return True
            except Exception:
                return False
    
    async def get_user_papers(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """获取用户的论文列表"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT p.*, upr.tags, upr.rating, upr.is_read, upr.added_at
                FROM papers p
                JOIN user_paper_relations upr ON p.id = upr.paper_id
                WHERE upr.user_id = $1
                ORDER BY upr.added_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id, limit, offset
            )
            return [dict(row) for row in rows]
    
    # 分析报告操作
    async def create_analysis_report(self, paper_id: str, summary: str,
                                   innovation_point: str, limitation: str,
                                   future_idea: str, vector_ids: Dict = None,
                                   user_id: str = None) -> str:
        """创建分析报告"""
        async with self.get_connection() as conn:
            report_id = await conn.fetchval(
                """
                INSERT INTO analysis_reports 
                (paper_id, generated_for_user_id, summary, innovation_point, limitation, future_idea, vector_ids)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                paper_id, user_id, summary, innovation_point, 
                limitation, future_idea, json.dumps(vector_ids or {})
            )
            return str(report_id)
    
    async def get_analysis_report(self, paper_id: str, user_id: str = None) -> Optional[Dict]:
        """获取分析报告"""
        async with self.get_connection() as conn:
            if user_id:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM analysis_reports 
                    WHERE paper_id = $1 AND (generated_for_user_id = $2 OR generated_for_user_id IS NULL)
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    paper_id, user_id
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM analysis_reports 
                    WHERE paper_id = $1 AND generated_for_user_id IS NULL
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    paper_id
                )
            return dict(row) if row else None
    
    # 引用缓存操作
    async def cache_reference(self, doi: str, bibtex: str, is_verified: bool = False):
        """缓存引用信息"""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO reference_cache (doi, bibtex_std, is_verified, last_check)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (doi) DO UPDATE SET
                    bibtex_std = EXCLUDED.bibtex_std,
                    is_verified = EXCLUDED.is_verified,
                    last_check = CURRENT_TIMESTAMP
                """,
                doi, bibtex, is_verified
            )
    
    async def get_cached_reference(self, doi: str) -> Optional[Dict]:
        """获取缓存的引用信息"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reference_cache WHERE doi = $1", doi
            )
            return dict(row) if row else None
    
    # ---- Agent 执行日志 ----
    async def log_agent_execution(
        self, agent_name: str, task_type: str = None, task_id: str = None,
        input_summary: str = None, status: str = "running"
    ) -> str:
        async with self.get_connection() as conn:
            exec_id = await conn.fetchval(
                """INSERT INTO agent_execution_logs
                   (agent_name, task_type, task_id, input_summary, status)
                   VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                agent_name, task_type, task_id, input_summary, status
            )
            return str(exec_id)

    async def update_agent_execution(
        self, execution_id: str, status: str, output_summary: str = None,
        tools_called: List = None, duration_ms: int = None, error_message: str = None
    ) -> bool:
        async with self.get_connection() as conn:
            result = await conn.execute(
                """UPDATE agent_execution_logs
                   SET status=$1, output_summary=$2, tools_called=$3,
                       duration_ms=$4, error_message=$5, completed_at=CURRENT_TIMESTAMP
                   WHERE id=$6""",
                status, output_summary, json.dumps(tools_called or []),
                duration_ms, error_message, execution_id
            )
            return result == "UPDATE 1"

    async def log_tool_call(
        self, execution_id: str, tool_name: str,
        tool_input: Dict = None, tool_output: Any = None, duration_ms: int = None
    ) -> str:
        async with self.get_connection() as conn:
            call_id = await conn.fetchval(
                """INSERT INTO agent_tool_calls
                   (execution_id, tool_name, tool_input, tool_output, duration_ms)
                   VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                execution_id, tool_name,
                json.dumps(tool_input or {}),
                json.dumps(tool_output or {}, default=str),
                duration_ms
            )
            return str(call_id)

    # ---- 工作流执行 ----
    async def create_workflow(self, user_id: str, workflow_type: str, steps: List = None) -> str:
        async with self.get_connection() as conn:
            wf_id = await conn.fetchval(
                """INSERT INTO workflow_executions (user_id, workflow_type, steps)
                   VALUES ($1, $2, $3) RETURNING id""",
                user_id, workflow_type, json.dumps(steps or [])
            )
            return str(wf_id)

    async def update_workflow(self, workflow_id: str, status: str, result: Dict = None) -> bool:
        async with self.get_connection() as conn:
            r = await conn.execute(
                """UPDATE workflow_executions
                   SET status=$1, result=$2, completed_at=CURRENT_TIMESTAMP
                   WHERE id=$3""",
                status, json.dumps(result or {}), workflow_id
            )
            return r == "UPDATE 1"

    async def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflow_executions WHERE id = $1", workflow_id
            )
            return dict(row) if row else None

    # ---- 数据库状态查询 ----
    async def get_table_counts(self) -> Dict[str, int]:
        tables = ["papers", "users", "user_paper_relations", "analysis_reports",
                  "agent_execution_logs", "agent_tool_calls", "workflow_executions"]
        counts = {}
        async with self.get_connection() as conn:
            for table in tables:
                row = await conn.fetchrow(f"SELECT COUNT(*) as cnt FROM {table}")
                counts[table] = row["cnt"] if row else 0
        return counts

    # ---- KB 去重查询（Hunter 阶段使用）----

    async def list_paper_titles_and_dois_in_kb(self, kb_id: str) -> List[Dict[str, Any]]:
        """列出 KB 内所有论文的 (id, title, doi, content_hash)。

        用于 Hunter 阶段: 拿到 arxiv 论文后,用 title/doi 在这里做命中检查。
        注: arxiv 论文 Hunter 阶段还没有 content_hash,所以只能在 DOI 上做精确匹配。
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT p.id, p.title, p.doi, p.content_hash
                FROM papers p
                JOIN kb_paper_relations r ON r.paper_id = p.id
                WHERE r.kb_id = $1
                """,
                kb_id,
            )
            return [dict(r) for r in rows]

    async def get_paper_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """按 DOI 查论文（精确匹配）。"""
        if not doi:
            return None
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM papers WHERE doi = $1 LIMIT 1", doi,
            )
            return dict(row) if row else None

    async def get_paper_by_title_fuzzy(self, title: str) -> Optional[Dict[str, Any]]:
        """按标题近似匹配（ILIKE 包含）。"""
        if not title:
            return None
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM papers WHERE title ILIKE $1 LIMIT 1",
                f"%{title.strip()[:120]}%",
            )
            return dict(row) if row else None

    async def close(self):
        """关闭数据库连接池"""
        if self.pool:
            await self.pool.close()

# 全局数据库管理器实例
db_manager = DatabaseManager()