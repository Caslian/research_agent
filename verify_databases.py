#!/usr/bin/env python3
"""
============================================================
InnoCore AI 数据库验证脚本 - v1.0
============================================================
验证范围:
  PostgreSQL (7 张表) — 连接 / 表结构 / CRUD 操作 / 事务
  Redis              — 连接 / 任务队列 / 缓存 / Agent 状态 / Pub/Sub
  Qdrant             — 连接 / 向量写入 / 混合检索 / RAG 召回 / 清理
  Embedding 服务      — 初始化 / 向量生成 / 维度检测

用法:
  python verify_databases.py            # 运行全部测试
  python verify_databases.py --pg       # 仅 PostgreSQL
  python verify_databases.py --redis    # 仅 Redis
  python verify_databases.py --qdrant   # 仅 Qdrant + Embedding
  python verify_databases.py --rag      # 仅 RAG 端到端测试
  python verify_databases.py --clean    # 清理测试数据后退出

前置条件:
  1. docker-compose up -d (启动 PostgreSQL + Redis + Qdrant)
  2. pip install -r requirements.txt
============================================================
"""

import asyncio
import sys
import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,  # 抑制项目 logger 噪音
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── 终端颜色 ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

results: dict[str, bool] = {}

def header(title: str):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

def step(msg: str):
    print(f"\n  {CYAN}→{RESET} {msg}")

def ok(msg: str = ""):
    print(f"    {PASS} {msg}")

def fail(msg: str = ""):
    print(f"    {FAIL} {msg}")

def warn(msg: str = ""):
    print(f"    {WARN} {msg}")


# ════════════════════════════════════════════════════════════
#  SECTION 1: PostgreSQL 验证
# ════════════════════════════════════════════════════════════

async def verify_postgresql():
    """验证 PostgreSQL：连接 → 表结构 → CRUD → 事务"""
    header("SECTION 1: PostgreSQL 数据库验证")
    import asyncpg

    from core.config import get_config
    cfg = get_config().database

    step(f"连接参数: host={cfg.host}, port={cfg.port}, db={cfg.database}, user={cfg.username}")
    print(f"    password={'***' if cfg.password else '(empty)'}")

    # ── 1.1 连接测试 ──
    step("1.1 连接测试")
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(
                host=cfg.host, port=cfg.port,
                database=cfg.database, user=cfg.username, password=cfg.password,
            ),
            timeout=10.0,
        )
        version = await conn.fetchval("SELECT version()")
        ok(f"已连接 — {str(version).split(',')[0]}")
        results["pg_connection"] = True
    except asyncio.TimeoutError:
        fail("连接超时 (10s) — 检查 docker-compose up -d 是否启动")
        results["pg_connection"] = False
        return
    except Exception as e:
        fail(f"连接失败: {e}")
        results["pg_connection"] = False
        return

    # ── 1.2 表结构检查 ──
    step("1.2 表结构检查 (预期 7 张表)")
    expected_tables = [
        "users", "papers", "user_paper_relations",
        "analysis_reports", "reference_cache",
        "agent_execution_logs", "agent_tool_calls", "workflow_executions",
    ]
    rows = await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
    )
    existing = {r["table_name"] for r in rows}
    all_ok = True
    for t in expected_tables:
        if t in existing:
            ok(f"表: {t}")
        else:
            fail(f"表: {t} — 缺失!")
            all_ok = False

    if len(existing - set(expected_tables)) > 0:
        warn(f"额外表: {existing - set(expected_tables)}")

    results["pg_tables"] = all_ok

    # ── 1.3 单行 CRUD 测试 ──
    step("1.3 CRUD 操作测试 (users 表)")
    test_id = str(uuid.uuid4())
    test_email = f"verify_test_{datetime.now().strftime('%Y%m%d%H%M%S')}@test.local"

    # CREATE
    try:
        user = await conn.fetchrow(
            "INSERT INTO users (id, email, profile) VALUES ($1, $2, $3) RETURNING id, email, created_at",
            test_id, test_email, json.dumps({"test": True, "source": "verify_databases"}),
        )
        ok(f"INSERT — id={str(user['id'])[:8]}..., email={user['email']}")
        results["pg_insert"] = True
    except Exception as e:
        fail(f"INSERT 失败: {e}")
        results["pg_insert"] = False
        await conn.close()
        return

    # READ
    try:
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", test_id)
        assert row is not None, "查询返回空"
        assert row["email"] == test_email, f"email 不匹配: {row['email']} != {test_email}"
        ok(f"SELECT — email 匹配, profile={row['profile']}")
        results["pg_select"] = True
    except Exception as e:
        fail(f"SELECT 失败: {e}")
        results["pg_select"] = False

    # UPDATE
    try:
        updated = await conn.fetchrow(
            "UPDATE users SET profile = $1 WHERE id = $2 RETURNING profile",
            json.dumps({"test": True, "updated": True}),
            test_id,
        )
        ok(f"UPDATE — profile={updated['profile']}")
        results["pg_update"] = True
    except Exception as e:
        fail(f"UPDATE 失败: {e}")
        results["pg_update"] = False

    # DELETE (cleanup)
    try:
        await conn.execute("DELETE FROM users WHERE id = $1", test_id)
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", test_id)
        assert row is None, "删除后仍能查到记录"
        ok("DELETE — 测试记录已清理")
        results["pg_delete"] = True
    except Exception as e:
        fail(f"DELETE 失败: {e}")
        results["pg_delete"] = False

    # ── 1.4 关联表写入测试 (papers + analysis_reports) ──
    step("1.4 关联写入测试 (papers → analysis_reports)")
    try:
        paper_id = str(uuid.uuid4())
        paper = await conn.fetchrow(
            """INSERT INTO papers (id, title, authors, abstract, doi)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (doi) DO UPDATE SET title = EXCLUDED.title
               RETURNING id, title""",
            paper_id,
            "Test Paper: Database Verification",
            ["Test Author"],
            "This is a test abstract for database verification.",
            f"10.9999/verify_test_{datetime.now().strftime('%H%M%S')}",
        )
        ok(f"INSERT papers — {paper['title'][:50]}")
        results["pg_papers_insert"] = True

        # 写入 analysis_reports
        report_id = str(uuid.uuid4())
        report = await conn.fetchrow(
            """INSERT INTO analysis_reports (id, paper_id, summary, innovation_point)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            report_id, paper_id,
            "Test summary: this paper is about database verification.",
            "The innovation is comprehensive automated testing.",
        )
        ok(f"INSERT analysis_reports — id={str(report['id'])[:8]}...")
        results["pg_analysis_insert"] = True

        # JOIN 查询验证
        joined = await conn.fetchrow(
            """SELECT p.title, a.summary
               FROM papers p
               JOIN analysis_reports a ON p.id = a.paper_id
               WHERE p.id = $1""",
            paper_id,
        )
        assert joined is not None, "JOIN 查询失败"
        ok(f"JOIN 查询 — paper={joined['title'][:40]}, summary={joined['summary'][:40]}")
        results["pg_join"] = True

        # ── 1.5 reference_cache UPSERT 测试 ──
        step("1.5 reference_cache UPSERT 测试")
        test_doi = f"10.9999/ref_test_{datetime.now().strftime('%H%M%S')}"
        await conn.execute(
            """INSERT INTO reference_cache (doi, bibtex_std)
               VALUES ($1, $2)
               ON CONFLICT (doi) DO UPDATE SET bibtex_std = EXCLUDED.bibtex_std""",
            test_doi, "@article{test, author={Test}, title={Test}}",
        )
        # 第二次写入同样的 DOI 应触发 UPDATE
        await conn.execute(
            """INSERT INTO reference_cache (doi, bibtex_std, is_verified)
               VALUES ($1, $2, $3)
               ON CONFLICT (doi) DO UPDATE SET bibtex_std = EXCLUDED.bibtex_std, is_verified = EXCLUDED.is_verified""",
            test_doi, "@article{test, author={Test}, title={Upserted}}", True,
        )
        cached = await conn.fetchrow("SELECT * FROM reference_cache WHERE doi = $1", test_doi)
        assert cached["is_verified"] is True, "UPSERT 后 is_verified 应为 True"
        ok(f"UPSERT — doi={test_doi}, verified={cached['is_verified']}")
        results["pg_upsert"] = True

        # ── 1.6 agent_execution_logs 写入测试 ──
        step("1.6 agent_execution_logs 写入测试")
        exec_id = str(uuid.uuid4())
        await conn.execute(
            """INSERT INTO agent_execution_logs (id, agent_name, task_type, status, tools_called)
               VALUES ($1, $2, $3, $4, $5)""",
            exec_id, "verify_script", "database_test", "completed",
            json.dumps(["tool_a", "tool_b"]),
        )
        log = await conn.fetchrow("SELECT * FROM agent_execution_logs WHERE id = $1", exec_id)
        ok(f"INSERT agent_execution_logs — agent={log['agent_name']}, status={log['status']}")
        results["pg_agent_log"] = True

        # ── 1.7 清理测试数据 ──
        step("1.7 清理测试数据")
        await conn.execute("DELETE FROM agent_tool_calls WHERE execution_id = $1", exec_id)
        await conn.execute("DELETE FROM agent_execution_logs WHERE id = $1", exec_id)
        await conn.execute("DELETE FROM analysis_reports WHERE id = $1", report_id)
        await conn.execute("DELETE FROM reference_cache WHERE doi = $1", test_doi)
        await conn.execute("DELETE FROM papers WHERE id = $1", paper_id)
        ok("所有测试数据已清理")

    except Exception as e:
        fail(f"关联写入测试异常: {e}")
        import traceback
        traceback.print_exc()

    # ── 1.8 表行数统计 ──
    step("1.8 表行数统计")
    counts = await conn.fetch("""
        SELECT
            (SELECT COUNT(*) FROM users) AS users,
            (SELECT COUNT(*) FROM papers) AS papers,
            (SELECT COUNT(*) FROM analysis_reports) AS analysis_reports,
            (SELECT COUNT(*) FROM reference_cache) AS reference_cache,
            (SELECT COUNT(*) FROM agent_execution_logs) AS agent_execution_logs,
            (SELECT COUNT(*) FROM workflow_executions) AS workflow_executions
    """)
    c = counts[0]
    print(f"    users={c['users']}  papers={c['papers']}  analyses={c['analysis_reports']}  "
          f"ref_cache={c['reference_cache']}  agent_logs={c['agent_execution_logs']}  "
          f"workflows={c['workflow_executions']}")

    await conn.close()
    ok("PostgreSQL 全部验证完成")


# ════════════════════════════════════════════════════════════
#  SECTION 2: Redis 验证
# ════════════════════════════════════════════════════════════

async def verify_redis():
    """验证 Redis：连接 → 任务队列 → 缓存 → Agent 状态 → Pub/Sub"""
    header("SECTION 2: Redis 验证")

    from core.redis_manager import redis_manager, HAS_REDIS

    if not HAS_REDIS:
        fail("redis 包未安装 — pip install redis")
        results["redis_all"] = False
        return

    # ── 2.1 连接测试 ──
    step("2.1 连接测试")
    try:
        await redis_manager.initialize()
        await redis_manager.redis.ping()
        ok("Redis PING 成功")
        results["redis_connection"] = True
    except Exception as e:
        fail(f"Redis 连接失败: {e}")
        results["redis_connection"] = False
        return

    r = redis_manager.redis

    # ── 2.2 任务队列 (Sorted Set) ──
    step("2.2 任务队列 (Sorted Set)")
    test_queue = "__verify_test_queue__"
    try:
        await r.zadd(test_queue, {"task_high": -10, "task_low": -1})
        popped = await r.zpopmin(test_queue)
        assert popped is not None and len(popped) > 0, "zpopmin 返回空"
        task_id, score = popped[0]
        assert task_id == "task_high", f"高优先级任务应先弹出: got {task_id}"
        ok(f"zadd/zpopmin — 高优先级先出 (score={score})")
        # 清理
        await r.zpopmin(test_queue)  # 弹出 task_low
        await r.delete(test_queue)
        results["redis_queue"] = True
    except Exception as e:
        fail(f"任务队列失败: {e}")
        results["redis_queue"] = False

    # ── 2.3 活跃任务 (Hash) ──
    step("2.3 活跃任务 (Hash)")
    try:
        test_task = {"id": "test-001", "type": "verify", "status": "active", "ts": str(datetime.now())}
        await redis_manager.set_active_task("test-001", test_task)
        fetched = await redis_manager.get_active_task("test-001")
        assert fetched is not None, "get_active_task 返回 None"
        assert fetched["id"] == "test-001", f"id 不匹配: {fetched}"
        ok(f"hset/hget — task_id={fetched['id']}, type={fetched['type']}")
        await redis_manager.remove_active_task("test-001")
        gone = await redis_manager.get_active_task("test-001")
        assert gone is None, "hdel 后仍能查到"
        ok("hdel — 删除后查询返回 None")
        results["redis_hash"] = True
    except Exception as e:
        fail(f"活跃任务 Hash 失败: {e}")
        results["redis_hash"] = False

    # ── 2.4 任务历史 (List) ──
    step("2.4 任务历史 (List)")
    try:
        await r.delete("task_history")  # 先清空
        for i in range(3):
            await redis_manager.push_task_history({"task": f"verify-{i}", "result": "ok"})
        history = await redis_manager.get_task_history(limit=10)
        assert len(history) == 3, f"期望 3 条, 得到 {len(history)} 条"
        ok(f"lpush/lrange — 写入 3 条, 读出 {len(history)} 条")
        await r.delete("task_history")
        results["redis_list"] = True
    except Exception as e:
        fail(f"任务历史 List 失败: {e}")
        results["redis_list"] = False

    # ── 2.5 通用缓存 (String + TTL) ──
    step("2.5 通用缓存 (String + TTL)")
    try:
        cache_key = "__verify_cache_test__"
        cache_val = {"data": "hello", "nested": {"count": 42}}
        await redis_manager.cache_set(cache_key, cache_val, ttl=60)
        cached = await redis_manager.cache_get(cache_key)
        assert cached is not None, "cache_get 返回 None"
        assert cached["nested"]["count"] == 42, f"嵌套数据不匹配: {cached}"
        ok(f"SET/GET — key={cache_key}, value={cached}")
        await redis_manager.cache_delete(cache_key)
        gone = await redis_manager.cache_get(cache_key)
        assert gone is None, "DEL 后仍能查到"
        ok("DEL — 删除后查询返回 None")
        results["redis_cache"] = True
    except Exception as e:
        fail(f"缓存失败: {e}")
        results["redis_cache"] = False

    # ── 2.6 Agent 状态 (Hash + TTL) ──
    step("2.6 Agent 状态 (Hash + TTL)")
    try:
        agent_state = {
            "name": "verify_agent",
            "status": "running",
            "last_task": "verify_redis",
            "tools_used": json.dumps(["redis_check"]),
        }
        await redis_manager.set_agent_state("verify_agent", agent_state, ttl=60)
        fetched_state = await redis_manager.get_agent_state("verify_agent")
        assert fetched_state is not None, "get_agent_state 返回 None"
        assert fetched_state["status"] == "running", f"status 不匹配: {fetched_state}"
        ok(f"hset/hgetall — agent={fetched_state['name']}, status={fetched_state['status']}")
        # 检查 TTL
        ttl = await r.ttl("agent_state:verify_agent")
        assert ttl > 0, f"TTL 应为正数: {ttl}"
        ok(f"TTL — {ttl}s remaining")
        await r.delete("agent_state:verify_agent")
        results["redis_agent_state"] = True
    except Exception as e:
        fail(f"Agent 状态失败: {e}")
        results["redis_agent_state"] = False

    # ── 2.7 Pub/Sub ──
    step("2.7 Pub/Sub (发布)")
    try:
        await redis_manager.publish("__verify_channel__", {
            "event": "test_message",
            "timestamp": datetime.now().isoformat(),
        })
        ok("PUBLISH — 消息已发送到 __verify_channel__")
        results["redis_pubsub"] = True
    except Exception as e:
        fail(f"Pub/Sub 失败: {e}")
        results["redis_pubsub"] = False

    print(f"\n  {INFO} Redis 全部验证完成")


# ════════════════════════════════════════════════════════════
#  SECTION 3: Qdrant 向量数据库 + Embedding 验证
# ════════════════════════════════════════════════════════════

async def verify_qdrant_and_embedding():
    """验证 Qdrant + Embedding：连接 → 向量写入 → 混合检索"""
    header("SECTION 3: Qdrant 向量数据库 + Embedding 服务验证")

    from core.config import get_config
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    cfg = get_config().vector_db

    # ── 3.1 Qdrant 连接测试 ──
    step(f"3.1 Qdrant 连接测试 ({cfg.get_qdrant_url()})")
    try:
        client = QdrantClient(
            url=cfg.get_qdrant_url(),
            api_key=cfg.api_key,
            prefer_grpc=False,
            https=cfg.get_qdrant_url().startswith("https://"),
            check_compatibility=False
        )
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections] if collections.collections else []
        ok(f"已连接 — 现有 collections: {col_names if col_names else '(空)'}")
        results["qdrant_connection"] = True
    except Exception as e:
        fail(f"Qdrant 连接失败: {e}")
        results["qdrant_connection"] = False
        return

    # ── 3.2 Embedding 服务初始化 ──
    step("3.2 Embedding 服务初始化")
    try:
        from utils.embedding import get_embedding_service
        embedding_service = get_embedding_service()

        if not embedding_service.embeddings:
            await embedding_service.initialize()

        ok(f"Embedding 已就绪 — model={embedding_service.embedding_model}")

        # 获取向量维度
        test_embed = await embedding_service.embeddings.aembed_query("test dimension check")
        dim = len(test_embed)
        ok(f"向量维度: {dim}")
        results["embedding_init"] = True
    except Exception as e:
        fail(f"Embedding 初始化失败: {e}")
        results["embedding_init"] = False
        return

    # ── 3.3 创建测试 Collection ──
    step("3.3 创建测试 Collection")
    test_collection = "__verify_test_collection__"
    try:
        # 删除可能残留的测试 collection
        try:
            client.delete_collection(test_collection)
        except Exception:
            pass

        from qdrant_client.http.models import Distance, VectorParams
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        ok(f"Collection '{test_collection}' 已创建 (dim={dim}, distance=COSINE)")
        results["qdrant_create_collection"] = True
    except Exception as e:
        fail(f"创建 Collection 失败: {e}")
        results["qdrant_create_collection"] = False
        return

    # ── 3.4 向量写入测试 ──
    step("3.4 向量写入测试 (5 条论文摘要)")
    test_papers = [
        {"title": "Deep Learning for NLP",     "abstract": "Transformer architectures have revolutionized natural language processing with attention mechanisms and self-supervised pre-training."},
        {"title": "Graph Neural Networks",      "abstract": "Graph neural networks extend deep learning to graph-structured data using message passing between nodes."},
        {"title": "Reinforcement Learning",     "abstract": "Reinforcement learning enables agents to learn optimal policies through interaction with environments and reward signals."},
        {"title": "Computer Vision Advances",   "abstract": "Convolutional neural networks and vision transformers achieve state-of-the-art results in image classification and object detection."},
        {"title": "Database Systems",           "abstract": "Modern database systems support ACID transactions, distributed consistency, and high availability through replication protocols."},
    ]

    from qdrant_client.http.models import PointStruct
    points = []
    for i, paper in enumerate(test_papers):
        vec = await embedding_service.embeddings.aembed_query(paper["abstract"])
        points.append(PointStruct(
            id=i + 1,
            vector=vec,
            payload={
                "paper_id": f"test_paper_{i+1}",
                "title": paper["title"],
                "abstract": paper["abstract"],
                "collection_type": "test",
                "source": "verify_databases",
            },
        ))

    client.upsert(collection_name=test_collection, points=points)
    info = client.get_collection(test_collection)
    ok(f"已写入 {info.points_count} 个向量 (预期 5)")
    results["qdrant_upsert"] = info.points_count == 5

    # ── 3.5 纯向量检索 (Semantic Search) ──
    step("3.5 纯向量检索 — 查询: 'attention mechanism in neural networks'")
    query_vec = await embedding_service.embeddings.aembed_query(
        "attention mechanism in neural networks"
    )
    search_response = client.query_points(
        collection_name=test_collection,
        query=query_vec,
        limit=3,
    )
    search_result = search_response.points if search_response else []
    print(f"    Top-3 语义匹配结果:")
    for i, hit in enumerate(search_result):
        title = hit.payload.get("title", "?")
        print(f"      #{i+1} score={hit.score:.4f} — {title}")
    assert len(search_result) >= 1, "向量检索返回空"
    # 语义上，"Deep Learning for NLP" (attention/transformer) 应该最接近
    top_title = search_result[0].payload.get("title", "")
    ok(f"向量检索成功 — Top-1: {top_title}")
    results["qdrant_vector_search"] = True

    # ── 3.6 混合检索 (Hybrid: vector + keyword) ──
    step("3.6 混合检索 — 查询: 'graph data neural'")
    query_text = "graph data neural"
    query_vec2 = await embedding_service.embeddings.aembed_query(query_text)

    # 向量检索
    vec_response = client.query_points(
        collection_name=test_collection,
        query=query_vec2,
        limit=5,
    )
    vec_hits = vec_response.points if vec_response else []

    # 关键词打分 (Jaccard — 与项目 vector_store.py 的 keyword_score 保持一致)
    def keyword_score(query: str, text: str) -> float:
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        if not q_words or not t_words:
            return 0.0
        return len(q_words & t_words) / len(q_words | t_words)

    # 混合打分 (vector weight 0.7, keyword weight 0.3)
    combined = []
    for hit in vec_hits:
        abstract = hit.payload.get("abstract", "")
        title = hit.payload.get("title", "")
        vec_score = hit.score
        kw_score = keyword_score(query_text, f"{title} {abstract}")
        # 归一化向量分数到 [0, 1] (cosine 范围本是 [-1, 1])
        vec_norm = (vec_score + 1) / 2
        hybrid = 0.7 * vec_norm + 0.3 * kw_score
        combined.append((hybrid, vec_score, kw_score, title, abstract[:60]))

    combined.sort(key=lambda x: x[0], reverse=True)

    print(f"    {'Title':<30} {'Hybrid':>7} {'Vector':>7} {'Keyword':>7}")
    print(f"    {'─'*30} {'─'*7} {'─'*7} {'─'*7}")
    for hy, vs, kw, title, _ in combined[:3]:
        print(f"    {title:<30} {hy:>7.4f} {vs:>7.4f} {kw:>7.4f}")

    # "Graph Neural Networks" 应该在关键词和语义两个维度上都排第一
    top_hybrid_title = combined[0][3]
    ok(f"混合检索成功 — Top-1 (hybrid): {top_hybrid_title}")
    assert "Graph" in top_hybrid_title, f"混合检索未找到最相关结果: {top_hybrid_title}"
    results["qdrant_hybrid_search"] = True

    # ── 3.7 过滤检索 (带 filter) ──
    step("3.7 过滤检索 — 按 source=verify_databases 过滤")
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    filter_response = client.query_points(
        collection_name=test_collection,
        query=query_vec2,
        query_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="verify_databases"))]
        ),
        limit=3,
    )
    filter_result = filter_response.points if filter_response else []
    assert len(filter_result) >= 1, "过滤检索返回空"
    ok(f"过滤检索 — 返回 {len(filter_result)} 条, 全部匹配 source=verify_databases")
    results["qdrant_filter_search"] = True

    # ── 3.8 清理测试 Collection ──
    step("3.8 清理测试数据")
    try:
        client.delete_collection(test_collection)
        ok(f"测试 Collection '{test_collection}' 已删除")
    except Exception as e:
        warn(f"删除 Collection 失败: {e}")

    client.close()
    print(f"\n  {INFO} Qdrant + Embedding 全部验证完成")


# ════════════════════════════════════════════════════════════
#  SECTION 4: RAG 端到端验证
# ════════════════════════════════════════════════════════════

async def verify_rag_e2e():
    """端到端 RAG 验证：写入 PG → 生成 Embedding → 写入 Qdrant → 检索召回 → 验证结果"""
    header("SECTION 4: RAG 端到端验证")
    import asyncpg

    from core.config import get_config
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct

    cfg_pg = get_config().database
    cfg_qd = get_config().vector_db

    # ── 4.1 准备连接 ──
    step("4.1 准备数据库连接")
    try:
        pg_conn = await asyncpg.connect(
            host=cfg_pg.host, port=cfg_pg.port,
            database=cfg_pg.database, user=cfg_pg.username, password=cfg_pg.password,
        )
        qd_client = QdrantClient(
            url=cfg_qd.get_qdrant_url(),
            api_key=cfg_qd.api_key,
            prefer_grpc=False,
            https=cfg_qd.get_qdrant_url().startswith("https://"),
            check_compatibility=False
        )

        from utils.embedding import get_embedding_service
        emb_svc = get_embedding_service()
        if not emb_svc.embeddings:
            await emb_svc.initialize()
        dim = len(await emb_svc.embeddings.aembed_query("dimension check"))
        ok(f"PG + Qdrant + Embedding 已就绪 (dim={dim})")
    except Exception as e:
        fail(f"连接准备失败: {e}")
        results["rag_e2e"] = False
        return

    # ── 4.2 写入论文到 PostgreSQL ──
    step("4.2 写入测试论文到 PostgreSQL")
    rag_paper_id = str(uuid.uuid4())
    paper_title = "RAG Verification: Efficient Vector Search for Scientific Literature"
    paper_abstract = (
        "This paper presents a novel approach to scientific literature retrieval using "
        "hybrid vector-keyword search combined with large language models. The method "
        "achieves 95% recall on benchmark datasets while maintaining sub-second latency. "
        "Key innovations include adaptive embedding fusion and multi-stage reranking."
    )
    try:
        paper = await pg_conn.fetchrow(
            """INSERT INTO papers (id, title, authors, abstract, doi)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (doi) DO UPDATE SET title = EXCLUDED.title
               RETURNING id, title, doi""",
            rag_paper_id, paper_title,
            ["RAG Tester", "Verification Bot"],
            paper_abstract,
            f"10.9999/rag_verify_{datetime.now().strftime('%H%M%S')}",
        )
        ok(f"PG 写入 — {paper['title'][:55]}")
    except Exception as e:
        fail(f"PG 写入失败: {e}")
        results["rag_e2e"] = False
        return

    # ── 4.3 生成 Embedding 并写入 Qdrant ──
    step("4.3 生成 Embedding → 写入 Qdrant")
    rag_collection = "__verify_rag_test__"
    try:
        # 清理并创建 collection
        try:
            qd_client.delete_collection(rag_collection)
        except Exception:
            pass
        qd_client.create_collection(
            collection_name=rag_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # 多段落嵌入 (模拟真实场景: 分 chunk 存入)
        chunks = [
            "hybrid vector-keyword search combined with large language models",
            "achieves 95% recall on benchmark datasets while maintaining sub-second latency",
            "adaptive embedding fusion and multi-stage reranking",
            "scientific literature retrieval using hybrid search methods",
        ]

        points = []
        for i, chunk in enumerate(chunks):
            vec = await emb_svc.embeddings.aembed_query(chunk)
            points.append(PointStruct(
                id=i + 1,
                vector=vec,
                payload={
                    "paper_id": rag_paper_id,
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "source": "rag_verify",
                },
            ))

        qd_client.upsert(collection_name=rag_collection, points=points)
        info = qd_client.get_collection(rag_collection)
        ok(f"Qdrant 写入 — {info.points_count} 个向量 (4 chunks)")
    except Exception as e:
        fail(f"Qdrant 写入失败: {e}")
        results["rag_e2e"] = False
        await pg_conn.close()
        return

    # ── 4.4 RAG 查询召回 ──
    step("4.4 RAG 查询 — 'how to improve literature search recall'")
    query = "how to improve literature search recall"
    query_vec = await emb_svc.embeddings.aembed_query(query)

    # 混合检索
    rag_response = qd_client.query_points(
        collection_name=rag_collection,
        query=query_vec,
        limit=3,
    )
    vec_hits = rag_response.points if rag_response else []

    def keyword_score(query: str, text: str) -> float:
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        if not q_words or not t_words:
            return 0.0
        return len(q_words & t_words) / len(q_words | t_words)

    print(f"    查询: '{query}'")
    print(f"    {'Chunk':<60} {'Vector':>7} {'Keyword':>7} {'Hybrid':>7}")
    print(f"    {'─'*60} {'─'*7} {'─'*7} {'─'*7}")
    for hit in vec_hits:
        chunk = hit.payload.get("chunk_text", "")[:55]
        vs = hit.score
        ks = keyword_score(query, chunk)
        vs_norm = (vs + 1) / 2
        hy = 0.7 * vs_norm + 0.3 * ks
        print(f"    {chunk:<60} {vs:>7.4f} {ks:>7.4f} {hy:>7.4f}")

    # 验证: 应该至少有一条结果
    assert len(vec_hits) >= 1, "RAG 检索返回空结果"
    ok(f"RAG 检索 — 返回 {len(vec_hits)} 条结果")

    # ── 4.5 验证召回质量 ──
    step("4.5 验证召回质量")
    recalled_texts = [h.payload.get("chunk_text", "") for h in vec_hits]
    all_text = " ".join(recalled_texts).lower()
    expected_terms = ["recall", "search", "hybrid", "retrieval"]
    found_terms = [t for t in expected_terms if t in all_text]
    print(f"    期望术语: {expected_terms}")
    print(f"    召回覆盖: {found_terms} ({len(found_terms)}/{len(expected_terms)})")
    if len(found_terms) >= 2:
        ok(f"召回质量合格 — 命中 {len(found_terms)}/{len(expected_terms)} 个关键术语")
        results["rag_quality"] = True
    else:
        warn(f"召回覆盖不足 — 仅命中 {len(found_terms)}/{len(expected_terms)}")
        results["rag_quality"] = False

    # ── 4.6 写入 analysis_reports (RAG 结果持久化) ──
    step("4.6 RAG 结果持久化 (analysis_reports)")
    report_id = str(uuid.uuid4())
    try:
        await pg_conn.fetchrow(
            """INSERT INTO analysis_reports (id, paper_id, summary, vector_ids)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            report_id, rag_paper_id,
            f"RAG 验证报告: 对查询 '{query}' 召回了 {len(vec_hits)} 个相关文本块。"
            f"召回文本覆盖了 {len(found_terms)}/{len(expected_terms)} 个关键术语。",
            json.dumps({
                "collection": rag_collection,
                "query": query,
                "num_results": len(vec_hits),
                "hit_ids": [h.id for h in vec_hits],
            }),
        )
        ok(f"analysis_reports 写入 — report_id={report_id[:8]}...")

        # 验证可回读
        saved = await pg_conn.fetchrow(
            "SELECT * FROM analysis_reports WHERE id = $1", report_id
        )
        assert saved is not None, "回读失败"
        ok("回读验证 — 报告可查询")
    except Exception as e:
        fail(f"持久化失败: {e}")

    # ── 4.7 清理 ──
    step("4.7 清理 RAG 测试数据")
    try:
        await pg_conn.execute("DELETE FROM analysis_reports WHERE id = $1", report_id)
        await pg_conn.execute("DELETE FROM papers WHERE id = $1", rag_paper_id)
        qd_client.delete_collection(rag_collection)
        ok("RAG 测试数据已全部清理")
    except Exception as e:
        warn(f"清理异常: {e}")

    await pg_conn.close()
    qd_client.close()
    results["rag_e2e"] = True
    print(f"\n  {INFO} RAG 端到端验证完成")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

async def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  InnoCore AI — 数据库与 RAG 功能验证套件{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version}")

    args = set(sys.argv[1:])
    run_all = not args or "--clean" in args

    if "--clean" in args:
        print(f"\n  {YELLOW}[CLEAN] 清理模式{RESET}")
        await cleanup_all()
        return

    if run_all or "--pg" in args:
        await verify_postgresql()

    if run_all or "--redis" in args:
        await verify_redis()

    if run_all or "--qdrant" in args:
        await verify_qdrant_and_embedding()

    if run_all or "--rag" in args:
        await verify_rag_e2e()

    # ── 总结 ──
    header("验证结果总结")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n  总计: {total} 项 | {GREEN}通过: {passed}{RESET} | {RED}失败: {failed}{RESET}")
    print()
    for name, ok_flag in results.items():
        icon = PASS if ok_flag else FAIL
        print(f"    {icon} {name}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}[SUCCESS] 全部通过！数据库和 RAG 功能正常运行。{RESET}")
    else:
        print(f"\n  {RED}{BOLD}[WARNING] {failed} 项未通过，请检查对应服务是否启动。{RESET}")
        print(f"    启动命令: docker-compose up -d")
        print(f"    检查状态: docker-compose ps")


async def cleanup_all():
    """清理所有可能的测试残留"""
    import asyncpg
    from core.config import get_config
    from qdrant_client import QdrantClient

    print("  清理 PostgreSQL 测试数据...")
    try:
        cfg_pg = get_config().database
        conn = await asyncpg.connect(
            host=cfg_pg.host, port=cfg_pg.port,
            database=cfg_pg.database, user=cfg_pg.username, password=cfg_pg.password,
        )
        # 清理所有包含 verify 标记的数据
        await conn.execute("DELETE FROM analysis_reports WHERE summary LIKE '%验证%' OR summary LIKE '%verify%'")
        await conn.execute("DELETE FROM user_paper_relations WHERE tags @> ARRAY['verify_test']")
        await conn.execute("DELETE FROM reference_cache WHERE doi LIKE '10.9999/verify_%' OR doi LIKE '10.9999/ref_test_%' OR doi LIKE '10.9999/rag_verify_%'")
        await conn.execute("DELETE FROM agent_execution_logs WHERE agent_name = 'verify_script'")
        await conn.execute("DELETE FROM papers WHERE doi LIKE '10.9999/verify_%' OR doi LIKE '10.9999/rag_verify_%'")
        await conn.execute("DELETE FROM users WHERE email LIKE '%@test.local'")
        await conn.close()
        ok("PostgreSQL 清理完成")
    except Exception as e:
        warn(f"PostgreSQL 清理: {e}")

    print("  清理 Qdrant 测试 Collection...")
    try:
        cfg_qd = get_config().vector_db
        client = QdrantClient(
            url=cfg_qd.get_qdrant_url(),
            api_key=cfg_qd.api_key,
            prefer_grpc=False,
            https=cfg_qd.get_qdrant_url().startswith("https://"),
            check_compatibility=False
        )
        for col_name in ["__verify_test_collection__", "__verify_rag_test__"]:
            try:
                client.delete_collection(col_name)
                ok(f"删除 {col_name}")
            except Exception:
                pass
        client.close()
    except Exception as e:
        warn(f"Qdrant 清理: {e}")

    print("  清理 Redis 测试 Key...")
    try:
        from core.redis_manager import redis_manager, HAS_REDIS
        if HAS_REDIS:
            await redis_manager.initialize()
            r = redis_manager.redis
            if r:
                await r.delete(
                    "task_history", "active_tasks",
                    "agent_state:verify_agent",
                    "__verify_cache_test__",
                )
                ok("Redis 清理完成")
    except Exception as e:
        warn(f"Redis 清理: {e}")

    print(f"\n  {GREEN}{BOLD}[CLEAN] 清理完成{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
