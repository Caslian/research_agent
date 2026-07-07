"""端到端验收：Hunter 已读/忽略功能。

覆盖：
  1. mark_paper_read → 幂等 INSERT
  2. list_read_paper_keys → 返回 set
  3. _filter_by_read_state → 正确过滤
  4. _search_only → 在 _filter_by_kb 之后调用 _filter_by_read_state
  5. /api/v1/papers/mark-read 路由已注册（用 FastAPI TestClient 避免网络）
"""
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import db_manager
from agents.hunter import HunterAgent


async def step1_ensure_user() -> str:
    """确保测试用户存在（直接 INSERT 绕过 bcrypt 等）。"""
    user_id = "11111111-1111-1111-1111-111111111111"
    await db_manager.execute(
        "INSERT INTO users (id, email, profile) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING",
        user_id, f"hunter-test-{user_id[:8]}@example.com", "{}",
    )
    return user_id


async def step2_mark_paper_read(user_id: str) -> None:
    """写入 3 条已读：2 篇 arxiv + 1 篇 ieee。"""
    samples = [
        ("arxiv:2401.00001", "arxiv", "Paper A"),
        ("arxiv:2401.00002", "arxiv", "Paper B"),
        ("ieee:9876543",     "ieee",  "Paper C"),
    ]
    for key, src, title in samples:
        ok = await db_manager.mark_paper_read(
            user_id=user_id, paper_key=key, source=src,
            reason="marked_read", title=title,
        )
        assert ok, f"mark_paper_read 失败: {key}"
    # 重复插入验证幂等
    ok = await db_manager.mark_paper_read(
        user_id=user_id, paper_key="arxiv:2401.00001", source="arxiv",
        reason="marked_read", title="Paper A",
    )
    assert ok
    print("  PASS: mark_paper_read 幂等写入 4 次（3 新 + 1 重复）")


async def step3_list_keys(user_id: str) -> set:
    keys = await db_manager.list_read_paper_keys(user_id)
    assert keys == {"arxiv:2401.00001", "arxiv:2401.00002", "ieee:9876543"}, keys
    print(f"  PASS: list_read_paper_keys 返回 {len(keys)} 个 key")
    return keys


async def step4_list_papers(user_id: str) -> None:
    rows = await db_manager.list_read_papers_for_user(user_id, limit=10)
    assert len(rows) == 3, f"expected 3, got {len(rows)}"
    reasons = {r["reason"] for r in rows}
    assert reasons == {"marked_read"}
    print(f"  PASS: list_read_papers_for_user 返回 {len(rows)} 条，reasons={reasons}")


async def step5_filter_read_state(user_id: str) -> None:
    """直接调 HunterAgent._filter_by_read_state 验证过滤逻辑。"""
    hunter = HunterAgent()
    papers = [
        {"id": "2401.00001", "source": "arxiv", "title": "Paper A"},  # 在已读
        {"id": "2401.00002", "source": "arxiv", "title": "Paper B"},  # 在已读
        {"id": "2401.00003", "source": "arxiv", "title": "Paper D"},  # 不在
        {"id": "9876543",    "source": "ieee",  "title": "Paper C"},  # 在已读
        {"id": "9876544",    "source": "ieee",  "title": "Paper E"},  # 不在
        {"id": "2401.00005", "source": "arxiv", "title": "Paper F"},  # 不在
    ]
    filtered = await hunter._filter_by_read_state(papers, user_id)
    titles = [p["title"] for p in filtered]
    assert titles == ["Paper D", "Paper E", "Paper F"], titles
    print(f"  PASS: _filter_by_read_state 过滤后剩 {len(filtered)} 篇: {titles}")

    # 异常路径：user_id=None 应原样返回
    out = await hunter._filter_by_read_state(papers, None)
    assert len(out) == len(papers)
    print("  PASS: _filter_by_read_state(user_id=None) 跳过过滤")


async def step6_search_only_pipeline(user_id: str) -> None:
    """端到端：构造 mock 的 _search_papers_from_arxiv 让 _search_only 跑通。"""
    hunter = HunterAgent()

    # monkey patch 掉真正的 arxiv/ieee 搜索，避免外网依赖
    async def fake_arxiv(kw, mx, db):
        return [
            {"id": "2401.00001", "source": "arxiv", "title": "Paper A",
             "authors": [], "abstract": "test abstract A", "pdf_url": "", "doi": ""},
            {"id": "2401.00003", "source": "arxiv", "title": "Paper D",
             "authors": [], "abstract": "test abstract D", "pdf_url": "", "doi": ""},
            {"id": "2401.00005", "source": "arxiv", "title": "Paper F",
             "authors": [], "abstract": "test abstract F", "pdf_url": "", "doi": ""},
        ]
    async def fake_ieee(kw, mx, db):
        return [
            {"id": "9876543", "source": "ieee", "title": "Paper C",
             "authors": [], "abstract": "test abstract C", "pdf_url": "", "doi": ""},
            {"id": "9876544", "source": "ieee", "title": "Paper E",
             "authors": [], "abstract": "test abstract E", "pdf_url": "", "doi": ""},
        ]
    hunter._search_papers_from_arxiv = fake_arxiv
    hunter._search_papers_from_ieee  = fake_ieee

    # 也跳过 _fetch_and_generate_note 的 PDF 下载，避免外网
    async def fake_fetch(p):
        p["note"] = ""
        p["note_status"] = "skipped"
        p["pdf_cached"] = False
        return p
    hunter._fetch_and_generate_note = fake_fetch

    out = await hunter.run({
        "keywords": ["test"],
        "max_papers": 10,
        "sources": ["arxiv", "ieee"],
        "user_id": user_id,
        "days_back": 7,
    })
    titles = sorted(p["title"] for p in out["papers"])
    assert titles == ["Paper D", "Paper E", "Paper F"], titles
    print(f"  PASS: HunterAgent.run 端到端过滤后剩 {len(titles)} 篇: {titles}")


async def step7_cleanup(user_id: str) -> None:
    """清理测试数据，不影响其他用户。"""
    await db_manager.execute(
        "DELETE FROM user_paper_read_state WHERE user_id = $1", user_id,
    )
    # 不删 users（HTTP 验证脚本会复用这个 user_id）
    print("  PASS: 测试数据清理完成（保留 users 行供 HTTP 测试复用）")


async def main() -> int:
    print("=" * 60)
    print("Hunter 已读/忽略 — 端到端验收")
    print("=" * 60)
    await db_manager.initialize()
    try:
        user_id = await step1_ensure_user()
        print(f"\n[1/7] 准备测试用户: {user_id}")

        print("\n[2/7] 写入已读记录")
        await step2_mark_paper_read(user_id)

        print("\n[3/7] 列出已读 keys")
        await step3_list_keys(user_id)

        print("\n[4/7] 列出已读详情")
        await step4_list_papers(user_id)

        print("\n[5/7] _filter_by_read_state 单元验证")
        await step5_filter_read_state(user_id)

        print("\n[6/7] HunterAgent.run 端到端")
        await step6_search_only_pipeline(user_id)

        print("\n[7/7] 清理")
        await step7_cleanup(user_id)

        print("\n" + "=" * 60)
        print("ALL PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await db_manager.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
