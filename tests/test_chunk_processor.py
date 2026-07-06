"""chunk_processor 单元测试 - 离线可跑，无需向量库。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.chunk_processor import (
    ChunkConfig,
    SectionChunk,
    _is_cjk_char,
    classify_section_type,
    estimate_token_len,
    preprocess_text_for_embedding,
    process_paper_to_chunks,
    split_section_to_paragraphs,
    normalize_section_name,
)
from utils.research_paper_parser import PaperMetadata, PaperSection, ResearchPaper


def test_cjk_basic():
    assert _is_cjk_char("中") is True
    assert _is_cjk_char("A") is False
    assert _is_cjk_char(" ") is False


def test_estimate_token_len():
    # 中文 3 字 + 2 个英文单词 → 3 + 2 = 5
    assert estimate_token_len("中文 abc def") == 5
    # 空
    assert estimate_token_len("") == 0
    # 纯英文 → 按词数计
    assert estimate_token_len("hello world foo") == 3


def test_normalize_section_name():
    assert normalize_section_name("INTRODUCTION") == "introduction"
    assert normalize_section_name("参考文献") == "参考文献"
    assert normalize_section_name("") == "unknown"


def test_classify_section_type():
    assert classify_section_type("Abstract") == "abstract"
    assert classify_section_type("INTRODUCTION") == "introduction"
    assert classify_section_type("Method") == "method"
    assert classify_section_type("Experiments and Results") == "experiment"
    assert classify_section_type("References") == "reference"
    assert classify_section_type("Conclusion") == "conclusion"
    assert classify_section_type("3. Approach") == "method"  # model 匹配
    assert classify_section_type("") == "body"


def test_split_section_to_paragraphs_simple():
    sec = PaperSection(
        name="method",
        content="Para one.\n\nPara two has more.\n\nPara three.",
        start_line=0, end_line=10, word_count=12,
    )
    paras = split_section_to_paragraphs(sec)
    assert len(paras) == 3
    assert paras[0]["text"].startswith("Para one")
    assert paras[2]["text"].startswith("Para three")


def test_split_handles_inline_page_footers():
    sec = PaperSection(
        name="method",
        content="Paragraph one body.\n\nPage 5 of 12\n\nPage 1 of 20\n---- 10 ----\n\nParagraph two.",
        start_line=0, end_line=20, word_count=20,
    )
    paras = split_section_to_paragraphs(sec)
    assert len(paras) >= 2
    for p in paras:
        assert "Page 5 of 12" not in p["text"]
        assert "---- 10 ----" not in p["text"]


def test_process_paper_basic():
    intro = PaperSection(
        name="introduction",
        content="Intro paragraph one.\n\nIntro paragraph two with more content.\n\nIntro paragraph three.",
        start_line=0, end_line=20, word_count=40,
    )
    method = PaperSection(
        name="method",
        content="Method paragraph one.\n\nMethod paragraph two.",
        start_line=21, end_line=30, word_count=20,
    )
    paper = ResearchPaper(
        metadata=PaperMetadata(
            title="Test Paper", authors=["Alice", "Bob"],
            abstract="Short abstract.", keywords=["chunking"],
        ),
        sections={"introduction": intro, "method": method},
        full_text="", page_count=1, total_word_count=60,
        key_terms=[], parsing_method="test", parsing_time="",
    )

    cfg = ChunkConfig(chunk_tokens=20, overlap_tokens=5, min_chunk_tokens=5)
    docs = process_paper_to_chunks(paper, paper_id="P1", config=cfg)
    assert len(docs) >= 1
    # 每个 doc 必须带关键 metadata
    for i, d in enumerate(docs):
        meta = d.metadata
        assert meta["paper_id"] == "P1"
        assert meta["paper_title"] == "Test Paper"
        assert meta["section_name"] in ("introduction", "method")
        assert meta["section_type"] in ("introduction", "method")
        assert meta["chunk_index"] == i  # 连续
        assert meta["chunk_token_count"] > 0


def test_abstract_is_single_chunk():
    abstract_sec = PaperSection(
        name="abstract",
        content="This is the abstract.\n\nIt may have multiple lines.",
        start_line=0, end_line=10, word_count=12,
    )
    paper = ResearchPaper(
        metadata=PaperMetadata(title="T", authors=[], abstract="", keywords=[]),
        sections={"abstract": abstract_sec},
        full_text="", page_count=1, total_word_count=12,
        key_terms=[], parsing_method="t", parsing_time="",
    )
    docs = process_paper_to_chunks(paper, paper_id="X", config=ChunkConfig())
    assert len(docs) == 1
    assert docs[0].metadata["section_type"] == "abstract"


def test_chunk_overlap_creates_multiple_chunks_with_shared_tail():
    """长段落应该被切成多 chunk，重叠区前后 chunk 共享。"""
    body = "\n\n".join(
        f"Paragraph {i} has some content that adds tokens. " * 5
        for i in range(8)
    )
    sec = PaperSection(
        name="method",
        content=body, start_line=0, end_line=200, word_count=2000,
    )
    paper = ResearchPaper(
        metadata=PaperMetadata(title="T", authors=[], abstract="", keywords=[]),
        sections={"method": sec},
        full_text="", page_count=2, total_word_count=2000,
        key_terms=[], parsing_method="t", parsing_time="",
    )
    cfg = ChunkConfig(chunk_tokens=120, overlap_tokens=20, min_chunk_tokens=20)
    docs = process_paper_to_chunks(paper, paper_id="OL", config=cfg)
    # 至少 2 个 chunk
    assert len(docs) >= 2
    # 验证 overlap：第二个 chunk 的开头文本应出现在第一个 chunk 的尾部附近
    first_tail = docs[0].page_content[-200:]
    second_head = docs[1].page_content[:200]
    # 共享内容用 set 字重叠判断
    overlap_chars = set(first_tail.split()) & set(second_head.split())
    assert len(overlap_chars) > 0, "overlap should produce shared words"


def test_preprocess_text_for_embedding():
    cleaned = preprocess_text_for_embedding("a  \n\n\n\n  b   \n   c")
    assert "\n\n\n\n" not in cleaned
    assert cleaned.count(" ") <= 1  # 内部多余空格收敛


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"---\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
