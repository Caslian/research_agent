"""
论文分块处理模块 - ChunkProcessor

把 ResearchPaper (utils/research_paper_parser.py 输出) 按照"段内切分 + token 感知 + 尾部重叠"
的策略切分成 LangChain Document 列表，供向量化入库使用。

设计目标:
1. 输入 ResearchPaper 对象（已有 sections: Dict[str, PaperSection]）
2. 每个 section 内部按段落/句子边界切分
3. 贪心合并到 chunk_tokens 上限，构造 overlap_tokens 的尾部重叠
4. 每条 Document.page_content = chunk text，metadata 包含:
       paper_id / paper_title / section_name / section_type(abstract/intro/method/ref/...)
       chunk_index / chunk_token_count / start_char / end_char
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document

# ResearchPaper / PaperSection 在同一包内导入，避免硬依赖循环
try:
    from utils.research_paper_parser import ResearchPaper, PaperSection
except ImportError:
    # 允许在本包内独立调试
    ResearchPaper = None  # type: ignore
    PaperSection = None  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================
# 章节类型归一化
# ============================================================

# 把章节名归类为不同语义类型，用于应用不同的切分策略
# 顺序很重要：先用具体模式匹配，失败再 fallback 到包含匹配
_SECTION_TYPE_PATTERNS = [
    ("abstract", ["abstract", "摘要"]),
    ("introduction", ["introduction", "引言", "前言", "1.", "1 "]),
    ("related_work", ["related work", "background", "related_work", "相关工作", "综述"]),
    ("method", ["method", "methods", "approach", "model", "architecture", "方法", "模型", "网络结构"]),
    ("experiment", ["experiment", "experiments", "evaluation", "result", "results", "实验", "评估", "结果"]),
    ("conclusion", ["conclusion", "conclusions", "discussion", "结论", "结语", "讨论"]),
    ("reference", ["reference", "references", "bibliography", "参考文献", "引用"]),
    ("appendix", ["appendix", "appendices", "附录", "supplementary"]),
]


def normalize_section_name(raw_name: str) -> str:
    """把 ResearchPaperParser 提取出来的章节名归一化为可比较的小写 key。"""
    if not raw_name:
        return "unknown"
    s = raw_name.strip().lower()
    s = s.replace(" ", "_")
    return s


def classify_section_type(raw_name: str) -> str:
    """判断章节语义类型。unknown 表示走默认切分策略。"""
    name = (raw_name or "").strip().lower()
    for sem_type, patterns in _SECTION_TYPE_PATTERNS:
        for p in patterns:
            if p in name:
                return sem_type
    return "body"


# ============================================================
# Token 估算
# ============================================================

# CJK 统一汉字 + 扩展区 (粗略覆盖，避免 MCP 复杂依赖 unicode-db)
_CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
]


def _is_cjk_char(ch: str) -> bool:
    """判定单个字符是否属于 CJK 范围。"""
    if not ch:
        return False
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def estimate_token_len(text: str) -> int:
    """
    Token 长度估算（与 model-specific tokenizer 解耦）：
      - CJK 字符每个算 1 token
      - 非 CJK 部分按空白分词，token 数 = 词数
    """
    if not text:
        return 0
    cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk_count + non_cjk_tokens


# ============================================================
# 段落切分
# ============================================================

# 三种段落边界信号，优先级从高到低
_PARA_BREAK_RE = re.compile(r"\n\s*\n+")  # 空行（≥2 个连续 \n）
_SENTENCE_END_RE = re.compile(r"(?<=[。！？!?\.])\s*\n|\n(?=[A-Z(])")
_LIST_BREAK_RE = re.compile(r"\n\s*(?=\(\d+\)|\[\d+\]|\d+\.\s+[A-Z])")


def _strip_inline_artifacts(text: str) -> str:
    """去掉常见 PDF 解析残留：页眉页脚、连续空白等。"""
    # 页眉页脚：Page N of M
    text = re.sub(r"Page\s+\d+\s*(?:of|/)\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*-+\s*\d+\s*-+\s*$", " ", text, flags=re.MULTILINE)  # ---- 12 ----
    # 将 3+ 连续空白（包括换行空格混合）规范化为最多两个换行
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_section_to_paragraphs(section: "PaperSection") -> List[Dict]:
    """
    把一个 PaperSection 的 content 切分成段落列表。

    返回元素 dict:
        {
            "text": str,
            "para_index": int,          # 段落在 section 内的顺序编号
            "start_line": int,           # 段落在 section.content 的局部行号偏移
            "end_line": int,
        }
    """
    raw = section.content or ""
    if not raw.strip():
        return []

    cleaned = _strip_inline_artifacts(raw)
    lines = cleaned.splitlines()
    if not lines:
        return []

    paragraphs: List[Dict] = []

    def _flush(buf: List[str], start: int, end: int) -> None:
        text = "\n".join(buf).strip()
        if text:
            paragraphs.append({
                "text": text,
                "para_index": len(paragraphs),
                "start_line": start,
                "end_line": end,
            })

    buf: List[str] = []
    buf_start = 0
    for offset, line in enumerate(lines):
        if not line.strip():
            # 段落分隔：flush 当前 buf
            if buf:
                _flush(buf, buf_start, offset - 1)
                buf = []
            continue
        # 单行: 看是不是单行 list/sentence
        if not buf:
            buf_start = offset
        buf.append(line)
        # 列表边界或句子边界强制结束
        if _LIST_BREAK_RE.match("\n" + line) or _SENTENCE_END_RE.match(line + "\n"):
            pass  # 由空行 flush
    if buf:
        _flush(buf, buf_start, len(lines) - 1)

    # 失败回退：cleaned 没有空行则整段作为一个段落
    if not paragraphs and cleaned:
        paragraphs.append({
            "text": cleaned,
            "para_index": 0,
            "start_line": 0,
            "end_line": len(cleaned.splitlines()) - 1,
        })

    return paragraphs


# ============================================================
# 分块算法：贪心合并 + 尾部重叠
# ============================================================

@dataclass
class ChunkConfig:
    chunk_tokens: int = 512          # 每个 chunk 的目标 token 上限
    overlap_tokens: int = 64          # 相邻 chunk 的尾部重叠 token
    min_chunk_tokens: int = 50        # 小于此阈值的 chunk 丢弃或并入上 chunk
    max_chunk_tokens: int = 1024      # 硬上限（用于 abstract 等超长 section 截断）
    merge_adjacent_sections: bool = True  # chunk 不满时是否允许从下一节收尾
    preserve_sections: List[str] = field(default_factory=lambda: ["abstract"])


@dataclass
class SectionChunk:
    """分块算法中间态产物，便于测试时单独验证 chunker 逻辑。"""
    section_name: str
    section_type: str
    text: str
    token_count: int
    paragraphs: List[Dict]  # 该 chunk 包含的段落（合并时跨段落）

    def as_document(self, paper_id: str, paper_title: str, chunk_index: int,
                    start_char_in_section: int, end_char_in_section: int) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_name": self.section_name,
                "section_type": self.section_type,
                "chunk_index": chunk_index,
                "chunk_token_count": self.token_count,
                "start_char": start_char_in_section,
                "end_char": end_char_in_section,
            },
        )


def _join_paragraphs(paragraphs: List[Dict], joiner: str = "\n\n") -> str:
    return joiner.join(p["text"] for p in paragraphs)


def _chunk_from_paragraphs(
    paragraphs: List[Dict],
    section_name: str,
    section_type: str,
    config: ChunkConfig,
    start_chunk_index: int,
    into: List[Document],
    paper_id: str,
    paper_title: str,
    global_counter: List[int],
) -> None:
    """核心：将若干段落贪心合并为 chunk，写入 into 列表。"""
    cur: List[Dict] = []
    cur_tokens = 0

    def _flush_cur(overlap_paras: List[Dict]) -> None:
        """把 cur 写出，如果需要 overlap，把 overlap_paras 接在 cur 头上再吐新 chunk。"""
        if not cur:
            return
        text = _join_paragraphs(cur)
        token = estimate_token_len(text)
        idx = global_counter[0]
        global_counter[0] += 1
        section_start_offset = cur[0]["start_line"]
        section_end_offset = cur[-1]["end_line"]
        into.append(SectionChunk(
            section_name=section_name,
            section_type=section_type,
            text=text,
            token_count=token,
            paragraphs=cur,
        ).as_document(
            paper_id=paper_id,
            paper_title=paper_title,
            chunk_index=idx,
            start_char_in_section=section_start_offset,
            end_char_in_section=section_end_offset,
        ))
        # 准备 overlap: 从 cur 尾部往前累加 token 直到 overlap_tokens
        overlap: List[Dict] = []
        overlap_tok = 0
        for para in reversed(cur):
            t = estimate_token_len(para["text"])
            if overlap_tok + t > config.overlap_tokens:
                break
            overlap.insert(0, para)
            overlap_tok += t
        cur.clear()
        cur.extend(overlap_paras + overlap)
        # 设置当前 token 计数
        nonlocal cur_tokens
        cur_tokens = sum(estimate_token_len(p["text"]) for p in cur)

    for p in paragraphs:
        p_tok = estimate_token_len(p["text"])
        # 单个段落就已经超过 hard 限制 → 强制切成单段 chunk，再叠加 overlap tail
        if p_tok > config.max_chunk_tokens:
            # 先把当前 cur flush
            if cur:
                _flush_cur([])
            # 把这个段落作为一个独立 chunk（即使超长）
            text = p["text"]
            token = p_tok
            idx = global_counter[0]
            global_counter[0] += 1
            into.append(SectionChunk(
                section_name=section_name,
                section_type=section_type,
                text=text,
                token_count=token,
                paragraphs=[p],
            ).as_document(
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=idx,
                start_char_in_section=p["start_line"],
                end_char_in_section=p["end_line"],
            ))
            continue

        if cur_tokens + p_tok <= config.chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tok
        else:
            _flush_cur([])
            cur.append(p)
            cur_tokens = p_tok

    # 末尾 flush
    if cur:
        text = _join_paragraphs(cur)
        token = estimate_token_len(text)
        if token < config.min_chunk_tokens and into:
            # 并入上一个 chunk
            last = into.pop()
            merged_text = last.page_content + "\n\n" + text
            merged_meta = dict(last.metadata)
            merged_meta["chunk_token_count"] = estimate_token_len(merged_text)
            merged_meta["end_char"] = cur[-1]["end_line"]
            # 用一个替换的 Document（id 不变）
            global_counter[0] -= 1  # 抵消下一次会 +1
            merged = Document(page_content=merged_text, metadata=merged_meta)
            # 重置编号：因为上面的 last 的 chunk_index 已经被占用
            merged.metadata["chunk_index"] = len(into)
            into.append(merged)
            global_counter[0] = len(into)
        else:
            idx = global_counter[0]
            global_counter[0] += 1
            into.append(SectionChunk(
                section_name=section_name,
                section_type=section_type,
                text=text,
                token_count=token,
                paragraphs=cur,
            ).as_document(
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=idx,
                start_char_in_section=cur[0]["start_line"],
                end_char_in_section=cur[-1]["end_line"],
            ))


def _section_chunks(
    section: "PaperSection",
    section_type: str,
    config: ChunkConfig,
    paper_id: str,
    paper_title: str,
    into: List[Document],
    global_counter: List[int],
) -> None:
    """对单个 section 跑分块算法，按 section 类型应用特殊策略。"""
    paragraphs = split_section_to_paragraphs(section)
    if not paragraphs:
        return

    # abstract / reference 特殊处理
    if section_type == "abstract":
        text = _join_paragraphs(paragraphs)
        token = estimate_token_len(text)
        text = text[: config.max_chunk_tokens * 4]  # 字符级粗略截断（abstract 极短）
        idx = global_counter[0]
        global_counter[0] += 1
        into.append(Document(
            page_content=text,
            metadata={
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_name": section.name,
                "section_type": "abstract",
                "chunk_index": idx,
                "chunk_token_count": token,
                "start_char": paragraphs[0]["start_line"],
                "end_char": paragraphs[-1]["end_line"],
            },
        ))
        return

    if section_type == "reference":
        text = _join_paragraphs(paragraphs)
        token = estimate_token_len(text)
        # references 通常很长，硬上限截断
        if token > config.max_chunk_tokens:
            text = text[: config.max_chunk_tokens * 4]
            token = config.max_chunk_tokens
        idx = global_counter[0]
        global_counter[0] += 1
        into.append(Document(
            page_content=text,
            metadata={
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_name": section.name,
                "section_type": "reference",
                "chunk_index": idx,
                "chunk_token_count": token,
                "start_char": paragraphs[0]["start_line"],
                "end_char": paragraphs[-1]["end_line"],
            },
        ))
        return

    # 普通情况
    _chunk_from_paragraphs(
        paragraphs=paragraphs,
        section_name=section.name,
        section_type=section_type,
        config=config,
        start_chunk_index=0,
        into=into,
        paper_id=paper_id,
        paper_title=paper_title,
        global_counter=global_counter,
    )


# ============================================================
# 入口：process_paper_to_chunks
# ============================================================

def preprocess_text_for_embedding(text: str) -> str:
    """送 embedding 前的最终清理（合并重复换行 + 头尾 trim）。"""
    if not text:
        return ""
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_paper_to_chunks(
    paper: "ResearchPaper",
    paper_id: Optional[str] = None,
    config: Optional[ChunkConfig] = None,
) -> List[Document]:
    """
    入口函数：
      paper:        ResearchPaper 对象
      paper_id:     论文唯一 ID（若不传，使用 metadata.title 的 hash）
      config:       ChunkConfig（不传时使用默认 512/64）
    返回:           List[Document]，每条都带精细化 metadata
    """
    if ResearchPaper is not None and paper is not None and not isinstance(paper, ResearchPaper):
        # 允许传入 dict-like 鸭子类型
        pass

    if config is None:
        config = ChunkConfig()

    pid = paper_id or (getattr(getattr(paper, "metadata", None), "title", None) or "")
    pid = str(pid).strip() or "paper"

    paper_title = getattr(getattr(paper, "metadata", None), "title", "") or ""
    sections = getattr(paper, "sections", {}) or {}
    full_text = getattr(paper, "full_text", "") or ""

    # 当 sections 为空（parser 失败）→ 把 full_text 当成一个匿名 body section 处理
    if not sections and full_text:
        fake_section = PaperSection(
            name="full_text",
            content=full_text,
            start_line=0,
            end_line=len(full_text.splitlines()),
            word_count=len(full_text.split()),
        ) if PaperSection else None
        if fake_section:
            sections = {"full_text": fake_section}
        else:
            # 不依赖真实 PaperSection，简单返回一个直接包装的文档
            idx = 0
            return [Document(
                page_content=preprocess_text_for_embedding(full_text),
                metadata={
                    "paper_id": pid,
                    "paper_title": paper_title,
                    "section_name": "full_text",
                    "section_type": "body",
                    "chunk_index": idx,
                    "chunk_token_count": estimate_token_len(full_text),
                    "start_char": 0,
                    "end_char": len(full_text.splitlines()) - 1,
                },
            )]

    documents: List[Document] = []
    global_counter = [0]

    for raw_name, section in sections.items():
        sec_type = classify_section_type(raw_name)
        try:
            _section_chunks(
                section=section,
                section_type=sec_type,
                config=config,
                paper_id=pid,
                paper_title=paper_title,
                into=documents,
                global_counter=global_counter,
            )
        except Exception as e:  # 单个 section 失败不应中断全部
            logger.warning(f"分块失败,跳过 section={raw_name!r}: {e}")

    # 后处理：每个 doc 重新写 chunk_index 为连续编号（防御 counter 抖动）
    for i, doc in enumerate(documents):
        doc.metadata["chunk_index"] = i

    return documents


# ============================================================
# 单元测试入口（直接 python utils/chunk_processor.py 跑）
# ============================================================

def _selftest() -> None:
    """简易自检：构造一段混合中英文 + 段落的文本，验证分块结果。"""
    print("[chunk_processor] running selftest...")

    class _Meta:
        title = "SELFTEST"

    body = """INTRODUCTION

This paper proposes a new method for document chunking. It is widely used.

The key insight is straightforward. We explore several design choices.

(1) First, we discuss the algorithm design.
(2) Second, we discuss the evaluation methodology.

METHOD

Our method consists of three components: parser, indexer, and retriever.

The parser splits text into sections.

The indexer builds a sparse and dense representation.

The retriever combines both signals."""

    section = PaperSection(
        name="introduction", content=body.split("METHOD")[0], start_line=0, end_line=10, word_count=80,
    ) if PaperSection else None

    method_section = PaperSection(
        name="method",
        content="METHOD\n" + body.split("METHOD")[1],
        start_line=11, end_line=22, word_count=60,
    ) if PaperSection else None

    paper = ResearchPaper(
        metadata=_Meta(),
        sections={"introduction": section, "method": method_section} if section and method_section else {},
        full_text=body, page_count=1, total_word_count=140,
        key_terms=[], parsing_method="selftest", parsing_time="",
    ) if ResearchPaper else None

    cfg = ChunkConfig(chunk_tokens=60, overlap_tokens=10, min_chunk_tokens=10)
    docs = process_paper_to_chunks(paper or paper, paper_id="selftest", config=cfg)

    print(f"[chunk_processor] produced {len(docs)} docs")
    for d in docs:
        meta = d.metadata
        print(f"  - sec={meta.get('section_name')} idx={meta.get('chunk_index')} "
              f"tokens={meta.get('chunk_token_count')} len(text)={len(d.page_content)}")
    print("[chunk_processor] selftest OK")


if __name__ == "__main__":
    _selftest()
