# InnoCore AI（InnocoreAI-langchain）项目熟悉指南

> 基于源码逐行阅读的自顶向下项目梳理，帮你快速建立项目心智模型。所有内容均与 `InnocoreAI-langchain/` 仓库当前代码一致，可作为二次开发的参考地图。

---

## 0. 项目全景（一图流）

```
┌─────────────────────────────────────────────────────────┐
│  run.py  →  uvicorn(api.main:app)                       │
│           启动 FastAPI、初始化 DB/Qdrant/Redis/Controller│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  api/routes/* （REST + WebSocket）                       │
│  papers / users / tasks / analysis / writing / citations│
│  / workflow                                            │
└─────────────────────────────────────────────────────────┘
                          │ submit_task(TaskType, input_data)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  agents/controller.py  AgentController                  │
│  任务队列（asyncio.Queue + Redis 可选）                 │
│  日志写 PostgreSQL（agent_execution_logs / workflows） │
└─────────────────────────────────────────────────────────┘
                          │ _dispatch_task → 五大 TaskType
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Hunter   │  Miner    │  Coach    │  Validator │ 流程   │
│  ArXiv/   │  PDF 解析 │  润色/解释│  BibTeX/APA│  Hunter│
│  IEEE 搜索│  L1+L2 RAG│  风格迁移 │  CrossRef  │  →Miner│
│  + PDF下载│  创新点挖掘│  RAG 参考 │  Scholar  │  →Vali │
│          │  四段报告 │           │  DOI 校验 │  →Coach│
└─────────────────────────────────────────────────────────┘
            │             │             │             │
            ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│  core 层                                                │
│  config / database / redis_manager / llm_adapter /      │
│  vector_store（Qdrant L1+L2 双库 + 混合检索）           │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  utils 层                                               │
│  pdf_parser / research_paper_parser / embedding /       │
│  text_processor / citation_formatter                    │
└─────────────────────────────────────────────────────────┘
```

**技术栈关键点**
- **LLM**：`langchain_openai.ChatOpenAI`（OpenAI 兼容端点：OpenAI / DashScope / ModelScope / SiliconFlow / vLLM），可由 `.env` 中的 `OPENAI_*` 系列变量切换
- **Agent 框架**：基于 `langgraph.prebuilt.create_agent`（也称 `langchain.agents.create_agent`）构建 ReAct Agent + `InMemorySaver` checkpointer
- **Embedding**：兼容 OpenAI 兼容端点、本地 sentence-transformers、DashScope 三种 provider，自动探测维度
- **向量库**：`langchain_qdrant.QdrantVectorStore`，双 collection（`innocore_l1_preset` 全局库 / `innocore_l2_user` 个人库）
- **关系数据库**：`asyncpg` 直连 PostgreSQL，存论文 / 用户 / 报告 / 引用缓存 / Agent 日志
- **缓存/队列（可选）**：`redis.asyncio`，Sorted Set 做优先级队列、Hash 存活跃任务、List 存历史
- **Web**：`FastAPI` + `WebSocket`（流式任务状态）

---

## 1. 入口与配置（最先看懂的部分）

### 1.1 入口链路

1. `run.py` → `uvicorn.run("api.main:app", ..., reload=True)`，先 `sys.path` 注入项目根
2. `api/main.py` 的 `lifespan(app)` 在启动时按顺序初始化：

```python
await db_manager.initialize()                # PostgreSQL（可选，失败降级）
await embedding_service.initialize()         # EmbeddingService（utils/embedding.py）
await vector_store_manager.initialize(embedding_service=...)
await redis_manager.initialize()             # Redis（可选）
await agent_controller.initialize()
asyncio.create_task(agent_controller.start_task_processor())
```

3. 关闭时逆序释放资源

### 1.2 `core/config.py` 全局配置（`InnoCoreConfig` dataclass）

| 子配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `LLMConfig` | `model_name`、`api_key`、`base_url`、`temperature`、`max_tokens`、`timeout` | 通过 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 覆盖 |
| `VectorDBConfig` | `db_type`、`url`/`host:port`、`embedding_provider`（`openai`/`dashscope`/`local`）、`embedding_model` | 远程 Qdrant URL 优先；本地 host+port 兜底 |
| `DatabaseConfig` | PostgreSQL 连接 | `POSTGRES_*` 环境变量 |
| `RedisConfig` | Redis 连接 | 可选 |
| `ExternalAPIConfig` | `arxiv_base_url`、`crossref_api_key`、`serpapi_key` | IEEE/Google Scholar 校验 |
| 顶层 Agent | `agent_max_steps=5`、`agent_timeout=300`、`concurrent_agents=4` | |
| 顶层 RAG | `retrieval_top_k=5`、`similarity_threshold=0.7`、`hybrid_search_weights={vector:0.7, keyword:0.3}` | |

> ⚠️ **二次开发常踩的坑**：所有 dataclass 字段是 **模块级单例**（`config = InnoCoreConfig()`），如果做运行时热改模型，请改 `update_config(...)` 而不要直接 `setattr`。

---

## 2. BaseAgent 抽象层（所有 Agent 的父类）

文件：`agents/base.py`（220 行）

### 2.1 类签名与构造

```python
class BaseAgent(ABC):
    def __init__(self, name, llm=None, max_steps=None, timeout=None):
        self.name = name
        self.llm = llm or get_llm_adapter()    # core/llm_adapter.py：ChatOpenAI 包装
        self.checkpointer = InMemorySaver()     # LangGraph 会话记忆
        self.tools: List[Tool] = []
        self.tools_dict: Dict[str, Dict] = {}
        self.state = "idle"
```

### 2.2 三种"调用工具/推理"的能力

| 方法 | 用途 | 何时用 |
| --- | --- | --- |
| `add_tool(name, func, desc)` | 注册一个 LangChain `Tool`，同时存进 `self.tools` 和 `self.tools_dict` | 构造 Agent 时一次性注册 |
| `call_tool(tool_name, tool_input)` | **手动同步/异步调用**指定工具，自带超时与历史记录 | 硬编码流程（非 ReAct） |
| `run_with_tools(input_text, thread_id)` | 把任务交给 LangGraph **ReAct Agent**，让 LLM 自主决策调用哪个工具 | 真正发挥 Agent 自主性 |
| `think(prompt, context)` | 走一次纯 LLM 调用，不经过任何工具 | 内部 prompt 调用（写报告、做对比时常用） |
| `_build_agent_graph(system_prompt)` | 用 `create_agent(model, tools, system_prompt, checkpointer)` 构建图 | 第一次 `run_with_tools` 时惰性构建 |

### 2.3 抽象方法

```python
@abstractmethod
async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]

@abstractmethod
def get_required_fields(self) -> List[str]
```

子类必须实现 `run`（业务主入口）和 `get_required_fields`（注意 Hunter/Miner/Validator 都设了 `[]` 柔性校验，实际校验在 `run()` 内部按需分支）。

### 2.4 关键设计要点

- **双模式执行**：每个具体 Agent 既可以 `run_with_tools`（ReAct 自主模式），又可以在 `run()` 里**手写硬编码流程**（更可控）。当前 Hunter/Miner/Coach/Validator 几乎都是**手写硬流程 + 内部多次 `think()`**，只有当 `_build_agent_graph` 出错时才会退回 ReAct。
- **历史记录**：`_add_to_history` 仅在内存中保留最近 100 条（裁剪后保留 50 条），不持久化；持久化的执行日志在 `db_manager.log_agent_execution`。
- **超时控制**：所有 LLM/工具调用通过 `asyncio.wait_for(..., timeout=self.timeout)` 包装。

---

## 3. Hunter Agent（前哨探员）

文件：`agents/hunter.py`（373 行）

### 3.1 职责与工具清单

```python
def __init__(self, llm=None):
    super().__init__("Hunter", llm)
    self.arxiv_base_url = "http://export.arxiv.org/api/query"
    self.ieee_base_url   = "https://ieeexploreapi.ieee.org/api/v1"
    self.download_dir    = "downloads/papers"
    self.add_tool("search_arxiv",     self._search_arxiv,     "搜索ArXiv论文")
    self.add_tool("search_ieee",      self._search_ieee,      "搜索IEEE论文")
    self.add_tool("download_pdf",     self._download_pdf,     "下载PDF文件")
    self.add_tool("extract_metadata", self._extract_metadata, "提取论文元数据")
```

### 3.2 `run(input_data)` 主流程

```
input_data: { keywords: List[str], max_papers=20, sources=["arxiv","ieee"], days_back=1 }
  ├─ validate_input → state="running"
  ├─ for source in sources:
  │     _search_papers_from_arxiv / _search_papers_from_ieee
  ├─ _deduplicate_papers     # 标题 MD5 去重
  ├─ _filter_papers          # 关键词打分：标题+2，摘要+1，阈值≥1
  └─ for paper in filtered_papers[:max_papers]:
        _download_and_save_paper → 落盘 → _save_paper_to_db
返回 { total_found, unique_papers, filtered_papers, downloaded_papers, papers:[...] }
```

### 3.3 关键实现细节

#### 3.3.1 ArXiv 查询构造

```python
query_parts = [f'all:"{kw}"' for kw in keywords]
query = " OR ".join(query_parts)

if days_back > 0:
    start_date = (datetime.now()-timedelta(days=days_back)).strftime("%Y%m%d")
    date_filter = f"submittedDate:[{start_date}0000 TO {datetime.now().strftime('%Y%m%d')}2359]"
```

实际请求还带了 `max_results=max_papers*2`、`sortBy=submittedDate`、`sortOrder=descending`，**额外多取一倍结果用于后续打分筛选**。

#### 3.3.2 重试与限流

```python
max_retries = 3
for attempt in range(max_retries):
    if response.status == 429:
        await asyncio.sleep(3 * (attempt + 1))   # 退避
        continue
```

通用异常按 `2 * (attempt + 1)` 退避，三次都失败则记日志返回空数组（不抛出，避免工作流中断）。

#### 3.3.3 PDF 下载与入库

- 文件名 = `{arxiv_id}_{safe_title[:50]}.pdf`，已存在则直接复用
- 计算 SHA-256 作为 `content_hash`，传给 `db_manager.create_paper(...)`：

```python
paper_id = await db_manager.create_paper(
    title, authors, abstract, doi, file_path, content_hash, is_preset=False
)
```

- 入库前会先 `db_manager.get_paper_by_hash(content_hash)` 去重

### 3.4 工具方法（可被 ReAct 模式直接调用）

| Tool | 用途 | 实现 |
| --- | --- | --- |
| `_search_arxiv(query)` | 关键词逗号分隔，搜索最近 7 天 | 包装 `_search_papers_from_arxiv(kw, max=10, days=7)` |
| `_search_ieee(query)` | 同上，IEEE 源 | 包装 `_search_papers_from_ieee` |
| `_download_pdf(url)` | 单 PDF 下载到 `downloads/` | 用时间戳命名 |
| `_extract_metadata(path)` | 仅返回文件大小、时间戳 | **当前是占位实现**，无 PDF 解析 |

### 3.5 ⚠️ 已知局限（接二手时建议关注）

- `_extract_metadata` 工具是占位的，真正解析在 Miner
- IEEE 实现假设 `ieee_api_key` 已配置，但 `core/config.py` 里没有 `ieee_api_key` 字段，会触发 `config.ieee_base_url` 误判
- 去重只按标题 MD5，跨语言翻译版不会被合并

---

## 4. Miner Agent（洞察专家 / 项目的"核心大脑"）

文件：`agents/miner.py`（687 行）

### 4.1 职责与工具清单

```python
self.paper_parser = ResearchPaperParser()    # utils/research_paper_parser.py
self.add_tool("parse_pdf",      self._parse_pdf,      "解析PDF文件")
self.add_tool("search_memory",  self._search_memory,  "搜索记忆库")  # RAG 检索
self.add_tool("compare_papers", self._compare_papers, "对比论文")
self.add_tool("generate_report",self._generate_report,"生成分析报告")
```

### 4.2 `run(input_data)` 的三种输入模式

| 输入 | 触发分支 | 说明 |
| --- | --- | --- |
| `paper_url` 字段 | `_resolve_paper_from_url` → 用 `arxiv` 库查元数据并存库 | 支持 `arxiv.org/abs/XXXX`、`arxiv.org/pdf/XXXX`、`arXiv:XXXX`、`XXXX.XXXX` |
| `title` + `abstract` | 直接构造 `paper` 字典 | 不写库，`paper_id = "direct_input"` |
| `paper_id` | `db_manager.get_paper(paper_id)` | 数据库里有完整 PDF 时走深度解析 |
| 都不提供 | 抛 `AgentException` | |

### 4.3 Miner 主流程（六大步骤）

```
run(input_data):
  1. 解析 paper_id → paper 字典
  2. _parse_paper_content(paper)             # 走 ResearchPaperParser
  3. _find_related_papers(title, abstract, user_id)   # RAG：向量库混合检索
  4. _perform_comparison_analysis(...)       # LLM 生成结构化对比
  5. _create_analysis_report(...)            # LLM 生成四段式报告
                                            # (summary / innovation_points / limitations / future_ideas)
  6. _save_analysis_report → db_manager.create_analysis_report
     _update_vector_store → L1（全局）+ L2（用户）
```

### 4.4 RAG 检索细节（**重点关注**）

**入口**：

```python
search_results = await vector_store_manager.hybrid_search(
    query=f"{title} {abstract}",
    user_id=user_id,
    top_k=10,
    include_l1=True,
    include_l2=bool(user_id),
)
```

**返回结构**：

```python
[
    {
        "id": ...,
        "score": ...,
        "payload": {paper_id, title, abstract, ...},
        "collection_type": "l1" | "l2"
    },
    ...
]
```

**逻辑**：

- 从 `payload` 中拿 `paper_id` 再回查 PostgreSQL 拿完整论文信息
- 附上 `similarity_score` 与 `collection_type`

### 4.5 报告生成 prompt（核心 prompt）

`_create_analysis_report` 把以下上下文拼给 LLM，要求返回严格 JSON：

```json
{
  "summary": "...",
  "innovation_points": ["..."],
  "limitations": ["..."],
  "future_ideas": ["..."]
}
```

为兼容 LLM 返回大小写不一致，**末尾有 `_key_map` 兜底映射**（Summary/Innovation/Limitations/Future Ideas 等别名）。

### 4.6 向量库写入（双写）

`_update_vector_store` 把同一篇论文同时写入：

- **L2 用户库**（`vector_store_manager.add_to_l2(user_id, paper_id, ..., metadata)`）—— `user_id` 必填
- **L1 全局库**（`vector_store_manager.add_to_l1(paper_id, ...)`）—— `content` 截断到 2000 字

拼接内容顺序：

```
title + abstract + 关键词 (key_terms[:10]) + 元数据关键词 +
sections 中每个 section 的前 200 字 (带 [section_name] 前缀) + 技术亮点
```

### 4.7 工具方法

| Tool | 用途 |
| --- | --- |
| `_parse_pdf(file_path)` | 包装 `_extract_structured_content`（深度解析） |
| `_search_memory(query, user_id=None)` | 直接走 `hybrid_search` |
| `_compare_papers(current, related)` | 走 `_perform_comparison_analysis` |
| `_generate_report(paper_info, analysis_result)` | 走 `_create_analysis_report` |

### 4.8 ⚠️ 接二手时建议

- `_resolve_paper_from_url` 的 `arxiv` 库 import 在函数内部，是**延迟 import**，方便缺失时优雅降级
- 写入 L1/L2 的 metadata 字段可能因 `langchain_qdrant` 版本差异而 schema 不同，注意 `vector_store.py` 中提到的"metadata 嵌套在 payload['metadata']"细节
- `_find_related_papers` 当 `related_papers` 为空时**不会阻断**，会照常生成报告

---

## 5. Coach Agent（写作助教）

文件：`agents/coach.py`（605 行）

### 5.1 职责与工具清单

```python
self.add_tool("explain_concept",      self._explain_concept,      "解释复杂概念")
self.add_tool("polish_text",          self._polish_text,          "润色文本")
self.add_tool("mimic_style",          self._mimic_style,          "模仿写作风格")
self.add_tool("get_user_style",       self._get_user_style,       "获取用户写作风格")
self.add_tool("suggest_improvements", self._suggest_improvements, "建议改进")
```

### 5.2 四种任务类型（TaskConfig 注册表）

```python
self._task_configs = {
    "explain": TaskConfig(
        prompt_template=...,
        default_result={explanation, examples, importance, applications},
        ...
    ),
    "polish": TaskConfig(
        default_result={polished_text, modifications, style_suggestions, references},
        ...
    ),
    "mimic": TaskConfig(
        default_result={rewritten_text, style_analysis, mimic_techniques, reference_structures},
        ...
    ),
    "suggest": TaskConfig(
        default_result={overall_evaluation, improvement_suggestions,
                        grammar_issues, structure_suggestions, academic_improvements},
        ...
    ),
}
```

每个 `TaskConfig` 都包含：`prompt_template`、`default_result`、`context_builder`、`result_validator`、`success_message`。**这是 Coach 的核心抽象——模板方法模式**。

### 5.3 `run(input_data)` 主流程

```
input_data: { user_id, task_type, content, context }
  ├─ validate_input → 检查必填字段
  ├─ _validate_content → 内容非空 + ≤10000 字符
  ├─ _execute_task(user_id, content, context, task_type)
  │    ├─ context_builder()             # 任务专属上下文（RAG 风格参考等）
  │    ├─ prompt_template(content, ctx) # 构造 prompt
  │    ├─ think(prompt)                  # 走 LLM
  │    ├─ _parse_llm_json_response(...)  # 三阶段解析
  │    ├─ result_validator(...)          # 字段补全
  │    └─ TimeoutException / Exception → 返回 default_result 并标注失败信息
  └─ 返回 { status, task_type, user_id, result, processing_time_seconds, timestamp }
```

### 5.4 三阶段 JSON 解析（非常实用）

```python
# 1) 直接 json.loads
try:
    return json.loads(response)
except:
    pass

# 2) 找 outermost { ... } 块（比 markdown regex 更可靠）
start = response.find('{')
end = response.rfind('}')
if start != -1 and end > start:
    try:
        return json.loads(response[start:end+1])
    except:
        pass

# 3) 提取 ```json ... ``` 代码块
match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
if match:
    try:
        return json.loads(match.group(1))
    except:
        pass
```

这是项目中**唯一稳健的 LLM JSON 解析器**，可以抽出复用。

### 5.5 RAG 与上下文构建（Coach 的 RAG 关注点）

| 任务 | 上下文构建 |
| --- | --- |
| `explain` | `_get_cached_user_context(user_id)` → 从 PostgreSQL 读 `users.profile`（仅接受 UUID 格式 user_id） |
| `polish` | `_get_user_writing_style(user_id)` + `_get_style_references(user_id, content)`（**混合检索 L2 用户库 top-3**） |
| `mimic` | `context["target_style"]` + `context["reference_papers"]`（缺失时回退到 `_get_user_top_papers(user_id, limit=3)`） |
| `suggest` | `_get_user_writing_history(user_id)`（当前是占位模拟数据） |

**用户上下文缓存**：`_user_context_cache` 内存缓存 5 分钟（TTL 300s），按 `user_id` 缓存。

### 5.6 工具方法（兼容旧的直接调用）

- `_explain_concept(concept, context=None)`、`_polish_text(text, context=None)`、`_mimic_style(text, target_style, context=None)`、`_suggest_improvements(text, context=None)` 都委托到 `_handle_legacy_task` → `_execute_task`
- `_get_user_style(user_id)` 直接返回 `_get_user_writing_style` 结果

### 5.7 ⚠️ 接二手时建议

- Coach 是项目里 **LLM JSON 输出最依赖** 的 Agent，`MAX_CONTENT_LENGTH=10000` 限制可能在长文本润色时触发
- `_get_user_context` 中 `uuid_pattern` 校验，非 UUID 的 user_id 会直接返回 `{}`，避免崩溃但也意味着匿名用户拿不到风格
- `_get_user_writing_history` 仍返回模拟数据，是个**待补全的 TODO**

---

## 6. Validator Agent（校验官）

文件：`agents/validator.py`（706 行）

### 6.1 职责与工具清单

```python
self.crossref_base_url = "https://api.crossref.org/works"
self.google_scholar_url = "https://serpapi.com/search"
self.add_tool("generate_bibtex",  self._generate_bibtex,  "生成BibTeX引用")
self.add_tool("generate_apa",     self._generate_apa,     "生成APA格式引用")
self.add_tool("generate_ieee",    self._generate_ieee,    "生成IEEE格式引用")
self.add_tool("verify_metadata",  self._verify_metadata,  "校验元数据")
self.add_tool("crossref_lookup",  self._crossref_lookup,  "CrossRef查询")
self.add_tool("scholar_lookup",   self._scholar_lookup,   "Google Scholar查询")
```

### 6.2 `run(input_data)` 两种输入模式

```
input_data:
  ├─ citation_text  → _resolve_paper_from_citation
  │     ① ArXiv ID 提取 → arxiv 库查
  │     ② DOI 提取      → _crossref_lookup_by_doi
  │     ③ LLM 解析     → _llm_parse_citation
  └─ paper_info     → 直接使用
```

### 6.3 主流程

```
1. _generate_citations(paper_info, formats)  # 默认 ["bibtex","apa","ieee"]
2. _verify_paper_metadata(paper_info)        # CrossRef + Google Scholar 双重校验
3. _merge_citation_data(...)                 # 把验证状态 [% Verified]/
                                            # [% Discrepancies]/[% Unverified]
                                            # 拼到引用末尾
4. _cache_citation_results(...)              # DOI → BibTeX 入库 reference_cache
```

### 6.4 三种引用格式的生成细节

**BibTeX**（`_generate_bibtex_citation`）

- `citation_key = last_name + year + title前3词`
- `_determine_entry_type(paper_info)` 决策 `article` / `inproceedings` / `book` / `misc`
- 作者格式化为 "Last, First" 并用 `and` 连接

**APA**（`_generate_apa_citation`）

- 作者 1→2→3~7→>7 各自有不同分隔格式
- 期刊卷期格式 `Journal, vol(issue)`，DOI 拼 `https://doi.org/{doi}`

**IEEE**（`_generate_ieee_citation`）

- 最多列前 3 作者 + `et al.`
- 格式：`"Title," *Journal*, vol. X, no. Y, pp. Z, Mon. Year. doi: ...`

### 6.5 外部校验（`_verify_paper_metadata`）

```python
if doi:
    crossref_data = await _crossref_lookup_by_doi(doi)  # 解析 → 对比差异
if title:
    scholar_data = await _scholar_lookup_by_title(title)  # 解析 → 对比差异

status = "verified" if (crossref_verified or scholar_verified) and 无差异
       | "discrepancies_found"
       | "unverified"
```

- `_compare_metadata` 比对 title / authors / year
- `_calculate_similarity` 用 Jaccard 系数
- `_generate_corrections` 仅在相似度 > 0.8 时替换 title

### 6.6 工具方法（ReAct 可直接调用）

- `_generate_bibtex(paper_info)` / `_generate_apa(paper_info)` / `_generate_ieee(paper_info)` 直接返回字符串
- `_verify_metadata(paper_info)` 调 `_verify_paper_metadata`
- `_crossref_lookup(identifier)` 自动判 DOI（必须以 `10.` 开头）
- `_scholar_lookup(title)` 走 SerpApi

### 6.7 ⚠️ 接二手时建议

- 校验结果写入 `db_manager.cache_reference(doi, bibtex, is_verified)`，下次可缓存命中
- IEEE 格式的 `month` 字段没标准化，需要 paper_info 中预先提供
- Google Scholar 必须有 `SERPAPI_KEY`，否则会跳过并标记 `unverified`

---

## 7. AgentController（编排器 / 顶层调度）

文件：`agents/controller.py`（454 行）

### 7.1 五大任务类型

```python
class TaskType(Enum):
    PAPER_HUNTING        = "paper_hunting"        # → Hunter
    PAPER_ANALYSIS       = "paper_analysis"       # → Miner
    WRITING_ASSISTANCE   = "writing_assistance"   # → Coach
    CITATION_VALIDATION  = "citation_validation"  # → Validator
    FULL_WORKFLOW        = "full_workflow"        # → Hunter → Miner(×N) → Validator
```

### 7.2 控制器初始化

```python
self.agents = {
    "hunter":    HunterAgent(),
    "miner":     MinerAgent(),
    "coach":     CoachAgent(),
    "validator": ValidatorAgent(),
}
self.semaphore = asyncio.Semaphore(self.config.concurrent_agents)  # 默认 4 个并发
self.active_tasks, self.task_history, self.task_queue = ...
```

`initialize()` 会尝试连 Redis，连不上就降级用内存 `asyncio.Queue`。

### 7.3 任务生命周期

```
submit_task(task_type, input_data, priority, callback)
  ├─ 生成 task_id = "task_YYYYMMDD_HHMMSS_{序号}"
  ├─ 写入 self.active_tasks[task_id]
  ├─ 若 Redis 可用：zadd 任务队列 + hset 活跃任务
  ├─ 推到 asyncio.Queue (priority, task)
  └─ 返回 task_id

execute_task(task_id)
  ├─ 防重复执行（状态 != PENDING 直接返回）
  ├─ semaphore 控制并发
  ├─ state = RUNNING
  ├─ db_manager.log_agent_execution  → exec_id
  ├─ _dispatch_task(task) → 五大 handler 之一
  ├─ 成功 → state=COMPLETED + db_manager.update_agent_execution
  └─ 失败 → state=FAILED + 写 error_message

  finally:
    ├─ 拷到 task_history
    ├─ 从 active_tasks 删除
    └─ Redis: remove_active_task + push_task_history
```

### 7.4 `FULL_WORKFLOW` 的真实执行顺序

```python
async def _execute_full_workflow(self, task):
    # Stage 1: Hunter
    hunting_result = await self.agents["hunter"].run({
        "keywords": ..., "max_papers": ..., "sources": ["arxiv"]
    })
    papers = hunting_result.get("papers", [])

    # Stage 2: Miner — 对前 5 篇并行分析
    analysis_tasks = []
    for paper in papers[:5]:
        if paper.get("db_id"):
            miner_input = {"paper_id": paper["db_id"], "user_id": user_id, "analysis_type": "full"}
        elif paper.get("pdf_url"):
            miner_input = {"paper_url": paper["pdf_url"], "user_id": user_id, "analysis_type": "full"}
        elif paper.get("title"):
            miner_input = {
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper.get("authors", []),
                "user_id": user_id,
            }
        analysis_tasks.append(self.agents["miner"].run(miner_input))

    analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    # Stage 3: Validator — 仅当 validate_citations=True
    if input_data.get("validate_citations", False):
        for paper in papers:
            v_result = await self.agents["validator"].run({...})
            paper["citations"] = v_result.get("citations", {})

    return { stages: {hunting}, analysis_reports, final_papers }
```

### 7.5 事件回调系统

```python
self.event_callbacks = {
    "task_started", "task_completed", "task_failed", "agent_status_changed"
}
add_event_callback(event_type, callback)
_trigger_event(event_type, data)  # 异步 / 同步回调都支持
```

### 7.6 ⚠️ 接二手时建议

- `start_task_processor()` 是后台循环，但你也可以**同步**调 `execute_task`，它会绕过 queue 直接跑
- `concurrent_agents` 改了直接生效（`asyncio.Semaphore`）
- `priority` 越大越优先（Redis 用 `-priority` 做 score，`zpopmin` 弹最小）

---

## 8. 核心服务层（core/）

### 8.1 `core/llm_adapter.py` — LLM 适配器

```python
class LLMAdapter:
    self.llm = ChatOpenAI(
        model, api_key, base_url, temperature, max_tokens, timeout,
        streaming=False  # 禁用流式以保持兼容
    )
```

**对外接口**：`ainvoke(prompt)`、`invoke(prompt)`、`batch(prompts)`；都返回字符串（内部用 `_extract_content` 兼容 `str` / `AIMessage` / `ChatResult`）。

**几个细节**：

- `_format_messages` 支持纯字符串、字典列表、Message 对象三种输入
- 用 `asyncio.to_thread` 把同步 invoke 包成异步，避免阻塞事件循环
- 全局单例 `get_llm_adapter()`，与 `BaseAgent` 共享同一个 LLM

### 8.2 `core/database.py` — PostgreSQL（asyncpg）

**8 张表**：

| 表 | 用途 |
| --- | --- |
| `users` | 用户账号 + JSONB profile |
| `papers` | 论文元数据 + content_hash 去重 + is_preset 标记 |
| `user_paper_relations` | 多对多 + tags/rating/is_read |
| `analysis_reports` | 四段式报告 (summary / innovation_point / limitation / future_idea) |
| `reference_cache` | 按 DOI 缓存 BibTeX + is_verified |
| `agent_execution_logs` | Controller 写入的 Agent 执行日志 |
| `agent_tool_calls` | 每次工具调用的输入/输出/耗时 |
| `workflow_executions` | 完整 workflow 状态（user_id 不级联，保留历史） |

**关键方法**：

- `create_paper(title, authors, abstract, doi, file_path, content_hash, is_preset)` / `get_paper_by_hash` / `get_user_papers`
- `create_analysis_report(...)` / `get_analysis_report(paper_id, user_id)`
- `cache_reference(doi, bibtex, is_verified)` / `get_cached_reference(doi)`
- `log_agent_execution` / `update_agent_execution` / `log_tool_call`
- `create_workflow` / `update_workflow` / `get_workflow`
- `get_table_counts` 用于 `verify_databases.py` 健康检查

**初始化降级**：`pool=None` 时所有方法自动跳过（`asyncpg` 未安装或 DB 不可用），项目仍可只跑向量库 + LLM。

### 8.3 `core/redis_manager.py` — Redis 管理

**功能矩阵**：

| 方法 | 数据结构 | 用途 |
| --- | --- | --- |
| `push_task` / `pop_task` | Sorted Set | 任务优先级队列（score=-priority） |
| `set_active_task` / `get_active_task` / `remove_active_task` | Hash `active_tasks` | 正在运行的任务快照 |
| `push_task_history` / `get_task_history` | List `task_history`（LTRIM 1000） | 历史任务回看 |
| `cache_set` / `cache_get` / `cache_delete` | String + TTL | 通用 KV 缓存 |
| `set_agent_state` / `get_agent_state` | Hash + TTL（86400s） | Agent 会话状态 |
| `publish` | Pub/Sub | Agent 间事件总线 |

> ⚠️ **降级策略**：所有 Redis 方法在 `self.redis is None` 时直接 `return` 不抛异常，确保 Redis 挂了不会让主流程失败。

### 8.4 `core/vector_store.py` — 向量存储（最重要的 RAG 基础设施）

**双 Collection**：

- `innocore_l1_preset`：全局库，所有用户共享
- `innocore_l2_user`：用户库，按 `metadata.user_id` 过滤

**初始化流程**：

```
initialize(embedding_service):
  ├─ QdrantClient(url=远程或本地, api_key, https, prefer_grpc=False)
  ├─ LangChainEmbeddings(embedding_service) 包装层
  ├─ _get_embedding_dimension:
  │    dashscope provider → 直接推断维度（避免 400）
  │    else → aembed_query("dimension probe") → 失败回退到 sync → 最终回退
  ├─ _create_collections(dim):
  │    对每个 collection：
  │      ├─ get_collection
  │      ├─ 维度不匹配 → delete + create
  │      ├─ 404 → create
  │      └─ 给 payload 字段建 keyword index（L1: paper_id/source；L2: 加 user_id）
  └─ _init_langchain_vectorstores
       ├─ 检查 langchain-qdrant 版本是否支持 validate_collection_config
       ├─ L1: QdrantVectorStore(client, collection_name, embedding)
       └─ L2: QdrantVectorStore(client, collection_name, embedding)
```

**关键方法**：

```python
async def add_to_l1(paper_id, title, abstract, content, metadata=None) -> str
async def add_to_l2(user_id, paper_id, title, abstract, content, metadata=None) -> str
async def hybrid_search(query, user_id=None, top_k=5, include_l1=True, include_l2=True) -> List[Dict]
async def get_user_vectors(user_id, limit=100)
async def delete_user_vectors(user_id)
```

**LangChainEmbeddings 包装**（`core/vector_store.py:31-117`）：

- **专用后台 event loop**：`asyncio.new_event_loop()` + daemon thread，避免与业务 loop 冲突
- **DashScope 特殊处理**：直接用 httpx POST DashScope OpenAI 兼容端点，**绕过** LangChain `OpenAIEmbeddings` 内置的 tiktoken 预分词（DashScope 不接受 token id 数组）
- v3/v4 embedding 强制指定 `dimensions: 1024`

**混合检索实现**：

```python
async def hybrid_search(query, user_id, top_k, include_l1, include_l2):
    # L1: similarity_search_with_score (无 filter)
    # L2: similarity_search_with_score, filter=metadata.user_id == user_id
    # 合并 → 关键词分数 (Jaccard) * keyword_weight 加成 → 排序
    vector_weight = config.hybrid_search_weights.get("vector", 0.7)
    keyword_weight = config.hybrid_search_weights.get("keyword", 0.3)
    for result in results:
        keyword_score = _calculate_keyword_score(query, title + abstract)
        result["score"] += keyword_score * keyword_weight
```

### 8.5 `core/exceptions.py`

层次化异常类（`InnoCoreException` 基类）：

- `AgentException` / `VectorStoreException` / `DatabaseException` / `LLMException`
- `PDFParsingException` / `ExternalAPIException` / `ConfigurationException`
- `ValidationException` / `TimeoutException` / `ResourceExhaustedException`

---

## 9. 工具层（utils/）

### 9.1 `utils/pdf_parser.py` — PDF 解析

**核心类**：`PDFParser`，三种入口

- `parse_pdf(file_path)` → 用 `langchain_community.PDFPlumberLoader.load()`
- `parse_pdf_from_bytes(pdf_bytes, filename)` → 写临时文件再调 loader
- `parse_pdf_from_url(url)` → `aiohttp` 下载 → 同上

**返回结构**：

```python
{
    "success": bool,
    "title": str,
    "authors": List[str],
    "abstract": str,
    "full_text": str,
    "page_count": int,
    "word_count": int,
    "metadata": {creator, producer, subject, keywords, source},
    "documents": List[Document]
}
```

**文本提取启发式**：

- `_extract_title`：元数据 Title 优先，否则前 10 行扫描排除 Abstract/Introduction/arXiv 等
- `_extract_authors`：元数据 Author 优先，否则扫描前 20 行含 `@` / `university` 的行
- `_extract_abstract`：先用 5 种 Abstract 正则（含中英文 + 通用回退），失败则跳过前 300 字符再取 1500

### 9.2 `utils/research_paper_parser.py` — 深度结构化解析

**数据类**：

```python
@dataclass
class PaperMetadata:
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_date: Optional[str] = None
    venue: Optional[str] = None           # 发表刊物或会议

@dataclass
class PaperSection:
    name: str
    content: str
    start_line: int
    end_line: int
    word_count: int

@dataclass
class ResearchPaper:
    metadata: PaperMetadata
    sections: Dict[str, PaperSection]
    full_text: str
    page_count: int
    total_word_count: int
    key_terms: List[Tuple[str, float]]    # (词语, 重要性分数)
    parsing_method: str
    parsing_time: str
```

**SECTION_PATTERNS 字典**：覆盖 abstract / introduction / related_work / method / experiment / result / conclusion / references 八个标准章节（中英文 + 数字编号都支持）

**关键流程**：

```
parse_paper(file_path):
  ├─ pdf_parser.parse_pdf(file_path)
  ├─ _extract_metadata
  ├─ _extract_sections   # 按 SECTION_PATTERNS 逐个正则提取
  │                       # 每个部分截断 50000 字符
  ├─ _extract_key_terms  # 从 abstract + method + introduction 中提词频
  │     词频分数：在 abstract 中 ×2.0，在 method 中 ×1.5
  └─ 构造 ResearchPaper 对象
```

**创新洞察提取** `extract_innovation_insights(paper)`：

- `innovation_indicators`：摘要里包含的创新关键词（novel/innovative/new/propose/首次/提出 等）
- `technical_highlights`：方法部分含元数据关键词的句子
- `methodology_novelty`：基于 method vs related_work 字数比
- `experimental_uniqueness`：基于实验关键词命中

### 9.3 `utils/embedding.py` — Embedding 服务

```python
class EmbeddingService:
    async def initialize():
        if provider == "local":
            # sentence-transformers 模型
            self.embeddings = LocalEmbeddings(model_name, device)
        else:
            self.embeddings = OpenAIEmbeddings(
                model, api_key, base_url,
                check_embedding_ctx_length=False,  # 关键：避免 DashScope 400
            )

class LocalEmbeddings(Embeddings):
    """sentence-transformers 模型，支持 Qwen3-Embedding-0.6B / bge / MiniLM 等"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # normalize_embeddings=True（适配 Qdrant COSINE）

    def embed_query(self, text: str) -> List[float]:
        ...
```

**辅助能力**：

- 内存 cache（`_clean_text` + MD5 缓存键）
- `generate_paper_embedding(paper_info)`：组合 title + abstract + authors + sections
- `calculate_similarity` / `find_most_similar` / `cluster_texts` / `extract_keywords`（TF-IDF 简化版）

### 9.4 `utils/text_processor.py` — 文本预处理

常用 NLP 工具：

- `clean_text` / `tokenize` / `remove_stop_words`（内置英文停用词）
- `extract_sentences` / `extract_paragraphs`
- `calculate_readability`（Flesch、句长、词长）
- `calculate_similarity`（Jaccard / 余弦）
- `find_similar_texts`

被 `ResearchPaperParser` 用于关键词提取过滤。

### 9.5 `utils/citation_formatter.py` — 引用格式化（备用工具）

> ⚠️ **目前未被 Agent 直接调用**！Validator 自己实现了 `_generate_bibtex/apa/ieee_citation`，`citation_formatter.py` 是历史遗留工具，**如果要复用，可以把 Validator 内部的实现替换成它**。

支持 BibTeX / APA / IEEE / MLA / Chicago / Harvard 等格式。

---

## 10. API 层（api/）

### 10.1 `api/main.py` 启动入口

- 创建 FastAPI app，注册 7 个 router（CORS 全开）
- `lifespan` 内做完整初始化（已述于 §1.1）
- `/` 返回 `frontend/index.html`
- `/health` 汇总各组件状态

### 10.2 路由全景

| Prefix | File | 核心端点 | 委托 Agent |
| --- | --- | --- | --- |
| `/api/v1/papers` | `papers.py` | `POST /search`, `POST /upload` | HunterAgent |
| `/api/v1/users` | `users.py` | CRUD 用户 + profile | 直接 db_manager |
| `/api/v1/tasks` | `tasks.py` | `POST /submit`, `GET /{id}/execute`, `WS /ws/{id}`, `WS /ws/stream` | Controller |
| `/api/v1/analysis` | `analysis.py` | `POST /analyze`（支持本地 PDF / ArXiv URL）, `POST /compare`, `POST /innovation/search`, `POST /upload-pdf`, `POST /batch` | MinerAgent |
| `/api/v1/writing` | `writing.py` | `POST /coach`, `/explain`, `/polish`, `/mimic`, `/suggest`, `/user/{id}/style`, `/user/{id}/templates` | CoachAgent |
| `/api/v1/citations` | `citations.py` | `POST /validate`, `POST /generate`, `GET /formats` | ValidatorAgent |
| `/api/v1/workflow` | `workflow.py` | `POST /complete`, `POST /search-and-analyze`, `GET /status/{id}` | Controller.FULL_WORKFLOW |

### 10.3 关键 API 流程示例

**搜索论文**（`POST /api/v1/papers/search`）：

```json
// Request
{ "keywords": "wifi fingerprint", "source": "arxiv", "limit": 10 }

// 内部
_build_arxiv_query("wifi fingerprint")
  → "all:wifi AND all:fingerprint"   // 关键修复：避免短语搜索的 6000+ 误匹配

agent_controller.submit_task(TaskType.PAPER_HUNTING, {keywords, max_papers, sources})
agent_controller.execute_task(task_id)
```

**完整工作流**（`POST /api/v1/workflow/complete`）：

```json
// Request
{
  "keywords": "machine learning",
  "analysis_type": "summary",
  "citation_format": "bibtex",
  "writing_task": "生成报告",
  "limit": 5
}

// 内部
submit_task(FULL_WORKFLOW, {
  keywords: [...], max_papers: 5, sources: ["arxiv"],
  validate_citations: True, citation_format: "bibtex", writing_task: ...
})
→ Controller 内部：Hunter → Miner×5 → Validator → 拼装 4 个 steps 返回
```

### 10.4 WebSocket 协议

- `/api/v1/tasks/ws/{task_id}`：每秒推一次任务状态，终止于 completed/failed/cancelled
- `/api/v1/tasks/ws/stream`：通用流式通道
  - `{"type":"writing_assistance", "data":{...}}` 调 CoachAgent
  - `{"type":"ping"}` 触发 pong

---

## 11. RAG 功能全景（**重点回顾**）

### 11.1 双库架构（L1 全局 + L2 用户）

```
写入路径：
  Hunter.download → paper (PostgreSQL papers 表)
  Miner.analyze → _update_vector_store
    ├─ add_to_l2(user_id, paper_id, ...)   # 含 user_id 过滤
    └─ add_to_l1(paper_id, ...)            # 全局共享

查询路径：
  Coach._get_style_references / Miner._find_related_papers
    └─ vector_store_manager.hybrid_search(
            query, user_id, top_k,
            include_l1=True, include_l2=bool(user_id)
       )
```

### 11.2 检索混合评分公式

```
final_score = vector_score * 0.7 + keyword_jaccard_score * 0.3
```

权重由 `core.config.hybrid_search_weights` 控制。

### 11.3 元数据 schema（写入 Qdrant payload）

| 字段 | 写入位置 | 说明 |
| --- | --- | --- |
| `metadata.user_id` | L2 | 用户过滤 |
| `metadata.paper_id` | L1/L2 | 回查 PostgreSQL |
| `metadata.source` | L1/L2 | 来源标记 |
| `metadata.collection_type` | L1/L2 | `"l1"` 或 `"l2"` |
| `metadata.title` / `metadata.abstract` | L1/L2 | 冗余存储便于展示 |
| `metadata.{key_terms_count, page_count, sections_identified, ...}` | L2 | Miner 写入的解析元数据 |

> 注意：**LangChain QdrantVectorStore 会把 Document.metadata 嵌套到 `payload["metadata"]` 下**，所以 filter 路径必须写 `metadata.user_id`，这是项目里多处显式注释强调的点。

---

## 12. 关键调优与坑点速查（接二手必读）

| 模块 | 坑点 | 解决建议 |
| --- | --- | --- |
| `llm_adapter.py` | DashScope 严格要求 `input: list[str]`，不能用 token id | 已在 `vector_store.py` 的 `LangChainEmbeddings._post_dashscope` 用 httpx 直连绕过 |
| `vector_store.py` | `langchain-qdrant` 0.1+ 才支持 `validate_collection_config` | 已做版本兼容 try/except |
| `vector_store.py` | collection 维度与 embedding 不匹配会写入失败 | 启动时 `_get_embedding_dimension` 探测 + 不匹配则重建 |
| `database.py` | 缺 asyncpg 或 DB 不可达会启动失败 | 整段 try/except 降级为无 DB 模式 |
| `redis_manager.py` | Redis 不可达 | 所有方法 `if not self.redis: return` 静默降级 |
| `controller.py` | 任务重复提交 | `execute_task` 显式校验 `status == PENDING` |
| `controller.py` | `_execute_full_workflow` 限制 Miner 只跑前 5 篇 | `for paper in papers[:5]`，可调 |
| `miner.py` | `_find_related_papers` 找不到相关论文不阻断 | 仍然生成报告（合理设计） |
| `coach.py` | `_get_user_writing_history` 返回硬编码模拟数据 | TODO：接入真实数据库 |
| `validator.py` | Google Scholar 必须有 `SERPAPI_KEY` | 缺 key 时跳过并标记 `unverified` |
| `hunter.py` | IEEE 实现假设 `ieee_api_key` 在 config 中，实际字段缺失 | 永远走"配置缺失跳过"分支 |
| `hunter.py` | `_extract_metadata` 工具是占位 | 真正解析在 Miner |
| LLM JSON 解析 | LLM 返回经常带 markdown 代码块或多余解释 | Coach 的 `_parse_llm_json_response` 三阶段方案可复用 |
| 向量库写入 | `metadata` 在 payload 里嵌套 | filter 路径写 `metadata.xxx` |

---

## 13. 二次开发路线建议（按"风险×价值"排序）

### Phase 1：低风险补全（1~2 天）

1. 替换 `Hunter._extract_metadata` 为真实 PDF 解析
2. 把 `utils/citation_formatter.py` 接到 `Validator` 替换内联实现
3. 把 `Coach._get_user_writing_history` 接入 PostgreSQL
4. 修复 `Hunter` 的 IEEE 配置读取（加 `ieee_api_key` 字段）

### Phase 2：核心能力升级（3~5 天）

1. **Miner 双库写入事务化**：确保 L1+L2+PG 三者一致，失败回滚
2. **Coach 流式输出**：当前 `ainvoke` 强制 `streaming=False`，可换为 `astream` 改善长文润色体验
3. **RAG 评估**：在 `vector_store_manager` 加召回率 / MRR 指标
4. **Controller 持久化**：当前 `task_history` 在内存，重启即丢，可落 Redis/PG

### Phase 3：架构演进（1~2 周）

1. **引入 LangGraph StateGraph 替换手写硬流程**：把 Hunter/Miner/Coach/Validator 节点化
2. **LLM 调用统一抽象**：`core/llm_adapter` + `utils/embedding` 都用工厂注入，便于做多模型路由
3. **API 鉴权**：当前 `users.py` 已经预留 `db_manager`，加 JWT/OAuth 即可
4. **前端 / 后端解耦**：把 `writing.py` 的 `_COACH_TASK_MAP` 提到配置
5. **Observability**：接入 LangSmith / OpenTelemetry，统一 trace 一次 `FULL_WORKFLOW` 的所有 Agent

---

## 14. 文件索引（按重要性排序）

| 文件 | 行数 | 重要性 | 摘要 |
| --- | --- | --- | --- |
| `agents/base.py` | 220 | ⭐⭐⭐⭐⭐ | 抽象基类，ReAct 双模式 |
| `agents/hunter.py` | 373 | ⭐⭐⭐⭐ | 论文搜索 + PDF 下载 |
| `agents/miner.py` | 687 | ⭐⭐⭐⭐⭐ | RAG + 深度分析 + L1/L2 写入 |
| `agents/coach.py` | 605 | ⭐⭐⭐⭐ | 四任务模板方法 + JSON 解析 |
| `agents/validator.py` | 706 | ⭐⭐⭐ | 引用生成 + CrossRef/Scholar 校验 |
| `agents/controller.py` | 454 | ⭐⭐⭐⭐ | 任务调度 + 工作流编排 |
| `core/vector_store.py` | 715 | ⭐⭐⭐⭐⭐ | Qdrant L1/L2 + 混合检索 |
| `core/database.py` | 462 | ⭐⭐⭐⭐ | PostgreSQL 8 张表 |
| `core/llm_adapter.py` | 149 | ⭐⭐⭐ | OpenAI 兼容 LLM |
| `core/redis_manager.py` | 138 | ⭐⭐ | 任务队列 + 缓存 |
| `core/config.py` | 184 | ⭐⭐⭐ | 全局配置 |
| `core/exceptions.py` | 50 | ⭐ | 异常类层次 |
| `utils/research_paper_parser.py` | 390 | ⭐⭐⭐⭐ | 论文结构化解析 |
| `utils/pdf_parser.py` | 330 | ⭐⭐⭐ | PDF 加载 + 元数据提取 |
| `utils/embedding.py` | 472 | ⭐⭐⭐ | Embedding 服务 |
| `utils/text_processor.py` | 371 | ⭐⭐ | NLP 工具 |
| `utils/citation_formatter.py` | 526 | ⭐ | 备用引用格式化（未使用） |
| `api/main.py` | 220 | ⭐⭐⭐ | FastAPI 入口 + lifespan |
| `api/routes/*.py` | - | ⭐⭐⭐ | 7 个路由模块 |
| `run.py` | 34 | ⭐ | 启动脚本 |

---

## 15. 一句话总结

> **InnoCore AI = LangGraph ReAct Agent 框架 + LangChain Qdrant 双层 RAG + asyncpg/Redis/可选降级的工程化实现**。`BaseAgent` 提供工具注册 + 三种调用方式（call_tool / think / run_with_tools）；`Controller` 做任务调度和持久化日志；`Miner` 是 RAG 与 L1/L2 写入的中枢；`Coach` 的 `TaskConfig` 模板方法 + 三阶段 JSON 解析是最值得复用的工程实践。
