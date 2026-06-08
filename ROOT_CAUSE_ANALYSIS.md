# 根因分析：为什么 Agents 没有参与工作流

## 一、问题现象

调试发现，所有 API 请求的处理逻辑都发生在 `api/routes/` 目录下的路由文件中，`agents/` 目录下的 HunterAgent、MinerAgent、CoachAgent、ValidatorAgent **虽然被实例化了，但从未被路由调用**。

当前实际数据流：
```
HTTP Request → api/routes/*.py (硬编码业务逻辑) → arxiv/LLM/CrossRef 直接调用 → Response
```

应有的数据流：
```
HTTP Request → api/routes/*.py (参数校验) → AgentController → Agent.run() → LangGraph ReAct → Tools → Response
```

## 二、逐路由根因分析

### 2.1 api/routes/papers.py — `/search`

**当前代码做的事**（第 82-156 行）：
```python
@router.post("/search")
async def search_papers(request: PaperSearchRequest):
    search = arxiv.Search(query=..., max_results=request.limit, ...)  # 直接调 arxiv 库
    for result in search.results():
        papers.append({...})
    return {"papers": papers, ...}
```

**缺少的部分**：
- 没有调用 `agent_controller.submit_task(TaskType.PAPER_HUNTING, ...)`
- 没有通过 HunterAgent 的 `_save_paper_to_db()` 写入 PostgreSQL
- 没有通过 HunterAgent 的工具链索引到 Qdrant
- HunterAgent 的 4 个 LangChain Tool (`search_arxiv`, `search_ieee`, `download_pdf`, `extract_metadata`) **从未被执行**

---

### 2.2 api/routes/analysis.py — `/analyze`

**当前代码做的事**（第 58-323 行）：
```python
@router.post("/analyze")
async def analyze_paper(request: PaperAnalysisRequest):
    paper = arxiv.Search(...)                # 直接调 arxiv
    response = await llm.ainvoke(prompt)      # 直接调 LLM
    return {"analysis": analysis_content, ...}
```

**缺少的部分**：
- 没有调用 `agent_controller.submit_task(TaskType.PAPER_ANALYSIS, ...)`
- MinerAgent 的 `_find_related_papers()` 没有被调用 → **Qdrant hybrid_search 永远不会触发**
- MinerAgent 的 `_update_vector_store()` 没有被调用 → **Qdrant L1/L2 永远为空**
- MinerAgent 的 `ResearchPaperParser` 深度解析没有被调用
- MinerAgent 的 4 个 LangChain Tool 从未被执行

---

### 2.3 api/routes/writing.py — `/coach`, `/explain`, `/polish`, `/mimic`, `/suggest`

**当前代码做的事**（第 68-150 行）：
```python
@router.post("/coach")
async def writing_coach(request: WritingCoachRequest):
    response = await llm.ainvoke(prompt)  # 直接调 LLM
    return {"result": result_content, ...}
```

而 `/explain`, `/polish`, `/mimic`, `/suggest` 端点**完全返回硬编码的模拟数据**（如 `"[Detailed explanation of ...]"`）。

**缺少的部分**：
- 没有调用 `agent_controller.submit_task(TaskType.WRITING_ASSISTANCE, ...)`
- CoachAgent 的 `_get_cached_user_context()` 缓存逻辑没有被调用
- CoachAgent 的 `TaskConfig` 驱动框架（4 种任务的 Prompt 模板 + 上下文构建器 + 结果验证器）**全部跳过**

---

### 2.4 api/routes/citations.py — `/validate`

**当前代码做的事**（第 43-275 行）：
```python
@router.post("/validate")
async def validate_citation(request: CitationValidationRequest):
    # 1. 正则匹配 ArXiv ID → 直接调 arxiv 库
    # 2. 正则匹配 DOI → 直接调 CrossRef API
    # 3. 失败后直接调 LLM
    # 4. 硬编码生成 BibTeX/APA/IEEE/MLA 格式
```

**缺少的部分**：
- 没有调用 `agent_controller.submit_task(TaskType.CITATION_VALIDATION, ...)`
- ValidatorAgent 的 `_cache_citation_results()` 没有被调用 → **PostgreSQL reference_cache 永远为空**
- ValidatorAgent 的 6 个 LangChain Tool 从未被执行
- `_generate_bibtex_citation()`, `_generate_apa_citation()` 等格式化方法被绕过

---

### 2.5 api/routes/workflow.py — `/complete`, `/search-and-analyze`

**当前代码做的事**：
```python
from api.routes.papers import search_papers       # 直接导入其他路由函数
from api.routes.analysis import analyze_paper      # 直接导入其他路由函数
from api.routes.citations import validate_citation  # 直接导入其他路由函数

@router.post("/complete")
async def complete_workflow(request: WorkflowRequest):
    search_result = await search_papers(...)       # 调 papers.py 的函数
    analysis_result = await analyze_paper(...)      # 调 analysis.py 的函数
    # ...
```

这是一个**"路由之间相互调用"**的反模式——`workflow.py` 导入 `papers.py` 的函数，这些函数内部又直接调 `arxiv`，始终绕过了 Agent 层。

---

## 三、根本原因总结

| 问题 | 严重级别 | 说明 |
|------|---------|------|
| **路由层持有全部业务逻辑** | 致命 | 所有 5 个路由文件都包含完整的 API 调用、格式化和返回逻辑 |
| **Agent 层被架空** | 致命 | 4 个 Agent 的 `run()` 方法只在 `api/routes/tasks.py` 的 `/submit` + `/execute` 路径被调用，主要用户流程不走这里 |
| **路由之间相互调用** | 严重 | `workflow.py` 直接导入 `papers.py`/`analysis.py` 的函数，形成路由间强耦合 |
| **LangGraph ReAct 从未被触发** | 严重 | 所有 Agent 的 `_build_agent_graph()` 和 `run_with_tools()` 方法从未在正常 API 流程中执行 |
| **Qdrant 写入路径缺失** | 严重 | 即使通过 `/tasks` 路径调用 HunterAgent，其 `_save_paper_to_db()` 也只写 PostgreSQL，不索引 Qdrant |
| **模拟数据端点** | 中等 | writing.py 的 `/explain`, `/polish`, `/mimic`, `/suggest` 返回硬编码 mock 数据 |
| **引用缓存未写入** | 中等 | citation 路由有独立的格式化逻辑，从不调用 ValidatorAgent 的缓存方法 |

## 四、影响范围

```
被绕过的核心功能：
├── agents/base.py        : BaseAgent.run_with_tools() — LangGraph ReAct 从未在正常流程执行
├── agents/hunter.py      : _save_paper_to_db(), _download_and_save_paper() — PostgreSQL 写入未触发
├── agents/miner.py       : _find_related_papers(), _update_vector_store() — Qdrant 读写未触发
├── agents/coach.py       : _execute_task(), 4 个 TaskConfig — 整个配置驱动框架未使用
├── agents/validator.py   : _cache_citation_results() — PostgreSQL 引用缓存未触发
├── agents/controller.py  : submit_task(), execute_task(), _execute_full_workflow() — 仅 /tasks 路由使用
├── core/database.py      : 新增的 agent_execution_logs 等表永远为空
├── core/vector_store.py  : L1/L2 集合永远为空
└── core/redis_manager.py : task_queue/task_history 仅在 /tasks 路由使用
```
