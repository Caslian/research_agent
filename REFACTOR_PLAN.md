# 修改计划：将路由层重构为 Multi-Agent 驱动

## 一、核心原则

**路由层只做两件事**：Pydantic 参数校验 + HTTP 协议转换。
**Agent 层负责全部业务逻辑**：调用工具、操作数据库、调用外部 API。

改造后的统一数据流：
```
HTTP Request
  → api/routes/*.py : Pydantic 校验请求参数
  → agent_controller.submit_task(TaskType.XXX, validated_input)
  → agent_controller.execute_task(task_id)
  → Agent.run(input_data)
  → [LangGraph ReAct] LLM 决定调用哪些 Tool
  → Tool 内部写 PostgreSQL / Qdrant / Redis
  → 返回结果
  → api/routes/*.py : 格式化 HTTP Response
```

## 二、逐文件修改计划

### 2.1 api/routes/papers.py

| 项目 | 内容 |
|------|------|
| **当前状态** | 直接使用 `arxiv.Search()` 搜索，返回结果 |
| **目标状态** | 路由只做校验，搜索委托 HunterAgent |
| **修改方式** | 保留原有 `_build_arxiv_query()` 工具函数 → 委托给 `agent_controller.submit_task(PAPER_HUNTING)` → `execute_task()` → 返回结果 |
| **需要新增** | 无。AgentController 已有 `_execute_paper_hunting()` |
| **需要调整** | HunterAgent 返回的 papers 格式需与前端 API 约定一致 |
| **数据库写入** | 修改后：HunterAgent 自动写入 PostgreSQL papers 表 |

### 2.2 api/routes/analysis.py

| 项目 | 内容 |
|------|------|
| **当前状态** | 直接使用 `llm.ainvoke()` 分析论文（ArXiv URL 或本地 PDF） |
| **目标状态** | 路由解析输入 → 委托 MinerAgent 分析 → 返回报告 |
| **修改方式** | `/analyze` → `submit_task(PAPER_ANALYSIS)`；`/batch` → 已有 agent_controller 调用，统一下；`/upload-pdf` 保留在路由层（纯文件操作） |
| **需要新增** | MinerAgent.run() 需要支持直接传入 ArXiv URL 或解析后的文本（当前只支持 paper_id） |
| **需要调整** | 扩展 MinerAgent.run() 使其接收 paper_url 参数，内部完成 ArXiv 元数据获取 → PDF 解析 → 分析 |
| **数据库写入** | 修改后：MinerAgent 自动写入 analysis_reports + Qdrant L1/L2 |

### 2.3 api/routes/writing.py

| 项目 | 内容 |
|------|------|
| **当前状态** | `/coach` 直接调 LLM（硬编码 prompt）；`/explain`, `/polish`, `/mimic`, `/suggest` 返回 mock 数据 |
| **目标状态** | 所有写作端点委托 CoachAgent 处理 |
| **修改方式** | 每个端点 → `submit_task(WRITING_ASSISTANCE, {task_type, content, context})` → `execute_task()` |
| **需要调整** | CoachAgent.run() 已经接受 `task_type` + `content`，但需要 `user_id`；提供默认 user_id |
| **涉及端点** | `/coach` (改造)、`/explain` (从 mock 改为真实)、`/polish` (从 mock 改为真实)、`/mimic` (从 mock 改为真实)、`/suggest` (从 mock 改为真实) |
| **数据库写入** | 修改后：CoachAgent 内部可通过 Redis 缓存 |

### 2.4 api/routes/citations.py

| 项目 | 内容 |
|------|------|
| **当前状态** | `/validate` 有完整的 ArXiv/CrossRef/LLM 逻辑；`/generate` 返回 mock |
| **目标状态** | 路由只做输入解析 → 委托 ValidatorAgent → 返回格式化结果 |
| **修改方式** | `/validate` → `submit_task(CITATION_VALIDATION)`；`/generate` → 改为真实调用 |
| **需要调整** | ValidatorAgent.run() 当前接收 `paper_info` dict，需要支持直接传 `citation_text` 字符串 |
| **数据库写入** | 修改后：ValidatorAgent 自动写入 PostgreSQL reference_cache + Redis 缓存 |

### 2.5 api/routes/workflow.py

| 项目 | 内容 |
|------|------|
| **当前状态** | 直接导入其他路由函数（`from api.routes.papers import search_papers`），形成路由间强耦合 |
| **目标状态** | 通过 AgentController 的 `_execute_full_workflow()` 串联所有 Agent |
| **修改方式** | `/complete` → `submit_task(FULL_WORKFLOW)` → `execute_task()` |
| **需要删除** | 所有 `from api.routes.xxx import ...` 的跨路由导入 |
| **涉及端点** | `/complete` (重构)、`/search-and-analyze` (重构) |

### 2.6 Agent 内部适配

当前 Agent 的 `run()` 方法需要微调以适配路由层传入的数据格式：

| Agent | 当前 run() 需要的 input | 需要增加的灵活性 |
|-------|----------------------|----------------|
| HunterAgent | `{keywords, max_papers, sources, days_back}` | ✅ 已匹配，无需调整 |
| MinerAgent | `{paper_id, user_id, analysis_type}` | 增加支持 `{paper_url}` 模式 |
| CoachAgent | `{user_id, task_type, content, context}` | 增加默认 user_id（无用户系统时） |
| ValidatorAgent | `{paper_info, formats, verify_external}` | 增加支持 `{citation_text}` 纯文本输入 |

## 三、文件修改清单

```
修改文件:
├── api/routes/papers.py       — 委托 HunterAgent
├── api/routes/analysis.py     — 委托 MinerAgent
├── api/routes/writing.py      — 委托 CoachAgent（含 mock→真实改造）
├── api/routes/citations.py    — 委托 ValidatorAgent
├── api/routes/workflow.py     — 通过 AgentController 编排（删除跨路由导入）
├── agents/miner.py            — 增加 paper_url 输入支持
└── agents/validator.py        — 增加 citation_text 输入支持

不修改文件:
├── agents/base.py             — LangGraph ReAct 已就绪
├── agents/controller.py       — AgentController 已就绪
├── agents/hunter.py           — run() 接口已匹配
├── agents/coach.py            — run() 接口已匹配
├── core/*.py                  — 基础设施已就绪
└── api/main.py                — 生命周期已就绪
```

## 四、验证方案

修改完成后，执行 Phase 5 完整工作流测试（见测试场景文档），确认：

1. `POST /api/v1/papers/search` → 日志中出现 `"HunterAgent"` 或 `"agent_controller"`
2. `POST /api/v1/analysis/analyze` → 日志中出现 `"MinerAgent"` 或 LangGraph 工具调用
3. PostgreSQL `agent_execution_logs` 表中出现执行记录
4. Qdrant 向量 Collection 的 `points_count` > 0
5. Redis 中出现 `task_history` 条目
