# InnoCore AI - 研创·智核

**智能科研创新助手 | Intelligent Research Innovation Assistant**

[Python 3.11](https://www.python.org/downloads/) · [FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [License](LICENSE)

*基于多智能体协作 + 双层知识库的科研全流程自动化系统*
*基于 LangChain / LangGraph 框架构建，支持灵活的 LLM 切换与本地向量模型*
*原项目地址：[InnoCore AI](https://github.com/A-pricity/innocore-ai)*

简体中文

---

## 📖 项目简介

InnoCore AI（研创·智核）是一个基于 **LangChain / LangGraph** 框架构建的智能科研创新助手系统。通过多智能体协作与**双层知识库（L1 全局 + L2 私有）**，实现从论文搜索、深度分析、写作辅助、引用校验到 **基于私有知识库的 RAG 问答**的科研全流程自动化。

### 核心特性

- 🤖 **多智能体协作**：Hunter / Miner / Coach / Validator 协同工作
- 🔗 **LangChain + LangGraph 驱动**：基于 LangChain 的 Agent / Tool / Memory / VectorStore，并使用 LangGraph 编排工作流
- 🔄 **双模式支持**：单独模式（精细控制）+ 协调模式（一键完成）
- 📚 **智能论文分析**：自动解析 PDF，提取章节/关键术语，生成深度分析报告
- 📂 **双层知识库（KB）**：每个用户可建多个私有 KB，PDF 上传 → SHA-256 去重入库 → 按章节-段落粒度 chunk → KB 范围 RAG 聊天
- ✍️ **AI 写作助手**：学术润色、风格转换、实时写作建议
- 🔍 **引用智能校验**：自动识别 DOI/ArXiv ID，生成多种格式引用
- 🎯 **工作流自动化**：一键完成搜索→分析→引用→报告全流程
- 🧠 **本地向量模型支持**：支持 sentence-transformers 等本地 Embedding，无需联网

### 技术亮点

- **LangChain / LangGraph 集成**：Agent / Tool / Memory / VectorStore / LangGraph 编排
- **OpenAI 兼容 API**：支持 OpenAI、ModelScope、DashScope、SiliconFlow 等多种 LLM 提供商
- **PDF 深度解析**：使用 LangChain Document Loaders + 自研 `ResearchPaperParser`，结构化提取元数据 / 章节 / 关键词
- **KB 感知 RAG**：`chunk_processor` 按 section / paragraph / token 三级粒度切块，向量化后按 `kb_id` 限定检索范围
- **混合检索**：LangChain QdrantVectorStore + 关键词匹配，提升检索准确度
- **流式输出**：WebSocket 实时传输，提供流畅的交互体验
- **异步架构**：基于 FastAPI 异步框架，高性能并发处理
- **模块化设计**：清晰的分层架构（agents / api / core / utils / frontend），易于扩展

## 🎯 应用场景

| 角色 | 价值 |
| --- | --- |
| 📖 研究生 / 博士生 | 快速了解研究领域，辅助文献综述与论文写作 |
| 👨‍🏫 高校教师 | 跟踪最新研究进展，辅助课题申报与综述 |
| 🔬 企业研发人员 | 技术调研、专利分析、竞品研究 |
| 📝 学术写作者 | 论文润色、引用管理、格式规范 |

### 典型使用场景

1. **文献综述**：自动搜索相关论文 → 批量分析 → 生成综述报告
2. **论文写作**：实时润色建议 → 引用自动生成 → 格式规范检查
3. **私人知识库 RAG**：上传 PDF → 自动入库 → 按 KB 范围问答
4. **学术翻译**：中英互译 → 学术表达优化 → 术语标准化

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端界面层                            │
│  论文搜索 | 深度分析 | 写作助手 | 引用管理 | 知识库面板 │
│  (含 KB 弹窗 / PDF 智能上传 / KB 范围聊天)              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   API 接口层 (FastAPI)                   │
│  papers / users / tasks / analysis / writing            │
│  / citations / workflow / kb / chat                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│             智能体编排层 (LangGraph + Agents)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 🕵️Hunter │  │ 🧠 Miner │  │ ✍️ Coach │  │ 🔎Valid. │ │
│  │ 论文搜索 │  │ 深度分析  │  │ 写作辅助  │  │ 引用校验  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  核心服务层 (LangChain)                  │
│  PDF Loaders | Qdrant VectorStore | Chat Models         │
│  Embeddings (OpenAI / 本地) | chunk_processor           │
│  ResearchPaperParser | KnowledgeBaseManager             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     数据持久层                           │
│  PostgreSQL(users/papers/kb/分析/日志) | Qdrant(L1/L2)  │
│  Redis(缓存) | 文件存储(PDF)                             │
└─────────────────────────────────────────────────────────┘
```

### 四大智能体

| 智能体 | 职责 | 核心能力 | LangChain 组件 |
| --- | --- | --- | --- |
| 🕵️ **Hunter** | 论文搜索与监控 | ArXiv/IEEE 实时搜索、智能过滤、自动下载 | Tools, AgentExecutor |
| 🧠 **Miner** | 深度分析与挖掘 | PDF 解析、创新点提取、对比分析、报告生成 | Document Loaders, Chains |
| ✍️ **Coach** | 写作辅助与润色 | 学术润色、风格转换、实时建议、术语优化 | Chat Models, Prompts |
| 🔎 **Validator** | 引用校验与格式化 | DOI 验证、多格式生成、元数据校验、标准化 | Tools, Output Parsers |

### LangChain / LangGraph 核心组件

- **Chat Models**：`langchain-openai.ChatOpenAI` — 支持 OpenAI 兼容 API
- **Embeddings**：`langchain-openai.OpenAIEmbeddings` 或 `sentence-transformers` 本地模型
- **VectorStore**：`langchain-qdrant.QdrantVectorStore` — 双层 collection（L1 全局 / L2 KB）
- **Document Loaders**：`langchain-community.PDFPlumberLoader` — PDF 文档加载
- **Text Splitters**：`chunk_processor` 自研 — 按章节-段落-小段三级粒度切块
- **Agents**：`langchain.agents.AgentExecutor` — 智能体执行器
- **LangGraph**：工作流编排（替代旧版 AgentExecutor）
- **Tools**：`langchain_core.tools.Tool` — 工具定义
- **Prompts**：`langchain_core.prompts.ChatPromptTemplate` — 提示模板

## 🚀 快速开始

### 1. 环境要求

| 组件 | 版本 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| Python | 3.11+ | ✅ | 推荐 3.11，避免 3.12+ 部分依赖不兼容 |
| PostgreSQL | 13+ | ✅ | 用户/论文/KB 元数据存储 |
| Qdrant | 1.5+ | ✅ | 向量存储（已支持本地文件 / Docker / 服务） |
| Redis | 6+ | ⚠️ 可选 | 缓存层，不启动也能跑 |
| LLM API | — | ✅ | OpenAI / 兼容 API / 本地 vLLM |

### 2. 安装依赖

推荐使用 `conda`（或 `venv`）创建独立环境后，通过 `requirements.txt` 一键安装：

```bash
# 创建并激活独立环境（任选其一）
conda create -n RA python=3.11
conda activate RA

# 或者使用 venv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

# 安装全部依赖
pip install -r requirements.txt
```

> ⚠️ 如果只想跑**最小核心**（不依赖向量库），可用 `clean_requirements.txt` 减重，KB / RAG 聊天功能将被禁用。

### 3. 启动基础设施

```bash
# PostgreSQL（任选其一）
# A) 使用已运行的实例：跳过
# B) Docker 一键起：
docker run -d --name innocore-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15

# Qdrant 向量库
docker run -d --name innocore-qdrant -p 6333:6333 qdrant/qdrant:latest

# Redis（可选）
docker run -d --name innocore-redis -p 6379:6379 redis:7-alpine
```

### 4. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key + 数据库连接信息
```

`.env` 主要字段：

```ini
# LLM（OpenAI 兼容）
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1     # 也可指向 ModelScope / DashScope / SiliconFlow / 本地 vLLM
LLM_MODEL=gpt-4o-mini

# Embedding（二选一）
# A) 远端 OpenAI 兼容
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
# B) 本地 sentence-transformers
# USE_LOCAL_EMBEDDING=true
# LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# PostgreSQL
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/innocore

# Qdrant
QDRANT_URL=http://localhost:6333

# Redis（可选）
REDIS_URL=redis://localhost:6379/0
```

### 5. 启动应用

```bash
python run.py
```

或直接用 uvicorn（更细的控制）：

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后会看到：

```
Server will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs
Health check at: http://localhost:8000/health
```

### 6. 访问入口

| 入口 | URL |
| --- | --- |
| 主应用 | http://localhost:8000 |
| Swagger API 文档 | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |
| KB 管理面板 | http://localhost:8000 → 侧边栏"我的知识库" |

## 📚 知识库（KB）使用指南

知识库是本项目特色功能之一。每个用户可创建多个 KB，将 PDF 上传后入库，聊天时按 KB 范围检索，避免全量召回噪声。

### 数据流

```
PDF 上传
  ↓
SHA-256 内容哈希（content_hash 唯一索引去重）
  ↓
存入 papers 表（拿到 paper_id）
  ↓
用户选择加入 KB（写入 kb_paper_relations）
  ↓
KnowledgeBaseManager 异步触发：
  ResearchPaperParser 解析 → chunk_processor 切块 →
  VectorStoreManager 按 kb_id 限定写入对应 collection
  ↓
聊天时按 kb_id / kb_ids 检索范围限定 RAG
```

### 关键设计

| 设计点 | 方案 |
| --- | --- |
| 一篇 paper 同时属于多个 KB | `kb_paper_relations` 多对多表，**不在 paper 表加 kb_id 列** |
| 检索范围隔离 | Qdrant payload 携带 `kb_id`，检索时 filter 强制限定 |
| 切块粒度 | 按 section（abstract/introduction/method/...）→ paragraph → token-aware split + overlap |
| abstract 强制单 chunk | 进入向量库不破坏语义 |
| 同步日志 | `kb_paper_chunks_log` 记录每次 chunk 状态（processing/ready/failed） |

### 主要 API

| Method & Path | 用途 |
| --- | --- |
| `POST /api/v1/analysis/upload-pdf` | 上传 PDF（带 SHA-256 去重，返回 paper_id） |
| `GET /api/v1/kb?user_id=xxx` | 列出我的 KB |
| `POST /api/v1/kb` | 创建 KB |
| `POST /api/v1/kb/{kb_id}/papers` | 把 paper 加入 KB |
| `GET /api/v1/kb/{kb_id}/papers` | 列出 KB 内的论文 |
| `DELETE /api/v1/kb/{kb_id}/papers/{paper_id}` | 从 KB 移除论文 |
| `POST /api/v1/chat` | KB 范围 RAG 聊天，`{user_id, kb_id\|kb_ids, query, top_k}` |

## 📁 项目结构

```
innocore_ai/
├── agents/                       # AI 智能体（LangChain Agents）
│   ├── base.py                   # BaseAgent 基类
│   ├── hunter.py                 # 论文搜索
│   ├── miner.py                  # 深度分析
│   ├── coach.py                  # 写作助手
│   ├── validator.py              # 引用校验
│   └── controller.py             # LangGraph 工作流编排
│
├── api/                          # REST API 路由层（FastAPI）
│   ├── main.py                   # FastAPI 入口（注册所有 router）
│   └── routes/
│       ├── papers.py             # 论文管理
│       ├── users.py              # 用户管理（email MVP 取/创建）
│       ├── tasks.py              # 任务管理
│       ├── analysis.py           # PDF 上传 + 分析报告
│       ├── writing.py            # 写作辅助
│       ├── citations.py          # 引用校验
│       ├── workflow.py           # 工作流
│       ├── knowledge_base.py     # ★ 知识库 CRUD + 加入论文
│       └── chat.py               # ★ KB 范围 RAG 聊天
│
├── core/                         # 核心服务层
│   ├── config.py                 # 配置加载（基于 .env）
│   ├── database.py               # PostgreSQL 管理器（users/papers/kb/日志 表）
│   ├── vector_store.py           # Qdrant VectorStore（L1 全局 / L2 KB 双层）
│   ├── llm_adapter.py            # LLM 适配器（OpenAI 兼容）
│   ├── knowledge_base_manager.py # ★ KB 管理 + 入库编排
│   └── exceptions.py             # 统一异常
│
├── utils/                        # 工具层
│   ├── pdf_parser.py             # PDF 解析（LangChain Loaders）
│   ├── research_paper_parser.py  # ★ 结构化论文解析（章节/关键词/元数据）
│   ├── chunk_processor.py        # ★ 按 section/paragraph/token 三级切块
│   ├── embedding.py              # Embedding 适配（远端 / 本地）
│   └── ...
│
├── tests/                        # 单元测试
│   └── test_chunk_processor.py   # ★ chunk_processor 离线测试
│
├── docs/                         # 项目文档
│   └── PROJECT_OVERVIEW.md       # ★ 自顶向下项目梳理（1095 行）
│
├── frontend/                     # Web 前端
│   ├── index.html                # 主界面（含 KB 面板 / PDF 智能上传 / KB 范围聊天）
│   └── static/css/style.css
│
├── run.py                        # 启动脚本（uvicorn api.main:app）
├── requirements.txt              # 完整依赖
├── clean_requirements.txt        # 核心依赖（不含向量库等可选）
├── .env.example                  # 配置示例
└── README.md                     # 本文档
```

## ⚙️ 配置项速查

`.env` 中常用配置：

| 字段 | 说明 |
| --- | --- |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | LLM 接入信息 |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | Embedding（远端） |
| `USE_LOCAL_EMBEDDING=true` + `LOCAL_EMBEDDING_MODEL` | Embedding（本地 sentence-transformers） |
| `DATABASE_URL` | PostgreSQL 连接串 |
| `QDRANT_URL` | Qdrant 服务地址 |
| `REDIS_URL` | Redis（可选） |
| `USE_FALLBACK_LLM` | LLM 失败时是否启用本地备援 |

支持的 LLM 提供商：

- **OpenAI**：默认 `LLM_BASE_URL=https://api.openai.com/v1`
- **ModelScope**：`LLM_BASE_URL=https://api-inference.modelscope.cn/v1`
- **DashScope**：`LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- **SiliconFlow**：`LLM_BASE_URL=https://api.siliconflow.cn/v1`
- **本地 vLLM**：`LLM_BASE_URL=http://localhost:8001/v1`

## 🧪 测试

```bash
# chunk_processor 单元测试（离线、无需向量库）
python -m tests.test_chunk_processor
```

## 📊 性能指标（参考）

| 模块 | 耗时 |
| --- | --- |
| 论文搜索（ArXiv API） | ~5 秒 |
| PDF 解析 | ~3 秒/篇 |
| 深度分析（含 AI 推理） | ~20 秒/篇 |
| 写作润色（流式首字） | ~2 秒 |
| 引用校验（含外部 API） | ~3 秒/条 |
| 完整工作流（3 篇+分析+引用+报告） | ~70 秒 |
| KB 入库（典型论文 30 chunk） | ~3-5 秒 |

## 🛣️ 路线图

### v1.0（已完成）✅
- [x] 四大智能体基础功能（Hunter / Miner / Coach / Validator）
- [x] PDF 深度解析（LangChain Document Loaders + ResearchPaperParser）
- [x] 双模式工作流（单独模式 + 协调模式）
- [x] Web 界面 + API 文档
- [x] LangChain / LangGraph 框架迁移
- [x] **双层知识库（KB）端到端**：PDF 上传 → SHA-256 去重 → KB 管理 → chunk 入库 → KB 范围 RAG 聊天
- [x] **本地向量模型支持**（sentence-transformers）
- [x] **结构化切块**（abstract 单 chunk + section/paragraph/token 三级粒度）

### v1.1（计划中）
- [ ] 用户系统完善：JWT / 登录注册 / 权限管理（当前是 email MVP 模式）
- [ ] KB 高级特性：共享 KB / 团队协作 / KB 导入导出
- [ ] 增量索引与混合检索（BM25 + dense）参数化
- [ ] 流式 RAG 回答（SSE / WebSocket 输出）
- [ ] 历史记录与收藏功能
- [ ] LangSmith 调试与监控集成
- [ ] 移动端适配

### v2.0（未来）
- [ ] 个性化写作风格学习
- [ ] 多语言支持（界面 + 模型层面）
- [ ] 多模态（PDF 图表理解）
- [ ] 图谱检索 / KG-RAG
- [ ] 离线 / 私有化部署优化

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'feat: add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://python.langchain.com/) — LLM 应用开发框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 工作流编排
- [FastAPI](https://fastapi.tiangolo.com/) — 现代 Web 框架
- [ArXiv API](https://arxiv.org/help/api) — 学术论文数据源
- [Qdrant](https://qdrant.tech/) — 向量数据库
- [InnoCore AI](https://github.com/A-pricity/innocore-ai) — A-pricity 原创项目

---

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**
