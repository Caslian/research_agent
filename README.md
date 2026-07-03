# InnoCore AI - 研创·智核



**智能科研创新助手 | Intelligent Research Innovation Assistant**

[Python](https://www.python.org/downloads/)
[FastAPI](https://fastapi.tiangolo.com/)
[LangChain](https://python.langchain.com/)
[License](LICENSE)

*基于多智能体协作的科研全流程自动化系统*
*基于 LangChain 框架构建，支持灵活的 LLM 切换*
*原项目地址：[InnoCore AI](https://github.com/A-pricity/innocore-ai)*

[English](README_EN.md) | 简体中文



---

## 📖 项目简介

InnoCore AI（研创·智核）是一个基于 **LangChain** 框架构建的智能科研创新助手系统。通过多智能体协作，实现从论文搜索、深度分析、写作辅助到引用校验的科研全流程自动化。

### 核心特性

- 🤖 **多智能体协作**：四大智能体（Hunter/Miner/Coach/Validator）协同工作
- 🔗 **LangChain 驱动**：基于 LangChain 框架，支持多种 LLM 和工具集成
- 🔄 **双模式支持**：单独模式（精细控制）+ 协调模式（一键完成）
- 📚 **智能论文分析**：自动解析 PDF，提取关键信息，生成深度分析报告
- ✍️ **AI 写作助手**：学术润色、风格转换、实时写作建议
- 🔍 **引用智能校验**：自动识别 DOI/ArXiv ID，生成多种格式引用
- 🎯 **工作流自动化**：一键完成搜索→分析→引用→报告全流程

### 技术亮点

- **LangChain 集成**：使用 LangChain 的 Agent、Tool、Memory、VectorStore 等核心组件
- **OpenAI 兼容 API**：支持 OpenAI、ModelScope、DashScope、SiliconFlow 等多种 LLM 提供商
- **PDF 深度解析**：使用 LangChain Document Loaders 进行学术论文的结构化提取
- **混合检索**：LangChain VectorStore + 关键词匹配，提升检索准确度
- **流式输出**：WebSocket 实时传输，提供流畅的交互体验
- **异步架构**：基于 FastAPI 异步框架，高性能并发处理
- **模块化设计**：清晰的分层架构，易于扩展和维护

## 🎯 应用场景

### 适合谁使用？

- 📖 **研究生/博士生**：快速了解研究领域，辅助论文写作
- 👨‍🏫 **高校教师**：跟踪最新研究进展，辅助课题申报
- 🔬 **企业研发人员**：技术调研，专利分析，竞品研究
- 📝 **学术写作者**：论文润色，引用管理，格式规范

### 典型使用场景

1. **文献综述**：自动搜索相关论文 → 批量分析 → 生成综述报告
2. **论文写作**：实时润色建议 → 引用自动生成 → 格式规范检查
3. **研究调研**：追踪特定主题 → 创新点挖掘 → 研究方向建议
4. **学术翻译**：中英互译 → 学术表达优化 → 术语标准化

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端界面层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 论文搜索  │  │ 深度分析  │  │ 写作助手  │  │ 引用管理 │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   API 接口层                             │
│  FastAPI + WebSocket + RESTful API                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 智能体编排层 (LangChain Agents)          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 🕵️Hunter │  │ 🧠 Miner│  │ ✍️ Coach│  │ 🔎 Validator│ │
│  │ 论文搜索 │  │ 深度分析  │  │ 写作助手  │  │ 引用校验  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   核心服务层 (LangChain)                 │
│  PDF Loaders | VectorStore | Chat Models | Embeddings   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   数据持久层                             │
│  PostgreSQL | Qdrant | Redis | 文件存储                  │
└─────────────────────────────────────────────────────────┘
```

### 四大智能体


| 智能体              | 职责       | 核心能力                      | LangChain 组件             |
| ---------------- | -------- | ------------------------- | ------------------------ |
| 🕵️ **Hunter**   | 论文搜索与监控  | ArXiv/IEEE 实时搜索，智能过滤，自动下载 | Tools, AgentExecutor     |
| 🧠 **Miner**     | 深度分析与挖掘  | PDF 解析，创新点提取，对比分析，报告生成    | Document Loaders, Chains |
| ✍️ **Coach**     | 写作辅助与润色  | 学术润色，风格转换，实时建议，术语优化       | Chat Models, Prompts     |
| 🔎 **Validator** | 引用校验与格式化 | DOI 验证，多格式生成，元数据校验，标准化    | Tools, Output Parsers    |


### LangChain 核心组件

本系统基于 LangChain 框架构建，使用了以下核心组件：

- **Chat Models**: `langchain-openai.ChatOpenAI` - 支持 OpenAI 兼容 API
- **Embeddings**: `langchain-openai.OpenAIEmbeddings` - 文本向量化
- **VectorStore**: `langchain-qdrant.QdrantVectorStore` - 向量存储与检索
- **Document Loaders**: `langchain-community.PDFPlumberLoader` - PDF 文档加载
- **Agents**: `langchain.agents.AgentExecutor` - 智能体执行器
- **Tools**: `langchain_core.tools.Tool` - 工具定义
- **Memory**: `langchain.memory.ConversationBufferMemory` - 对话记忆
- **Prompts**: `langchain_core.prompts.ChatPromptTemplate` - 提示模板

## 快速开始

### 1. 安装依赖

推荐使用 `conda`（或 `venv`）创建独立环境后，通过 `requirements.txt` 一键安装全部依赖：

```bash
# 创建并激活独立环境（任选其一）
conda create -n RA python=3.11
conda activate RA

# 或者使用 venv
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# 安装全部依赖（LangChain / LangGraph / OpenAI / Qdrant / PDF 解析 等）
pip install -r requirements.txt
```

>

### 2. 配置

复制示例配置文件并填入你自己的 API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 LLM API Key（或兼容 API）
```

支持多种 LLM 提供商：

- **OpenAI**: 设置 `LLM_API_KEY` 和 `LLM_BASE_URL`（可选）
- **ModelScope**: 设置 `LLM_BASE_URL=https://api-inference.modelscope.cn/v1`
- **DashScope**: 设置 `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- **SiliconFlow**: 设置 `LLM_BASE_URL=https://api.siliconflow.cn/v1`
- **自定义 vLLM**: 设置 `LLM_BASE_URL` 为您的 vLLM 端点

### 3. Run Application

```bash
python run.py
```

### 4. Access

- Main Application: [http://localhost:8000](http://localhost:8000)
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

## 功能特性

### 工作模式

- **Individual Mode（单独模式）**：针对特定任务独立使用每一个智能体
- **Workflow Mode（协调模式）** ⭐：自动化完成的全流程工作流，多个智能体协同工作

### 智能体

- 🕵️ **Hunter Agent**：负责文献搜索与跟踪监控
- 🧠 **Miner Agent**：深度论文分析与创新点挖掘
- ✍️ **Coach Agent**：写作辅助与学术风格润色
- 🔎 **Validator Agent**：引用真实性校验与多格式排版

### 工作流自动化

一键完成完整科研流程：

1. 论文搜索（Hunter）
2. 内容分析（Miner）
3. 生成引用（Validator）
4. 输出报告（Coach）

## Project Structure

```
innocore_ai/
├── agents/          # AI agents (LangChain Agents)
│   ├── base.py      # BaseAgent with LangChain AgentExecutor
│   ├── hunter.py    # Paper search agent
│   ├── miner.py     # Deep analysis agent
│   ├── coach.py     # Writing assistant agent
│   ├── validator.py # Citation validator agent
│   └── controller.py# Agent orchestration
├── api/             # REST API routes
├── core/            # Core functionality
│   ├── llm_adapter.py    # LangChain ChatOpenAI adapter
│   ├── vector_store.py   # LangChain QdrantVectorStore
│   └── ...
├── models/          # Data models
├── services/        # Business logic
├── utils/           # Utilities
│   ├── embedding.py      # LangChain Embeddings
│   ├── pdf_parser.py     # LangChain Document Loaders
│   └── ...
├── frontend/        # Web interface
├── main.py          # Main application entry
├── run.py           # Simple run script
└── requirements.txt # Dependencies
```

## Requirements

- Python 3.8+
- OpenAI API key (or compatible API)
- Redis (optional, for caching)
- Qdrant (optional, for vector storage)

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
python run.py
```

## 演示效果

### 主界面 - 双模式切换

主界面

### 论文搜索功能

论文搜索

### 深度分析功能

论文分析

## 📊 性能指标

- **论文搜索**：~5秒（ArXiv API 响应时间）
- **PDF 解析**：~3秒/篇（标准学术论文）
- **深度分析**：~20秒/篇（含 AI 推理）
- **写作润色**：~2秒首字生成（流式输出）
- **引用校验**：~3秒/条（含外部 API 验证）
- **完整工作流**：~70秒（搜索3篇+分析+引用+报告）

## 🛣️ 开发路线图

### v1.0（当前版本）✅

- [x] 四大智能体基础功能
- [x] PDF 深度解析（LangChain Document Loaders）
- [x] 双模式工作流
- [x] Web 界面
- [x] API 文档
- [x] LangChain 框架迁移

### v1.1（计划中）

- [ ] 向量数据库集成（LangChain QdrantVectorStore）
- [ ] 用户系统与权限管理
- [ ] 历史记录与收藏功能
- [ ] 批量处理优化
- [ ] LangChain LangSmith 集成（调试与监控）

### v2.0（未来）

- [ ] 双层知识库（L1预置+L2私有）
- [ ] 个性化写作风格学习
- [ ] 多语言支持
- [ ] 移动端适配
- [ ] LangGraph 工作流编排

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://python.langchain.com/) - LLM 应用开发框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [ArXiv API](https://arxiv.org/help/api) - 学术论文数据源
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [InnoCore AI](https://github.com/A-pricity/innocore-ai) - A-pricity开发

---



**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

