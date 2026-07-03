# InnoCore AI 模型配置指南：切换到 ModelScope 免费模型

> 📌 **适用场景**：无 OpenAI 账号 / 想免费体验 / 国内网络环境
> 📌 **目标**：将默认的 OpenAI GPT-3.5 替换为 ModelScope 免费模型
> 📌 **难度**：⭐⭐（简单，主要改配置文件）

---

## 一、现状分析：项目默认使用什么模型？

### 1.1 默认配置

打开 `.env` 文件，当前配置如下：

```env
OPENAI_API_KEY="sk-proj-xxxxxxxx..."   # OpenAI 的 API Key
DATABASE_URL=sqlite:///./innocore.db
SECRET_KEY=your_secret_key_here_change_this_in_production
DEBUG=True
```

**结论：项目默认使用 OpenAI 的 `gpt-3.5-turbo` 模型。**

### 1.2 模型调用链路

```
.env 文件
  └─ OPENAI_API_KEY / LLM_MODEL
       └─ core/config.py (LLMConfig)
            └─ core/llm_adapter.py (HelloAgentsLLM)
                 └─ agents/base.py (think 方法)
                      └─ 四大智能体调用 LLM
```

### 1.3 涉及模型的两个地方


| 用途               | 文件                                  | 默认值                      | 说明            |
| ---------------- | ----------------------------------- | ------------------------ | ------------- |
| **对话/推理**        | `core/config.py` → `LLMConfig`      | `gpt-3.5-turbo`          | 智能体思考、分析、生成报告 |
| **向量 Embedding** | `core/config.py` → `VectorDBConfig` | `text-embedding-3-small` | 论文向量化、相似度检索   |


> ⚠️ **注意**：两个地方都需要修改，否则 Embedding 部分仍会调用 OpenAI。

---

## 二、ModelScope 免费方案说明

### 2.1 为什么选 ModelScope？


| 对比项      | OpenAI     | ModelScope API     |
| -------- | ---------- | ------------------ |
| **费用**   | 按 token 计费 | **免费额度（每天限量）**     |
| **网络**   | 需要科学上网     | 国内直连 ✅             |
| **注册**   | 需要境外手机号    | 国内手机号即可 ✅          |
| **模型质量** | GPT-3.5/4  | Qwen2.5 系列（效果接近）   |
| **接口格式** | OpenAI 标准  | **兼容 OpenAI 格式** ✅ |


### 2.2 ModelScope 推荐免费模型


| 模型名称                           | 参数量 | 特点      | 推荐场景        |
| ------------------------------ | --- | ------- | ----------- |
| `Qwen/Qwen2.5-7B-Instruct`     | 7B  | 均衡，免费   | **首选，日常使用** |
| `Qwen/Qwen2.5-14B-Instruct`    | 14B | 更强，免费   | 复杂分析任务      |
| `Qwen/Qwen2.5-72B-Instruct`    | 72B | 最强，有限免费 | 高质量报告生成     |
| `deepseek-ai/DeepSeek-V2-Chat` | -   | 推理强     | 论文逻辑分析      |
| `ZhipuAI/glm-4-9b-chat`        | 9B  | 中文好     | 中文论文处理      |


**Embedding 模型（替代 text-embedding-3-small）：**


| 模型名称                                           | 维度   | 特点          |
| ---------------------------------------------- | ---- | ----------- |
| `iic/nlp_gte_sentence-embedding_chinese-large` | 1024 | 中文语义好       |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct`          | 1536 | 维度兼容 OpenAI |
| `BAAI/bge-large-zh-v1.5`                       | 1024 | 中文 SOTA     |


---

## 三、快速上手：5 分钟完成切换

### Step 1：注册 ModelScope 账号并获取 API Token

1. 访问 [https://www.modelscope.cn/](https://www.modelscope.cn/)
2. 点击右上角「登录/注册」，使用手机号注册
3. 登录后，点击右上角头像 → **「访问令牌」**
4. 点击「创建新令牌」，复制生成的 Token（格式类似 `ms-xxxxxxxxxxxxxxxx`）

> 💡 **免费额度说明**：ModelScope 对注册用户提供每日免费推理额度，
> 7B/14B 模型通常够日常体验使用。

### Step 2：修改 `.env` 文件

打开 `E:\Program\innocore_AI\.env`，**替换为以下内容**：

```env
# ============================================================
# InnoCore AI Configuration - ModelScope 免费模型版
# ============================================================

# LLM 配置（ModelScope 兼容 OpenAI 接口）
OPENAI_API_KEY=your_modelscope_token_here
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Embedding 配置（使用 ModelScope Embedding 服务）
EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
EMBEDDING_BASE_URL=https://api-inference.modelscope.cn/v1

# 数据库配置（保持不变）
DATABASE_URL=sqlite:///./innocore.db

# 安全配置
SECRET_KEY=your_secret_key_here_change_this_in_production

# 调试模式
DEBUG=True
LOG_LEVEL=INFO
```

> ⚠️ **注意**：将 `your_modelscope_token_here` 替换为你在 Step 1 获取的真实 Token。

### Step 3：修改 `core/config.py`（读取新环境变量）

找到 `__post_init__` 方法，在现有代码基础上**添加以下内容**：

```python
def __post_init__(self):
    """初始化后处理"""
    # 原有代码（保持不变）
    self.llm.api_key = self.llm.api_key or os.getenv("OPENAI_API_KEY")
    self.llm.base_url = self.llm.base_url or os.getenv("OPENAI_BASE_URL")
    
    env_model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
    if env_model:
        self.llm.model_name = env_model
    
    # ✅ 新增：读取 Embedding 配置
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if embedding_model:
        self.vector_db.embedding_model = embedding_model
    
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    if embedding_base_url:
        self.vector_db.embedding_base_url = embedding_base_url  # 新增字段
    
    # 原有代码（保持不变）
    self.database.password = self.database.password or os.getenv("DATABASE_PASSWORD")
    self.redis.password = self.redis.password or os.getenv("REDIS_PASSWORD")
    # ... 其余代码不变
```

同时在 `VectorDBConfig` 数据类中添加 `embedding_base_url` 字段：

```python
@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    db_type: VectorDBType = VectorDBType.QDRANT
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    collection_name_prefix: str = "innocore"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: Optional[str] = None  # ✅ 新增这一行
```

### Step 4：修改 `utils/embedding.py`（使用新的 Embedding 服务）

找到 `initialize` 方法，修改如下：

```python
async def initialize(self):
    """初始化向量生成器"""
    try:
        # ✅ 修改：支持自定义 base_url（用于 ModelScope）
        init_kwargs = {
            "api_key": self.config.llm.api_key,
        }
        
        # 优先使用 Embedding 专用 base_url，否则使用 LLM 的 base_url
        embedding_base_url = getattr(self.config.vector_db, 'embedding_base_url', None)
        if embedding_base_url:
            init_kwargs["base_url"] = embedding_base_url
        elif self.config.llm.base_url:
            init_kwargs["base_url"] = self.config.llm.base_url
        
        self.client = AsyncOpenAI(**init_kwargs)
        
    except Exception as e:
        raise AgentException(f"向量生成器初始化失败: {str(e)}")
```

### Step 5：验证配置是否生效

```bash
# 进入项目目录
cd E:\Program\innocore_AI

# 激活虚拟环境
venv\Scripts\Activate.ps1

# 运行诊断脚本
python diagnose.py

# 或者直接启动，观察日志
python run.py
```

启动成功后，日志中应该看到：

```
INFO: HelloAgent LLM 初始化成功: Qwen/Qwen2.5-7B-Instruct
```

---

## 四、完整修改清单（Checklist）

```
□ Step 1: 注册 ModelScope，获取 API Token
□ Step 2: 修改 .env 文件（替换 API Key、Base URL、模型名）
□ Step 3: 修改 core/config.py（添加 embedding_base_url 字段和读取逻辑）
□ Step 4: 修改 utils/embedding.py（支持自定义 Embedding base_url）
□ Step 5: 重启服务，验证日志
```

---

## 五、各文件修改对照表

### 5.1 `.env` 文件修改对照


| 配置项               | 修改前（OpenAI）                       | 修改后（ModelScope）                          |
| ----------------- | --------------------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`  | `sk-proj-xxxxxxxx`                | `your_modelscope_token`                  |
| `OPENAI_BASE_URL` | *(未设置，默认 OpenAI)*                 | `https://api-inference.modelscope.cn/v1` |
| `LLM_MODEL`       | *(未设置，默认 gpt-3.5-turbo)*          | `Qwen/Qwen2.5-7B-Instruct`               |
| `EMBEDDING_MODEL` | *(未设置，默认 text-embedding-3-small)* | `Alibaba-NLP/gte-Qwen2-1.5B-instruct`    |


### 5.2 `core/config.py` 修改对照


| 位置                 | 修改内容                                               |
| ------------------ | -------------------------------------------------- |
| `VectorDBConfig` 类 | 新增 `embedding_base_url: Optional[str] = None` 字段   |
| `__post_init__` 方法 | 新增读取 `EMBEDDING_MODEL` 和 `EMBEDDING_BASE_URL` 环境变量 |


### 5.3 `utils/embedding.py` 修改对照


| 位置              | 修改内容                                                 |
| --------------- | ---------------------------------------------------- |
| `initialize` 方法 | 支持从 `vector_db.embedding_base_url` 读取 Embedding 服务地址 |


---

## 六、常见问题排查

### Q1：启动报错 `ImportError: hello-agents`

```bash
pip install "hello-agents[all]>=0.2.7"
```

### Q2：报错 `AuthenticationError` 或 `401 Unauthorized`

- 检查 `.env` 中的 Token 是否正确复制（无多余空格）
- 确认 ModelScope Token 已激活（登录后在「访问令牌」页面确认状态）

### Q3：报错 `Model not found` 或 `404`

- 确认模型名称拼写正确，区分大小写
- 推荐使用 `Qwen/Qwen2.5-7B-Instruct`（注意斜杠和大小写）
- 可在 [ModelScope 模型库](https://www.modelscope.cn/models) 搜索确认模型 ID

### Q4：Embedding 维度不匹配报错

如果使用 `BAAI/bge-large-zh-v1.5`（1024 维），需要同步修改向量库维度：

```python
# core/vector_store.py 中，修改 VectorParams 的 size
self.client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1024,   # ← 从 1536 改为 1024
        distance=Distance.COSINE
    )
)
```

推荐使用 `Alibaba-NLP/gte-Qwen2-1.5B-instruct`（1536 维），无需修改向量库。

### Q5：响应速度慢

- ModelScope 免费 API 有并发限制，可适当降低 `MAX_CONCURRENT_TASKS`
- 在 `.env` 中添加：`MAX_CONCURRENT_TASKS=2`

### Q6：免费额度用完了怎么办？

**备选方案（同样国内可用）：**

```env
# 方案 A：阿里云灵积 DashScope（Qwen 官方，更稳定）
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo

# 方案 B：硅基流动 SiliconFlow（多模型，有免费额度）
OPENAI_API_KEY=sk-your-siliconflow-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# 方案 C：智谱 AI（GLM 系列，国内稳定）
OPENAI_API_KEY=your-zhipu-key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-flash  # 免费模型
```

---

## 七、推荐配置方案汇总

### 方案一：ModelScope（免费体验首选）

```env
OPENAI_API_KEY=your_modelscope_token
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
EMBEDDING_BASE_URL=https://api-inference.modelscope.cn/v1
```

**优点**：完全免费，国内直连，模型质量好  
**缺点**：有每日额度限制，并发较低

---

### 方案二：SiliconFlow（免费额度更多）

```env
OPENAI_API_KEY=your_siliconflow_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
```

**注册地址**：[https://siliconflow.cn/](https://siliconflow.cn/)  
**优点**：注册送 14 元额度，支持多种开源模型  
**缺点**：Embedding 维度 1024，需修改向量库配置

---

### 方案三：DashScope（稳定生产推荐）

```env
OPENAI_API_KEY=your_dashscope_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

**注册地址**：[https://dashscope.console.aliyun.com/](https://dashscope.console.aliyun.com/)  
**优点**：阿里云官方，稳定可靠，有免费试用额度  
**缺点**：免费额度用完后需付费

---

## 八、技术原理说明（面试加分）

### 为什么只改 `.env` 就能换模型？

因为项目使用了 **OpenAI 兼容接口**设计：

```
ModelScope / DashScope / SiliconFlow
        ↓
  兼容 OpenAI API 格式
  （/v1/chat/completions）
        ↓
  只需修改 base_url + api_key
        ↓
  HelloAgentsLLM 无感知切换
```

这是一个很好的**开闭原则**实践：对扩展开放（新增 Provider），对修改关闭（不改核心代码）。

### Embedding 为什么也要换？

`utils/embedding.py` 使用 `AsyncOpenAI` 客户端调用 Embedding API。
ModelScope 的 Embedding 服务同样兼容 OpenAI 格式，只需修改 `base_url` 即可复用同一套代码。

---

> 💡 **最终建议**：
>
> - **快速体验**：用 ModelScope 方案，5 分钟搞定
> - **长期使用**：用 DashScope 方案，稳定且中文效果最好
> - **面试展示**：重点讲"兼容 OpenAI 接口的多 Provider 设计"，这是亮点

