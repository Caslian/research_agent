"""
InnoCore AI 向量生成工具 - 基于 LangChain 框架
支持两种 Provider:
  - openai: 远程 OpenAI-compatible API（ModelScope / DeepSeek / OpenAI）
  - local:  本地 sentence-transformers 模型（无需网络，无 API 限流）
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
import numpy as np

# LangChain Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

import hashlib
import json

from core.config import get_config
from core.exceptions import AgentException

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  本地 Embedding 模型（基于 sentence-transformers）
# ══════════════════════════════════════════════════════════════

class LocalEmbeddings(Embeddings):
    """
    本地 Embedding 模型，基于 sentence-transformers。
    实现 LangChain Embeddings 接口，无需外部 API。

    支持所有 HuggingFace 上的 sentence-transformers 兼容模型，
    例如:
      - Qwen/Qwen3-Embedding-0.6B   (~1.2 GB, dim=1024)
      - BAAI/bge-small-zh-v1.5       (~100 MB, dim=512)
      - sentence-transformers/all-MiniLM-L6-v2  (~80 MB, dim=384)
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "本地 Embedding 需要 sentence-transformers 包。"
                "  pip install sentence-transformers"
            )
        logger.info(f"正在加载本地 Embedding 模型: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        # 获取向量维度
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"本地模型加载完成 — 维度: {self._dim}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（同步 — LangChain 基类自动提供异步包装）"""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化，适配 Qdrant COSINE 距离
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询"""
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ══════════════════════════════════════════════════════════════
#  Embedding 服务
# ══════════════════════════════════════════════════════════════

class EmbeddingService:
    """向量生成服务 - LangChain 实现"""

    def __init__(self):
        self.config = get_config()
        self.embeddings: Optional[Embeddings] = None
        self.embedding_model = self.config.vector_db.embedding_model
        self.provider = self.config.vector_db.embedding_provider
        self.cache = {}

    async def initialize(self):
        """初始化向量生成器（根据 EMBEDDING_PROVIDER 选择后端）"""
        try:
            if self.provider == "local":
                await self._init_local()
            else:
                await self._init_openai()
        except Exception as e:
            logger.error(f"向量生成器初始化失败: {str(e)}", exc_info=True)
            raise AgentException(f"向量生成器初始化失败: {str(e)}")

    async def _init_local(self):
        """初始化本地 sentence-transformers 模型"""
        logger.info(f"Embedding 服务: LOCAL 模式")
        logger.info(f"  - 模型: {self.embedding_model}")
        logger.info(f"  - 设备: CPU (如需 GPU 加速请安装 CUDA 版 torch)")

        # 在线程池中加载（避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        self.embeddings = await loop.run_in_executor(
            None, lambda: LocalEmbeddings(model_name=self.embedding_model, device="cpu")
        )
        logger.info("本地 Embedding 服务初始化成功 ✓")

    async def _init_openai(self):
        """初始化远程 OpenAI-compatible API"""
        # 优先使用 Embedding 专用 key，与 Qdrant key 独立
        embedding_api_key = (
            self.config.vector_db.embedding_api_key
            or self.config.llm.api_key
        )
        embedding_base_url = (
            self.config.vector_db.embedding_base_url
            or getattr(self.config.llm, 'base_url', None)
        )

        if not embedding_api_key:
            raise AgentException(
                "未配置 Embedding API key。"
                "请在 .env 文件中设置 EMBEDDING_API_KEY 或 OPENAI_API_KEY，"
                "或设置 EMBEDDING_PROVIDER=local 使用本地模型。"
            )

        api_key_source = (
            "EMBEDDING_API_KEY" if self.config.vector_db.embedding_api_key
            else "OPENAI_API_KEY"
        )
        logger.info(f"Embedding 服务: {self.provider.upper()} 模式")
        logger.info(f"  - 模型: {self.embedding_model}")
        logger.info(f"  - Base URL: {embedding_base_url or '未配置'}")
        logger.info(f"  - API Key 来源: {api_key_source}")
        logger.info(f"  - API Key 前缀: {embedding_api_key[:10] if embedding_api_key else 'None'}...")

        init_kwargs = {
            "model": self.embedding_model,
            "api_key": embedding_api_key,
            # 关键：DashScope 走 OpenAI-compatible 接口，但只接受 str/list[str]，
            # 不接受 LangChain 默认 tiktoken 预编码后的 token id 数组。
            # 关闭长度检查后，LangChain 会把原始字符串直接发出去。
            "check_embedding_ctx_length": False,
        }

        if embedding_base_url:
            init_kwargs["base_url"] = embedding_base_url

        logger.debug(f"OpenAIEmbeddings 初始化参数: init_kwargs={init_kwargs}")
        self.embeddings = OpenAIEmbeddings(**init_kwargs)
        logger.info("Embedding 服务初始化成功")
    
    def _get_fallback_dim(self) -> int:
        """获取 fallback 零向量维度（从已加载的模型动态获取）"""
        if self.embeddings is not None:
            if hasattr(self.embeddings, 'dimension'):
                return self.embeddings.dimension
            if self.provider == "local" and hasattr(self.embeddings, 'model'):
                return self.embeddings.model.get_sentence_embedding_dimension()
        return 1024  # Qwen3-Embedding-0.6B 默认维度

    async def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """生成文本向量"""
        if not text:
            return [0.0] * self._get_fallback_dim()

        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        if not self.embeddings:
            try:
                await self.initialize()
            except Exception as e:
                print(f"嵌入服务初始化失败: {str(e)}")
                return [0.0] * self._get_fallback_dim()
        
        try:
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            # 使用 LangChain Embeddings
            embedding = await self.embeddings.aembed_query(cleaned_text)
            
            # 缓存结果
            if use_cache:
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            raise AgentException(f"向量生成失败: {str(e)}")
    
    async def generate_batch_embeddings(self, texts: List[str], 
                                       batch_size: int = 10) -> List[List[float]]:
        """批量生成向量"""
        if not self.embeddings:
            try:
                await self.initialize()
            except Exception as e:
                print(f"嵌入服务初始化失败: {str(e)}")
                # 返回零向量作为备选
                return [[0.0] * self._get_fallback_dim() for _ in texts]
        
        embeddings = []
        
        # 清理文本
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        try:
            # 使用 LangChain 批量嵌入
            embeddings = await self.embeddings.aembed_documents(cleaned_texts)
            
        except Exception as e:
            # 如果批量失败，逐个生成
            for text in cleaned_texts:
                try:
                    embedding = await self.generate_embedding(text)
                    embeddings.append(embedding)
                except Exception as single_error:
                    print(f"单个向量生成失败: {str(single_error)}")
                    embeddings.append([0.0] * self._get_fallback_dim())
        
        return embeddings
    
    async def generate_paper_embedding(self, paper_info: Dict[str, Any]) -> List[float]:
        """为论文生成综合向量"""
        # 组合论文的关键信息
        title = paper_info.get("title", "")
        abstract = paper_info.get("abstract", "")
        authors = " ".join(paper_info.get("authors", []))
        
        # 构建综合文本
        combined_text = f"{title} {abstract} {authors}"
        
        # 如果有结构化内容，也包含进来
        sections = paper_info.get("sections", {})
        if sections:
            section_text = " ".join(sections.values())
            combined_text += " " + section_text
        
        return await self.generate_embedding(combined_text)
    
    async def generate_section_embeddings(self, sections: Dict[str, str]) -> Dict[str, List[float]]:
        """为各个章节生成向量"""
        section_embeddings = {}
        
        for section_name, section_content in sections.items():
            if section_content.strip():
                try:
                    embedding = await self.generate_embedding(section_content)
                    section_embeddings[section_name] = embedding
                except Exception as e:
                    print(f"章节 {section_name} 向量生成失败: {str(e)}")
                    section_embeddings[section_name] = [0.0] * self._get_fallback_dim()
        
        return section_embeddings
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 截断过长的文本（OpenAI有token限制）
        max_length = 8000  # 保守估计
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            embedding1 = await self.generate_embedding(text1)
            embedding2 = await self.generate_embedding(text2)
            
            return self._cosine_similarity(embedding1, embedding2)
            
        except Exception as e:
            print(f"相似度计算失败: {str(e)}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    async def find_most_similar(self, query_text: str, 
                               candidate_texts: List[str],
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """找到最相似的文本"""
        if not candidate_texts:
            return []
        
        try:
            # 生成查询向量
            query_embedding = await self.generate_embedding(query_text)
            
            # 生成候选文本向量
            candidate_embeddings = await self.generate_batch_embeddings(candidate_texts)
            
            # 计算相似度
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                similarities.append({
                    "text": candidate_texts[i],
                    "similarity": similarity,
                    "index": i
                })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"相似文本查找失败: {str(e)}")
            return []
    
    async def cluster_texts(self, texts: List[str], 
                          num_clusters: int = 3) -> Dict[str, Any]:
        """文本聚类（简化实现）"""
        try:
            # 生成所有文本的向量
            embeddings = await self.generate_batch_embeddings(texts)
            
            # 简单的聚类逻辑（基于相似度阈值）
            clusters = {}
            cluster_id = 0
            used_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in used_indices:
                    continue
                
                # 创建新聚类
                clusters[f"cluster_{cluster_id}"] = {
                    "texts": [texts[i]],
                    "indices": [i],
                    "center": embedding
                }
                used_indices.add(i)
                
                # 查找相似文本加入同一聚类
                for j, other_embedding in enumerate(embeddings):
                    if j in used_indices:
                        continue
                    
                    similarity = self._cosine_similarity(embedding, other_embedding)
                    if similarity > 0.8:  # 相似度阈值
                        clusters[f"cluster_{cluster_id}"]["texts"].append(texts[j])
                        clusters[f"cluster_{cluster_id}"]["indices"].append(j)
                        used_indices.add(j)
                
                cluster_id += 1
            
            return {
                "clusters": clusters,
                "num_clusters": len(clusters),
                "total_texts": len(texts)
            }
            
        except Exception as e:
            print(f"文本聚类失败: {str(e)}")
            return {"clusters": {}, "num_clusters": 0, "total_texts": len(texts)}
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词（基于TF-IDF的简化实现）"""
        try:
            # 分词
            words = text.lower().split()
            
            # 过滤停用词（简化版）
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # 计算词频
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 按频率排序
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # 返回前N个关键词
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except Exception as e:
            print(f"关键词提取失败: {str(e)}")
            return []
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """获取向量生成器信息"""
        return {
            "model": self.embedding_model,
            "cache_size": len(self.cache),
            "vector_dimension": self._get_fallback_dim(),
            "provider": self.provider,
        }
    
    def get_langchain_embeddings(self) -> Embeddings:
        """获取 LangChain Embeddings 对象"""
        return self.embeddings


# 向后兼容的别名
EmbeddingGenerator = EmbeddingService


# 全局嵌入服务实例
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """获取全局嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
