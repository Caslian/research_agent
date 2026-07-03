"""
InnoCore AI 向量存储管理模块 - 基于 LangChain 框架
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# LangChain 向量存储组件
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Qdrant 客户端
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.models import CollectionInfo

import hashlib
import json

from .config import get_config
from .exceptions import VectorStoreException


class LangChainEmbeddings(Embeddings):
    """
    LangChain Embeddings 适配器。

    实现要点：
    - 同步接口走 httpx 直接 POST DashScope OpenAI-compatible 接口，
      避免 OpenAIEmbeddings 内置 tiktoken 预分词（DashScope 不接受 token id 数组）。
    - 异步接口复用一个专用后台 event loop 处理 httpx 异步调用，
      避免与外层业务 loop 冲突。
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        # 专用后台 loop，专门跑异步 embedding 请求
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="embedding-async-loop"
        )
        self._thread.start()

    def _post_dashscope(self, texts):
        """同步直连 DashScope OpenAI 兼容接口，绕过 langchain OpenAIEmbeddings。"""
        import httpx
        from core.config import get_config

        cfg = get_config().vector_db
        api_key = cfg.embedding_api_key or cfg.api_key
        base_url = cfg.embedding_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = cfg.embedding_model

        # DashScope 强制走 str 字段；单条也用 list，传字符串有的版本不支持
        payload = {
            "model": model,
            "input": texts if isinstance(texts, list) else [texts],
        }
        # text-embedding-v4 支持指定维度（必须 >= 64 且 <= 2048，且为 8 的倍数）
        if "v4" in (model or "") or "v3" in (model or ""):
            payload["dimensions"] = 1024

        with httpx.Client(timeout=60.0) as cli:
            r = cli.post(
                f"{base_url.rstrip('/')}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        return [item["embedding"] for item in data["data"]]

    def _run_async_in_dedicated_loop(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（同步入口）"""
        if not texts:
            return []
        try:
            # 直接走 httpx 同步请求，最稳
            return self._post_dashscope(texts)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"批量嵌入失败: {str(e)}")
            raise RuntimeError(f"向量生成失败: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询（同步入口）"""
        try:
            return self._post_dashscope([text])[0]
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"查询嵌入失败: {str(e)}")
            raise RuntimeError(f"向量生成失败: {str(e)}") from e

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入"""
        return await self.embedding_service.generate_batch_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询"""
        return await self.embedding_service.generate_embedding(text)


class VectorStoreManager:
    """向量存储管理器 - LangChain 实现"""
    
    def __init__(self):
        self.config = get_config().vector_db
        self.client = None
        self.l1_collection = f"{self.config.collection_name_prefix}_l1_preset"
        self.l2_collection = f"{self.config.collection_name_prefix}_l2_user"
        
        # LangChain 向量存储
        self.l1_vectorstore: Optional[QdrantVectorStore] = None
        self.l2_vectorstore: Optional[QdrantVectorStore] = None
        
        # 嵌入服务
        self.embeddings: Optional[LangChainEmbeddings] = None
    
    async def initialize(self, embedding_service=None):
        """初始化向量数据库连接"""
        try:
            # 获取 Qdrant 连接地址（远程 URL 优先，本地兜底）
            qdrant_url = self.config.get_qdrant_url()
            use_https = qdrant_url.startswith("https://")

            # 初始化 Qdrant 客户端
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=self.config.api_key if self.config.api_key else None,
                prefer_grpc=False,  # 远程建议关闭 gRPC
                https=use_https,
                check_compatibility=False  # 跳过版本检查
            )

            # 设置嵌入服务
            self.embeddings = LangChainEmbeddings(embedding_service) if embedding_service else None
            # ═══ 关键修复 ═══
            # 优先通过小模型做一次"维度探测"，这样即便 Qdrant 上已经有历史 collection 残存，
            # 也能立刻知道当前 embedding 的真实维度，进而决定是复用还是重建 collection。
            # 旧问题：探测失败 fallback 1536，导致后续 collection 维度与 embeddings 不匹配而报错。
            embedding_dimension = (
                await self._get_embedding_dimension()
                if embedding_service
                else self._infer_default_dim()
            )

            # 创建集合（使用实际的 embedding 维度）
            await self._create_collections(embedding_dimension)

            # 初始化 LangChain 向量存储
            if embedding_service:
                self._init_langchain_vectorstores()

        except Exception as e:
            raise VectorStoreException(f"向量数据库初始化失败: {str(e)}")
    
    def _init_langchain_vectorstores(self):
        """初始化 LangChain 向量存储"""
        if not self.embeddings:
            return

        # 不同版本的 langchain_qdrant 签名不一致，需要做版本兼容处理
        # 老版本：只接受 client/collection_name/embedding
        # 新版本（≥0.1）：支持 validate_collection_config / retrieval_mode 等
        try:
            import langchain_qdrant
            from importlib.metadata import version as _v, PackageNotFoundError
            try:
                _pkg_version = _v("langchain-qdrant")
            except PackageNotFoundError:
                _pkg_version = "0.0.0"
        except Exception:
            _pkg_version = "0.0.0"

        base_kwargs = {
            "client": self.client,
            "collection_name": None,  # 占位，下面会覆盖
            "embedding": self.embeddings,
        }

        # 检查是否支持 validate_collection_config
        try:
            import inspect
            from langchain_qdrant import QdrantVectorStore
            sig_params = inspect.signature(QdrantVectorStore.__init__).parameters
            supports_validate = "validate_collection_config" in sig_params
        except Exception:
            supports_validate = False

        try:
            l1_kwargs = dict(base_kwargs, collection_name=self.l1_collection)
            l2_kwargs = dict(base_kwargs, collection_name=self.l2_collection)
            if supports_validate:
                l1_kwargs["validate_collection_config"] = False
                l2_kwargs["validate_collection_config"] = False

            # L1 预置库向量存储
            self.l1_vectorstore = QdrantVectorStore(**l1_kwargs)

            # L2 用户库向量存储
            self.l2_vectorstore = QdrantVectorStore(**l2_kwargs)
        except TypeError as e:
            # 兼容老版本：去掉可选参数再试
            if "unexpected keyword" in str(e):
                self.l1_vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.l1_collection,
                    embedding=self.embeddings,
                )
                self.l2_vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.l2_collection,
                    embedding=self.embeddings,
                )
            else:
                raise VectorStoreException(f"LangChain 向量存储初始化失败: {str(e)}")
        except Exception as e:
            raise VectorStoreException(f"LangChain 向量存储初始化失败: {str(e)}")
    
    async def _get_embedding_dimension(self) -> int:
        """获取 embedding 的实际维度。

        探测策略：
        1) provider 是 dashscope 时跳过实际请求（兼容层对 OpenAIEmbeddings 参数格式支持不一致），
           直接用模型名推断维度，避免发出无效请求；
        2) 否则尝试一次 aembed_query，失败再退到 sync embed_query；
        3) 最终回退到根据 provider/model 推断的默认维度。
        """
        if not self.embeddings:
            return self._infer_default_dim()

        provider = (self.config.embedding_provider or "").lower()
        model = (self.config.embedding_model or "").lower()

        # DashScope 对 OpenAIEmbeddings 客户端的 "contents" 字段格式要求与 OpenAI 不一致，
        # 实际探测会得到 400。直接走模型名推断既准确又无副作用。
        if provider == "dashscope" or "dashscope" in model:
            logger.info(
                f"DashScope provider，跳过实际探测，使用推断维度: {self._infer_default_dim()}"
            )
            return self._infer_default_dim()

        # 1) async embed_query
        try:
            test_embedding = await self.embeddings.aembed_query("dimension probe")
            if test_embedding:
                return len(test_embedding)
        except Exception as e:
            logger.debug(f"aembed_query 探测失败: {e}")

        # 2) sync embed_query
        try:
            import asyncio
            test_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, "dimension probe"
            )
            if test_embedding:
                return len(test_embedding)
        except Exception as e:
            logger.debug(f"sync embed_query 探测失败: {e}")

        logger.warning(
            f"所有维度探测策略均失败，使用 provider 默认维度: {self._infer_default_dim()}"
        )
        return self._infer_default_dim()

    def _infer_default_dim(self) -> int:
        """根据 embedding provider / 模型名推断默认维度"""
        provider = (self.config.embedding_provider or "").lower()
        model = (self.config.embedding_model or "").lower()
        if provider == "local":
            return 1024  # Qwen3-Embedding-0.6B 默认 1024
        if "text-embedding-v3" in model or "text-embedding-v4" in model:
            return 1024
        if "text-embedding-3-small" in model:
            return 1536
        if "text-embedding-3-large" in model:
            return 3072
        if "text-embedding-ada-002" in model:
            return 1536
        # 默认给 DashScope 通用模型 1024（已是大趋势）
        if provider == "dashscope" or "dashscope" in model:
            return 1024
        return 1024  # 现代 embedding 模型普遍 1024
    
    async def _create_collections(self, embedding_dimension: int = None):
        """创建向量集合"""
        if embedding_dimension is None:
            embedding_dimension = self._infer_default_dim()

        collections = [
            (self.l1_collection, "L1预置库"),
            (self.l2_collection, "L2用户库")
        ]

        # 每个集合需要的 payload index（filter 字段）
        # 注意：LangChain QdrantVectorStore 把 Document.metadata 嵌套存到 payload["metadata"]，
        # 所以 filter 字段路径必须写成 "metadata.user_id" 这种嵌套形式。
        indexes_per_collection = {
            self.l1_collection: ["paper_id", "source", "metadata.paper_id", "metadata.source"],
            self.l2_collection: [
                "paper_id", "user_id", "source",
                "metadata.paper_id", "metadata.user_id", "metadata.source",
            ],
        }

        for collection_name, description in collections:
            created_now = False
            try:
                # 检查集合是否已存在
                existing_collection = self.client.get_collection(collection_name)

                # 检查维度是否匹配
                existing_dim = existing_collection.config.params.vectors.size
                if existing_dim != embedding_dimension:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"集合 {collection_name} 维度不匹配 "
                        f"(现有: {existing_dim}, 新: {embedding_dimension})，"
                        f"将删除并重新创建"
                    )
                    # 删除不匹配的集合
                    self.client.delete_collection(collection_name)
                    # 创建新集合
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    created_now = True
                    logger.info(
                        f"集合 {collection_name} 已重新创建 (dim={embedding_dimension})"
                    )
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"集合 {collection_name} 维度匹配 (dim={existing_dim})，使用现有集合")
            except Exception as get_err:
                # 区分"集合不存在"与"其他错误"——避免对非 404 错误也盲目 create
                err_msg = str(get_err).lower()
                not_found = (
                    "not found" in err_msg
                    or "404" in err_msg
                    or "doesn't exist" in err_msg
                )
                if not_found:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"集合 {collection_name} 不存在，准备创建 (dim={embedding_dimension})"
                    )
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"获取集合 {collection_name} 失败，将尝试强制重建: {get_err}"
                    )
                    try:
                        self.client.delete_collection(collection_name)
                    except Exception:
                        pass
                # 创建新集合
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                created_now = True
                logger.info(f"集合 {collection_name} 创建成功 (dim={embedding_dimension})")

            # 给 filter 字段建 payload index（keyword 类型）
            # 新建 collection 时必须建，否则后续按 user_id/paper_id 过滤会 400
            for field in indexes_per_collection.get(collection_name, []):
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema="keyword",
                    )
                except Exception as idx_err:
                    # 已存在时报错是正常的
                    if "already exists" not in str(idx_err).lower():
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"为 {collection_name}.{field} 建索引失败: {idx_err}"
                        )
    
    def _generate_point_id(self, content: str) -> str:
        """生成向量点ID"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def add_to_l1(self, paper_id: str, title: str, abstract: str, 
                       content: str, metadata: Dict = None) -> str:
        """添加到L1预置库 - 使用 LangChain"""
        try:
            if self.l1_vectorstore:
                # 使用 LangChain 添加文档
                doc = Document(
                    page_content=f"{title} {abstract} {content}",
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "abstract": abstract,
                        "collection_type": "l1",
                        **(metadata or {})
                    }
                )
                
                ids = await asyncio.to_thread(
                    self.l1_vectorstore.add_documents,
                    [doc]
                )
                
                return ids[0] if ids else ""
            else:
                # 降级为直接操作
                return await self._add_to_collection_direct(
                    self.l1_collection, paper_id, title, abstract, content, metadata
                )
            
        except Exception as e:
            raise VectorStoreException(f"添加到L1库失败: {str(e)}")
    
    async def add_to_l2(self, user_id: str, paper_id: str, title: str, 
                       abstract: str, content: str, metadata: Dict = None) -> str:
        """添加到L2用户库 - 使用 LangChain"""
        try:
            if self.l2_vectorstore:
                # 使用 LangChain 添加文档
                doc = Document(
                    page_content=f"{title} {abstract} {content}",
                    metadata={
                        "user_id": user_id,
                        "paper_id": paper_id,
                        "title": title,
                        "abstract": abstract,
                        "collection_type": "l2",
                        **(metadata or {})
                    }
                )
                
                ids = await asyncio.to_thread(
                    self.l2_vectorstore.add_documents,
                    [doc]
                )
                
                return ids[0] if ids else ""
            else:
                # 降级为直接操作
                return await self._add_to_collection_direct(
                    self.l2_collection, paper_id, title, abstract, content, 
                    {**{"user_id": user_id}, **(metadata or {})}
                )
            
        except Exception as e:
            raise VectorStoreException(f"添加到L2库失败: {str(e)}")
    
    async def _add_to_collection_direct(self, collection_name: str, 
                                        paper_id: str, title: str, 
                                        abstract: str, content: str, 
                                        metadata: Dict = None) -> str:
        """直接添加到集合（降级方案）"""
        try:
            # 生成embedding
            embedding = await self._generate_embedding(f"{title} {abstract} {content}")
            
            point_id = self._generate_point_id(f"{paper_id}_{collection_name}")
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "content": content[:1000],
                    "metadata": metadata or {},
                    "collection_type": "l1" if "l1" in collection_name else "l2",
                    "created_at": str(asyncio.get_event_loop().time())
                }
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            return point_id
            
        except Exception as e:
            raise VectorStoreException(f"直接添加失败: {str(e)}")
    
    async def hybrid_search(self, query: str, user_id: str = None, 
                           top_k: int = 5, include_l1: bool = True,
                           include_l2: bool = True) -> List[Dict]:
        """混合搜索 - 使用 LangChain 相似度搜索"""
        try:
            results = []
            
            config = get_config()
            vector_weight = config.hybrid_search_weights.get("vector", 0.7)
            keyword_weight = config.hybrid_search_weights.get("keyword", 0.3)
            
            # L1库搜索
            if include_l1 and self.l1_vectorstore:
                l1_docs = await asyncio.to_thread(
                    self.l1_vectorstore.similarity_search_with_score,
                    query, top_k
                )
                
                for doc, score in l1_docs:
                    results.append({
                        "id": doc.metadata.get("paper_id", ""),
                        "score": score * vector_weight,
                        "payload": {
                            "paper_id": doc.metadata.get("paper_id", ""),
                            "title": doc.metadata.get("title", ""),
                            "abstract": doc.metadata.get("abstract", ""),
                            **doc.metadata
                        },
                        "collection_type": "l1"
                    })
            
            # L2库搜索
            if include_l2 and user_id and self.l2_vectorstore:
                # 使用 filter 进行用户过滤
                # 注意：LangChain QdrantVectorStore 把 Document.metadata 嵌套存在
                # payload["metadata"]，所以 filter 路径是 "metadata.user_id"
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                l2_docs = await asyncio.to_thread(
                    self.l2_vectorstore.similarity_search_with_score,
                    query, top_k,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    )
                )
                
                for doc, score in l2_docs:
                    results.append({
                        "id": doc.metadata.get("paper_id", ""),
                        "score": score * vector_weight,
                        "payload": {
                            "paper_id": doc.metadata.get("paper_id", ""),
                            "title": doc.metadata.get("title", ""),
                            "abstract": doc.metadata.get("abstract", ""),
                            "user_id": doc.metadata.get("user_id", ""),
                            **doc.metadata
                        },
                        "collection_type": "l2"
                    })
            
            # 关键词匹配加分
            for result in results:
                payload = result["payload"]
                keyword_score = self._calculate_keyword_score(
                    query, 
                    f"{payload.get('title', '')} {payload.get('abstract', '')}"
                )
                result["score"] += keyword_score * keyword_weight
            
            # 按分数排序并返回top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            raise VectorStoreException(f"混合搜索失败: {str(e)}")
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """计算关键词匹配分数"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本向量"""
        if self.embeddings:
            return await self.embeddings.aembed_query(text)
        else:
            # 警告：降级为随机向量（embedding_service 未初始化）
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Embedding 服务未初始化，使用随机向量替代。"
                "请确保调用 vector_store_manager.initialize(embedding_service=...) 时传入了 embedding_service 参数。"
            )
            import random
            return [random.random() for _ in range(1536)]
    
    async def get_user_vectors(self, user_id: str, limit: int = 100) -> List[Dict]:
        """获取用户的向量数据"""
        try:
            user_filter = Filter(
                must=[
                    FieldCondition(
                        # LangChain QdrantVectorStore 把 metadata 嵌套在 payload["metadata"] 下
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            results = self.client.scroll(
                collection_name=self.l2_collection,
                scroll_filter=user_filter,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload
                }
                for point in results[0]
            ]
            
        except Exception as e:
            raise VectorStoreException(f"获取用户向量失败: {str(e)}")
    
    async def delete_user_vectors(self, user_id: str) -> bool:
        """删除用户的所有向量数据"""
        try:
            user_filter = Filter(
                must=[
                    FieldCondition(
                        # LangChain QdrantVectorStore 把 metadata 嵌套在 payload["metadata"] 下
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.l2_collection,
                points_selector=user_filter
            )
            
            return True
            
        except Exception as e:
            raise VectorStoreException(f"删除用户向量失败: {str(e)}")
    
    async def get_collection_info(self, collection_type: str = "l1") -> CollectionInfo:
        """获取集合信息"""
        collection_name = self.l1_collection if collection_type == "l1" else self.l2_collection
        return self.client.get_collection(collection_name)
    
    def get_retriever(self, collection_type: str = "l1", search_kwargs: Dict = None):
        """获取 LangChain Retriever"""
        vectorstore = self.l1_vectorstore if collection_type == "l1" else self.l2_vectorstore
        
        if not vectorstore:
            raise VectorStoreException(f"{collection_type} 向量存储未初始化")
        
        search_kwargs = search_kwargs or {"k": 5}
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    async def close(self):
        """关闭向量数据库连接"""
        if self.client:
            self.client.close()
    
    def is_embedding_initialized(self) -> bool:
        """检查 embedding 服务是否已初始化"""
        return self.embeddings is not None
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """获取初始化状态诊断信息"""
        return {
            "qdrant_url": self.config.get_qdrant_url(),
            "qdrant_client_ready": self.client is not None,
            "l1_vectorstore_ready": self.l1_vectorstore is not None,
            "l2_vectorstore_ready": self.l2_vectorstore is not None,
            "embedding_service_ready": self.embeddings is not None,
            "embedding_service_type": type(self.embeddings).__name__ if self.embeddings else "None"
        }


# 全局向量存储管理器实例
vector_store_manager = VectorStoreManager()
