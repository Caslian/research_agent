"""
InnoCore AI 向量存储管理模块 - 基于 LangChain 框架
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

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
    """LangChain Embeddings 适配器"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        # 创建线程池用于运行异步操作
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding_")
    
    def _run_async_in_thread(self, coro):
        """在新线程中运行异步代码"""
        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        # 在线程池中运行
        future = self._executor.submit(run_in_new_loop)
        return future.result(timeout=120)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        try:
            # 检查是否有事件循环在运行
            try:
                loop = asyncio.get_running_loop()
                # 有事件循环在运行，使用线程池运行异步代码
                return self._run_async_in_thread(
                    self.embedding_service.generate_batch_embeddings(texts)
                )
            except RuntimeError:
                # 没有事件循环在运行，直接运行
                return asyncio.run(
                    self.embedding_service.generate_batch_embeddings(texts)
                )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"批量嵌入失败: {str(e)}")
            raise RuntimeError(f"向量生成失败: {str(e)}") from e
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        try:
            # 检查是否有事件循环在运行
            try:
                loop = asyncio.get_running_loop()
                # 有事件循环在运行，使用线程池运行异步代码
                return self._run_async_in_thread(
                    self.embedding_service.generate_embedding(text)
                )
            except RuntimeError:
                # 没有事件循环在运行，直接运行
                return asyncio.run(
                    self.embedding_service.generate_embedding(text)
                )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"查询嵌入失败: {str(e)}")
            raise RuntimeError(f"向量生成失败: {str(e)}") from e
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入文档"""
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
            # 初始化 Qdrant 客户端
            # 禁用 HTTPS 和版本检查以支持本地 HTTP 连接
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                prefer_grpc=False,  # 使用 HTTP REST API 而不是 gRPC
                https=False,  # 明确指定不使用 HTTPS
                check_compatibility=False  # 跳过版本检查
            )
            
            # 设置嵌入服务
            embedding_dimension = None
            if embedding_service:
                self.embeddings = LangChainEmbeddings(embedding_service)
                # 获取实际的 embedding 维度
                embedding_dimension = await self._get_embedding_dimension()
            
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
        
        try:
            # L1 预置库向量存储
            self.l1_vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.l1_collection,
                embedding=self.embeddings,
            )
            
            # L2 用户库向量存储
            self.l2_vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.l2_collection,
                embedding=self.embeddings,
            )
        except Exception as e:
            raise VectorStoreException(f"LangChain 向量存储初始化失败: {str(e)}")
    
    async def _get_embedding_dimension(self) -> int:
        """获取 embedding 的实际维度"""
        if not self.embeddings:
            return 1536  # 默认返回 OpenAI 维度
        
        try:
            # 生成一个测试 embedding 来获取维度
            test_embedding = await self.embeddings.aembed_query("test")
            return len(test_embedding)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"无法获取 embedding 维度，使用默认值 1536: {str(e)}")
            return 1536
    
    async def _create_collections(self, embedding_dimension: int = None):
        """创建向量集合"""
        if embedding_dimension is None:
            embedding_dimension = 1536  # 默认 OpenAI 维度
        
        collections = [
            (self.l1_collection, "L1预置库"),
            (self.l2_collection, "L2用户库")
        ]
        
        for collection_name, description in collections:
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
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"集合 {collection_name} 维度匹配，使用现有集合")
            except Exception as e:
                # 集合不存在，创建新集合
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dimension,
                        distance=Distance.COSINE
                    )
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
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                l2_docs = await asyncio.to_thread(
                    self.l2_vectorstore.similarity_search_with_score,
                    query, top_k,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
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
                        key="user_id",
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
                        key="user_id",
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
            "qdrant_client_ready": self.client is not None,
            "l1_vectorstore_ready": self.l1_vectorstore is not None,
            "l2_vectorstore_ready": self.l2_vectorstore is not None,
            "embedding_service_ready": self.embeddings is not None,
            "embedding_service_type": type(self.embeddings).__name__ if self.embeddings else "None"
        }


# 全局向量存储管理器实例
vector_store_manager = VectorStoreManager()
