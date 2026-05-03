"""
InnoCore AI 向量存储管理模块 - 基于 LangChain 框架
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.embedding_service.generate_batch_embeddings(texts)
        )
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.embedding_service.generate_embedding(text)
        )
    
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
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key
            )
            
            # 创建集合
            await self._create_collections()
            
            # 设置嵌入服务
            if embedding_service:
                self.embeddings = LangChainEmbeddings(embedding_service)
                
                # 初始化 LangChain 向量存储
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
    
    async def _create_collections(self):
        """创建向量集合"""
        collections = [
            (self.l1_collection, "L1预置库"),
            (self.l2_collection, "L2用户库")
        ]
        
        for collection_name, description in collections:
            try:
                self.client.get_collection(collection_name)
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding维度
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
            # 降级为随机向量
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


# 全局向量存储管理器实例
vector_store_manager = VectorStoreManager()
