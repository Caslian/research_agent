"""
PDF 解析工具 - 基于 LangChain 框架
支持从 PDF 文件中提取文本、标题、作者等信息
"""

import logging
from typing import Dict, Any, Optional, List
import re
import asyncio

# LangChain Document Loaders
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF 解析器 - LangChain 实现"""
    
    def __init__(self):
        """初始化 PDF 解析器"""
        self.supported_formats = ['.pdf']
    
    async def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        解析 PDF 文件 - 使用 LangChain Document Loaders
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            包含解析结果的字典
        """
        try:
            logger.info(f"开始解析 PDF: {file_path}")
            
            # 使用 LangChain PDFPlumberLoader
            loader = PDFPlumberLoader(file_path)
            
            # 同步加载文档（LangChain 的 loader 通常是同步的）
            documents: List[Document] = await asyncio.to_thread(loader.load)
            
            if not documents:
                logger.warning("PDF 文件为空或无法提取文本")
                return {
                    "success": False,
                    "error": "无法从 PDF 中提取文本"
                }
            
            # 合并所有页面的文本
            full_text = "\n".join([doc.page_content for doc in documents])
            
            if not full_text.strip():
                logger.warning("PDF 文件为空或无法提取文本")
                return {
                    "success": False,
                    "error": "无法从 PDF 中提取文本"
                }
            
            # 提取元数据
            metadata = documents[0].metadata if documents else {}
            
            # 尝试从文本中提取标题（通常在第一页的前几行）
            title = self._extract_title(full_text, metadata)
            
            # 尝试提取作者
            authors = self._extract_authors(full_text, metadata)
            
            # 尝试提取摘要
            abstract = self._extract_abstract(full_text)
            
            # 统计信息
            page_count = len(documents)
            word_count = len(full_text.split())
            
            result = {
                "success": True,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "full_text": full_text,
                "page_count": page_count,
                "word_count": word_count,
                "metadata": {
                    "creator": metadata.get("Creator", ""),
                    "producer": metadata.get("Producer", ""),
                    "subject": metadata.get("Subject", ""),
                    "keywords": metadata.get("Keywords", ""),
                    "source": metadata.get("source", file_path)
                },
                # LangChain Documents
                "documents": documents
            }
            
            logger.info(f"PDF 解析成功: {page_count} 页, {word_count} 词")
            return result
            
        except ImportError as e:
            logger.error(f"LangChain PDF loader 未安装: {str(e)}")
            return {
                "success": False,
                "error": f"PDF 解析库未安装，请运行: pip install langchain-community pdfplumber"
            }
        except Exception as e:
            logger.error(f"PDF 解析失败: {str(e)}")
            return {
                "success": False,
                "error": f"PDF 解析失败: {str(e)}"
            }
    
    def _extract_title(self, text: str, metadata: Dict) -> str:
        """从文本或元数据中提取标题"""
        # 首先尝试从元数据获取
        if metadata.get("Title"):
            return metadata["Title"]
        
        # 从文本前几行提取（通常标题在最前面且字体较大）
        lines = text.split('\n')
        for i, line in enumerate(lines[:10]):  # 只检查前10行
            line = line.strip()
            # 标题通常较长且不包含特殊字符
            if len(line) > 10 and len(line) < 200 and not line.startswith(('http', 'www', '@')):
                # 排除一些常见的非标题行
                if not any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'page', 'arxiv']):
                    return line
        
        return "未知标题"
    
    def _extract_authors(self, text: str, metadata: Dict) -> list:
        """从文本或元数据中提取作者"""
        authors = []
        
        # 首先尝试从元数据获取
        if metadata.get("Author"):
            author_str = metadata["Author"]
            authors = [a.strip() for a in re.split(r'[,;]', author_str) if a.strip()]
            if authors:
                return authors
        
        # 从文本中提取（通常在标题后面）
        lines = text.split('\n')
        for i, line in enumerate(lines[:20]):  # 检查前20行
            line = line.strip()
            # 查找包含作者信息的行（通常包含邮箱或机构）
            if '@' in line or 'university' in line.lower() or 'institute' in line.lower():
                # 尝试提取前面几行作为作者名
                for j in range(max(0, i-3), i):
                    potential_author = lines[j].strip()
                    if potential_author and len(potential_author) < 100:
                        # 简单的名字模式匹配
                        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', potential_author):
                            authors.append(potential_author)
        
        return authors if authors else ["未知作者"]
    
    def _extract_abstract(self, text: str) -> str:
        """从文本中提取摘要"""
        # 查找 Abstract 关键词
        abstract_patterns = [
            r'Abstract\s*[:\-]?\s*(.*?)(?=\n\n|\nIntroduction|\n1\.|\nKeywords)',
            r'ABSTRACT\s*[:\-]?\s*(.*?)(?=\n\n|\nINTRODUCTION|\n1\.|\nKEYWORDS)',
            r'摘要\s*[:\-]?\s*(.*?)(?=\n\n|关键词|引言|1\.)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # 限制摘要长度
                if len(abstract) > 50 and len(abstract) < 2000:
                    return abstract[:1000]  # 最多返回1000字符
        
        # 如果没找到，返回前500个字符作为摘要
        return text[:500].strip() + "..."
    
    async def parse_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        从字节流解析 PDF
        
        Args:
            pdf_bytes: PDF 文件的字节内容
            filename: 文件名（用于日志）
            
        Returns:
            包含解析结果的字典
        """
        try:
            import io
            import tempfile
            import os
            
            logger.info(f"开始解析 PDF 字节流: {filename}")
            
            # LangChain loader 需要文件路径，创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            try:
                # 使用 LangChain loader 解析
                loader = PDFPlumberLoader(tmp_path)
                documents: List[Document] = await asyncio.to_thread(loader.load)
                
                # 删除临时文件
                os.unlink(tmp_path)
                
                if not documents:
                    return {
                        "success": False,
                        "error": "无法从 PDF 中提取文本"
                    }
                
                # 合并所有页面的文本
                full_text = "\n".join([doc.page_content for doc in documents])
                
                # 提取元数据
                metadata = documents[0].metadata if documents else {}
                
                # 提取信息
                title = self._extract_title(full_text, metadata)
                authors = self._extract_authors(full_text, metadata)
                abstract = self._extract_abstract(full_text)
                
                result = {
                    "success": True,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "full_text": full_text,
                    "page_count": len(documents),
                    "word_count": len(full_text.split()),
                    "metadata": {
                        "creator": metadata.get("/Creator", ""),
                        "producer": metadata.get("/Producer", ""),
                        "subject": metadata.get("/Subject", ""),
                        "keywords": metadata.get("/Keywords", ""),
                        "source": filename
                    },
                    "documents": documents
                }
                
                logger.info(f"PDF 字节流解析成功")
                return result
                
            except Exception as e:
                # 确保删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e
                
        except Exception as e:
            logger.error(f"PDF 字节流解析失败: {str(e)}")
            return {
                "success": False,
                "error": f"PDF 解析失败: {str(e)}"
            }
    
    async def parse_pdf_from_url(self, url: str) -> Dict[str, Any]:
        """
        从 URL 解析 PDF
        
        Args:
            url: PDF 文件的 URL
            
        Returns:
            包含解析结果的字典
        """
        try:
            import aiohttp
            import tempfile
            import os
            
            logger.info(f"开始从 URL 下载 PDF: {url}")
            
            # 下载 PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"下载失败，HTTP 状态码: {response.status}"
                        }
                    
                    pdf_bytes = await response.read()
            
            # 解析 PDF
            filename = url.split('/')[-1] or "document.pdf"
            return await self.parse_pdf_from_bytes(pdf_bytes, filename)
            
        except Exception as e:
            logger.error(f"从 URL 解析 PDF 失败: {str(e)}")
            return {
                "success": False,
                "error": f"PDF 解析失败: {str(e)}"
            }
    
    def get_langchain_documents(self, file_path: str) -> List[Document]:
        """
        获取 LangChain Document 对象列表
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            Document 对象列表
        """
        try:
            loader = PDFPlumberLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"获取 LangChain Documents 失败: {str(e)}")
            return []


# 全局 PDF 解析器实例
pdf_parser = PDFParser()
