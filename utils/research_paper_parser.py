"""
科研论文深度解析器 - 基于 LangChain 框架
用于提取论文结构、关键信息和创新点
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from utils.pdf_parser import PDFParser
from utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """论文元数据"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_date: Optional[str] = None
    venue: Optional[str] = None  # 发表刊物或会议


@dataclass
class PaperSection:
    """论文部分"""
    name: str
    content: str
    start_line: int
    end_line: int
    word_count: int


@dataclass
class ResearchPaper:
    """完整的论文解析结果"""
    metadata: PaperMetadata
    sections: Dict[str, PaperSection]
    full_text: str
    page_count: int
    total_word_count: int
    key_terms: List[Tuple[str, float]]  # (词语, 重要性分数)
    parsing_method: str
    parsing_time: str


class ResearchPaperParser:
    """科研论文解析器"""
    
    # 论文常见段落模式
    SECTION_PATTERNS = {
        "abstract": [
            r"(?:^|\n)(?:Abstract|ABSTRACT|摘要|ABSTRACT\s*[:：])(.*?)(?=\n(?:Introduction|INTRODUCTION|引言|Keyword|KEYWORD|1\.|1\s|Index|INTRODUCTION))",
        ],
        "introduction": [
            r"(?:^|\n)(?:1\.?\s*)?(?:Introduction|INTRODUCTION|引言|前言)(.*?)(?=\n(?:2\.?\s*|Related|METHOD|背景|相关工作))",
        ],
        "related_work": [
            r"(?:^|\n)(?:2\.?\s*)?(?:Related Works?|RELATED WORKS?|Related Literature|背景和相关工作|相关工作)(.*?)(?=\n(?:3\.?\s*|Method|METHODS?))",
        ],
        "method": [
            r"(?:^|\n)(?:(?:2|3)\.?\s*)?(?:Method(?:ology|s)?|METHODS?|方法|技术方案)(.*?)(?=\n(?:(?:3|4)\.?\s*|Experiment|EXPERIMENT|实验))",
        ],
        "experiment": [
            r"(?:^|\n)(?:(?:3|4)\.?\s*)?(?:Experiments?|EXPERIMENTS?|实验|结果)(.*?)(?=\n(?:(?:4|5)\.?\s*|Result|Conclusion|CONCLUSION|讨论))",
        ],
        "result": [
            r"(?:^|\n)(?:(?:4|5)\.?\s*)?(?:Results?|RESULTS?|结果)(.*?)(?=\n(?:(?:5|6)\.?\s*|Conclusion|CONCLUSION|讨论|Analysis))",
        ],
        "conclusion": [
            r"(?:^|\n)(?:(?:5|6|7)\.?\s*)?(?:Conclusion|CONCLUSION|结论|讨论)(.*?)(?=\n(?:References|REFERENCES|参考文献|Appendix|APPENDIX))",
        ],
        "references": [
            r"(?:^|\n)(?:References?|REFERENCES?|参考文献|Bibliography)(.*?)$",
        ]
    }
    
    def __init__(self):
        """初始化论文解析器"""
        self.pdf_parser = PDFParser()
        self.text_processor = TextProcessor()
        self.section_names = list(self.SECTION_PATTERNS.keys())
    
    async def parse_paper(self, file_path: str, extract_keywords: bool = True) -> Optional[ResearchPaper]:
        """
        解析科研论文
        
        Args:
            file_path: PDF 文件路径
            extract_keywords: 是否提取关键词
            
        Returns:
            解析后的论文对象
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"开始深度解析论文: {file_path}")
            
            # 1. 使用 PDF 解析器提取原始内容
            pdf_result = await self.pdf_parser.parse_pdf(file_path)
            
            if not pdf_result.get("success"):
                logger.error(f"PDF 解析失败: {pdf_result.get('error')}")
                return None
            
            full_text = pdf_result.get("full_text", "")
            
            # 2. 提取元数据
            metadata = self._extract_metadata(pdf_result)
            
            # 3. 识别和提取论文各部分
            sections = self._extract_sections(full_text)
            
            # 4. 提取关键词/术语
            key_terms = []
            if extract_keywords:
                key_terms = self._extract_key_terms(full_text, sections)
            
            # 5. 构建完整的论文对象
            end_time = datetime.now()
            parsing_duration = (end_time - start_time).total_seconds()
            
            paper = ResearchPaper(
                metadata=metadata,
                sections=sections,
                full_text=full_text,
                page_count=pdf_result.get("page_count", 0),
                total_word_count=pdf_result.get("word_count", 0),
                key_terms=key_terms[:15],  # 取前15个关键词
                parsing_method="research_paper_parser",
                parsing_time=f"{parsing_duration:.2f}s"
            )
            
            logger.info(f"论文解析完成: {metadata.title}")
            logger.info(f"识别的部分: {', '.join(sections.keys())}")
            logger.info(f"关键词数量: {len(key_terms)}")
            
            return paper
            
        except Exception as e:
            logger.error(f"论文解析异常: {str(e)}", exc_info=True)
            return None
    
    def _extract_metadata(self, pdf_result: Dict[str, Any]) -> PaperMetadata:
        """提取论文元数据"""
        return PaperMetadata(
            title=pdf_result.get("title", "Unknown"),
            authors=pdf_result.get("authors", ["Unknown"]),
            abstract=pdf_result.get("abstract", ""),
            keywords=self._extract_keywords_from_text(pdf_result.get("full_text", "")),
            publication_date=None,
            venue=None
        )
    
    def _extract_sections(self, full_text: str) -> Dict[str, PaperSection]:
        """识别和提取论文的各个部分"""
        sections = {}
        
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                
                if match:
                    try:
                        section_content = match.group(1).strip()
                        
                        # 清理过多的空白
                        section_content = re.sub(r'\s+', ' ', section_content)
                        
                        # 截断过长的内容（防止极端情况）
                        if len(section_content) > 50000:
                            logger.warning(f"部分 {section_name} 过长，已截断")
                            section_content = section_content[:50000]
                        
                        if len(section_content) > 50:  # 只保存有实质内容的部分
                            start_line = full_text[:match.start()].count('\n')
                            end_line = full_text[:match.end()].count('\n')
                            
                            sections[section_name] = PaperSection(
                                name=section_name,
                                content=section_content,
                                start_line=start_line,
                                end_line=end_line,
                                word_count=len(section_content.split())
                            )
                            
                            logger.debug(f"成功提取 {section_name}: {len(section_content)} 字符")
                            break
                    
                    except Exception as e:
                        logger.warning(f"提取 {section_name} 时出错: {str(e)}")
                        continue
        
        # 如果没有提取到任何部分，至少提取摘要和引言
        if not sections:
            logger.warning("未能识别出任何标准部分，将使用完整文本")
            sections["full_content"] = PaperSection(
                name="full_content",
                content=full_text[:10000],  # 取前10000字符
                start_line=0,
                end_line=full_text.count('\n'),
                word_count=len(full_text.split())
            )
        
        return sections
    
    def _extract_key_terms(self, full_text: str, sections: Dict[str, PaperSection]) -> List[Tuple[str, float]]:
        """
        提取论文中的关键术语
        
        Returns:
            列表，包含 (术语, 重要性分数) 元组
        """
        try:
            # 1. 从摘要和方法部分提取候选词
            important_sections = ["abstract", "method", "introduction"]
            key_text = ""
            
            for section_name in important_sections:
                if section_name in sections:
                    key_text += sections[section_name].content + " "
            
            if not key_text:
                key_text = full_text[:5000]  # 如果没有标准部分，用开头5000字
            
            # 2. 分词和过滤
            words = re.findall(r'\b[a-zA-Z]{3,}\b', key_text.lower())
            
            # 3. 计算词频
            from collections import Counter
            word_freq = Counter(words)
            
            # 4. 过滤停用词
            filtered_terms = [
                (word, count) for word, count in word_freq.most_common(50)
                if word not in self.text_processor.stop_words and len(word) > 2
            ]
            
            # 5. 计算重要性分数（基于词频和在不同部分出现）
            key_terms_with_scores = []
            for term, freq in filtered_terms:
                score = freq
                
                # 如果在摘要中出现，增加权重
                if "abstract" in sections and term in sections["abstract"].content.lower():
                    score *= 2.0
                
                # 如果在方法部分出现，增加权重
                if "method" in sections and term in sections["method"].content.lower():
                    score *= 1.5
                
                key_terms_with_scores.append((term, float(score)))
            
            # 按分数排序
            key_terms_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"提取了 {len(key_terms_with_scores)} 个关键术语")
            return key_terms_with_scores
            
        except Exception as e:
            logger.warning(f"提取关键词时出错: {str(e)}")
            return []
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        keywords = []
        
        # 查找明确的关键词段落
        patterns = [
            r"(?:Keywords?|KEYWORDS?|关键词|KEY TERMS?)\s*[:：]?\s*(.+?)(?=\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keyword_text = match.group(1)
                # 按逗号、分号或空格分割
                keywords = re.split(r'[,;、|]', keyword_text)
                keywords = [kw.strip() for kw in keywords if kw.strip()]
                break
        
        return keywords[:10]  # 最多返回10个
    
    def paper_to_dict(self, paper: ResearchPaper) -> Dict[str, Any]:
        """将论文对象转换为字典"""
        return {
            "metadata": {
                "title": paper.metadata.title,
                "authors": paper.metadata.authors,
                "abstract": paper.metadata.abstract,
                "keywords": paper.metadata.keywords,
                "publication_date": paper.metadata.publication_date,
                "venue": paper.metadata.venue
            },
            "sections": {
                name: {
                    "name": section.name,
                    "content": section.content[:1000],  # 限制内容长度
                    "word_count": section.word_count,
                    "start_line": section.start_line,
                    "end_line": section.end_line
                }
                for name, section in paper.sections.items()
            },
            "statistics": {
                "page_count": paper.page_count,
                "total_word_count": paper.total_word_count,
                "sections_count": len(paper.sections),
                "key_terms_count": len(paper.key_terms)
            },
            "key_terms": [{"term": term, "score": score} for term, score in paper.key_terms],
            "parsing_info": {
                "method": paper.parsing_method,
                "parsing_time": paper.parsing_time
            }
        }
    
    def extract_innovation_insights(self, paper: ResearchPaper) -> Dict[str, Any]:
        """
        从论文中提取创新点和洞察
        
        Returns:
            包含创新点、技术亮点等的字典
        """
        insights = {
            "innovation_indicators": [],
            "technical_highlights": [],
            "methodology_novelty": "",
            "experimental_uniqueness": ""
        }
        
        try:
            # 1. 从摘要中提取创新指标
            if "abstract" in paper.sections:
                abstract_text = paper.sections["abstract"].content.lower()
                innovation_keywords = [
                    "novel", "innovative", "new", "propose", "present",
                    "first", "framework", "algorithm", "method", "approach",
                    "新", "提出", "首次", "创新", "改进", "方法", "框架"
                ]
                
                found_keywords = [kw for kw in innovation_keywords if kw in abstract_text]
                insights["innovation_indicators"] = found_keywords
            
            # 2. 从方法部分提取技术亮点
            if "method" in paper.sections:
                method_text = paper.sections["method"].content
                # 提取包含特殊符号或技术术语的句子
                sentences = re.split(r'[。.!?！？]', method_text)
                technical_sentences = [
                    s.strip() for s in sentences 
                    if any(term in s.lower() for term in paper.metadata.keywords) 
                    and len(s) > 20
                ]
                insights["technical_highlights"] = technical_sentences[:5]
            
            # 3. 评估方法的新颖性
            if "method" in paper.sections and "related_work" in paper.sections:
                method_length = paper.sections["method"].word_count
                related_length = paper.sections["related_work"].word_count
                
                # 方法部分的详细程度表明可能的创新
                if method_length > related_length:
                    insights["methodology_novelty"] = "方法论阐述详细，可能具有一定创新性"
                else:
                    insights["methodology_novelty"] = "相关工作阐述较多，建议关注与现有工作的差异"
            
            # 4. 评估实验的独特性
            if "experiment" in paper.sections or "result" in paper.sections:
                exp_section = paper.sections.get("experiment") or paper.sections.get("result")
                if exp_section:
                    exp_text = exp_section.content.lower()
                    exp_keywords = ["evaluation", "benchmark", "comparison", "dataset", "metric",
                                  "评估", "基准", "对比", "数据集", "指标"]
                    found_exp = [kw for kw in exp_keywords if kw in exp_text]
                    if found_exp:
                        insights["experimental_uniqueness"] = f"实验包含: {', '.join(found_exp)}"
            
        except Exception as e:
            logger.warning(f"提取创新洞察时出错: {str(e)}")
        
        return insights
