"""
InnoCore AI 洞察专家 (Miner Agent) - 基于 LangChain 框架
核心大脑。负责阅读、理解、检索历史库、对比分析并生成报告
"""

import asyncio
from typing import Dict, List, Optional, Any
import json
import re
from datetime import datetime

from agents.base import BaseAgent
from core.database import db_manager
from core.vector_store import vector_store_manager
from core.exceptions import AgentException
from utils.research_paper_parser import ResearchPaperParser, ResearchPaper

class MinerAgent(BaseAgent):
    """洞察专家智能体"""
    
    def __init__(self, llm=None):
        super().__init__("Miner", llm)
        
        # 初始化论文解析器
        self.paper_parser = ResearchPaperParser()
        
        # 添加工具
        self.add_tool("parse_pdf", self._parse_pdf, "解析PDF文件")
        self.add_tool("search_memory", self._search_memory, "搜索记忆库")
        self.add_tool("compare_papers", self._compare_papers, "对比论文")
        self.add_tool("generate_report", self._generate_report, "生成分析报告")
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行论文分析和创新点挖掘任务"""
        await self.validate_input(input_data)
        
        self.set_state("running")
        
        try:
            paper_id = input_data["paper_id"]
            user_id = input_data.get("user_id")
            analysis_type = input_data.get("analysis_type", "full")  # full, quick, innovation_only
            
            # 获取论文信息
            paper = await db_manager.get_paper(paper_id)
            if not paper:
                raise AgentException(f"论文不存在: {paper_id}")
            
            self._add_to_history(f"开始分析论文: {paper['title']}")
            
            # 1. 解析PDF内容
            parsed_content = await self._parse_paper_content(paper)
            
            # 2. 检索相关历史论文
            related_papers = await self._find_related_papers(
                paper["title"], 
                paper["abstract"], 
                user_id
            )
            
            # 3. 进行对比分析
            comparison_result = await self._perform_comparison_analysis(
                parsed_content, 
                related_papers
            )
            
            # 4. 生成分析报告
            report = await self._create_analysis_report(
                paper, 
                parsed_content, 
                related_papers, 
                comparison_result,
                user_id
            )
            
            # 5. 保存报告到数据库
            report_id = await self._save_analysis_report(paper_id, report, user_id)
            
            # 6. 更新向量库
            await self._update_vector_store(paper_id, paper, parsed_content, user_id)
            
            self.set_state("completed")
            
            return {
                "status": "success",
                "paper_id": paper_id,
                "report_id": report_id,
                "analysis_type": analysis_type,
                "parsed_content": {
                    "sections": list(parsed_content.get("sections", {}).keys()),
                    "word_count": parsed_content.get("word_count", 0)
                },
                "related_papers_count": len(related_papers),
                "report_summary": {
                    "summary": report.get("summary", "")[:200] + "...",
                    "innovation_points": len(report.get("innovation_points", [])),
                    "limitations": len(report.get("limitations", [])),
                    "future_ideas": len(report.get("future_ideas", []))
                }
            }
            
        except Exception as e:
            self.set_state("error")
            raise AgentException(f"Miner Agent执行失败: {str(e)}")
    
    def get_required_fields(self) -> List[str]:
        """获取必需的输入字段"""
        return ["paper_id"]
    
    async def _parse_paper_content(self, paper: Dict) -> Dict[str, Any]:
        """解析论文内容"""
        file_path = paper.get("file_path")
        
        # 如果有PDF文件，使用深度解析器
        if file_path:
            parsed_paper = await self.paper_parser.parse_paper(file_path, extract_keywords=True)
            if parsed_paper:
                return await self._convert_paper_to_dict(parsed_paper)
        
        # 如果没有PDF文件，使用基础元数据
        self._add_to_history(f"使用元数据进行解析: {paper.get('title', '')}")
        return {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", []),
            "sections": {
                "abstract": paper.get("abstract", ""),
                "introduction": "",
                "method": "",
                "experiment": "",
                "conclusion": ""
            },
            "key_terms": [],
            "word_count": len(paper.get("abstract", "").split()),
            "parsing_method": "metadata_only",
            "parsing_time": "0.00s"
        }
    
    async def _extract_structured_content(self, file_path: str) -> Dict[str, Any]:
        """提取结构化内容 - 使用深度解析器"""
        try:
            # 使用 ResearchPaperParser 进行深度解析
            parsed_paper = await self.paper_parser.parse_paper(file_path, extract_keywords=True)
            
            if not parsed_paper:
                self._add_to_history(f"PDF解析失败: {file_path}")
                return {
                    "title": "",
                    "abstract": "",
                    "sections": {},
                    "word_count": 0,
                    "parsing_method": "failed",
                    "error": "无法解析PDF文件"
                }
            
            result = await self._convert_paper_to_dict(parsed_paper)
            self._add_to_history(f"PDF深度解析完成: {parsed_paper.metadata.title}")
            return result
            
        except Exception as e:
            self._add_to_history(f"PDF解析异常: {str(e)}")
            return {
                "title": "",
                "abstract": "",
                "sections": {},
                "word_count": 0,
                "parsing_method": "failed",
                "error": str(e)
            }
    
    async def _find_related_papers(self, title: str, abstract: str, user_id: str = None) -> List[Dict]:
        """查找相关论文"""
        try:
            # 构建查询
            query = f"{title} {abstract}"
            
            # 执行混合搜索
            search_results = await vector_store_manager.hybrid_search(
                query=query,
                user_id=user_id,
                top_k=10,
                include_l1=True,
                include_l2=bool(user_id)
            )
            
            # 获取详细论文信息
            related_papers = []
            for result in search_results:
                payload = result["payload"]
                paper_id = payload.get("paper_id")
                
                if paper_id:
                    paper_info = await db_manager.get_paper(paper_id)
                    if paper_info:
                        paper_info["similarity_score"] = result["score"]
                        paper_info["collection_type"] = result["collection_type"]
                        related_papers.append(paper_info)
            
            self._add_to_history(f"找到 {len(related_papers)} 篇相关论文")
            return related_papers
            
        except Exception as e:
            self._add_to_history(f"搜索相关论文失败: {str(e)}")
            return []
    
    async def _perform_comparison_analysis(self, current_paper: Dict, related_papers: List[Dict]) -> Dict[str, Any]:
        """执行对比分析 - 利用深度解析结果"""
        if not related_papers:
            return {
                "comparison_summary": "未找到相关论文进行对比",
                "unique_contributions": [],
                "similar_works": [],
                "gaps_identified": [],
                "method_comparison": ""
            }
        
        # 提取关键技术术语用于对比
        current_key_terms = current_paper.get("key_terms", [])
        current_sections = current_paper.get("sections", {})
        current_insights = current_paper.get("insights", {})
        
        # 构建高度结构化的对比分析prompt
        comparison_prompt = f"""
        基于以下详细分析结果，进行深度对比分析：
        
        当前论文：
        标题：{current_paper.get('title', '')}
        作者：{', '.join(current_paper.get('authors', []))}
        摘要：{current_paper.get('abstract', '')}
        关键词：{', '.join([t[0] for t in current_key_terms[:5]])}
        
        创新指标：{current_insights.get('innovation_indicators', [])}
        方法论新颖性：{current_insights.get('methodology_novelty', '')}
        
        论文结构：
        {self._format_sections_for_comparison(current_sections)}
        
        相关论文对比：
        {self._format_related_papers_for_comparison(related_papers[:5])}
        
        请从以下维度进行深度对比分析：
        1. **方法创新性**：与相关论文相比的方法改进
        2. **实验设计**：实验的新颖性和完整性评估
        3. **技术突破**：关键技术术语和创新点对比
        4. **研究空白**：当前文献中未覆盖的领域
        5. **应用价值**：相比现有工作的实际应用潜力
        
        返回JSON格式的分析结果，包含：
        - comparison_summary：总体对比总结
        - unique_contributions：独特贡献列表
        - similar_works：相似工作列表
        - gaps_identified：发现的研究空白
        - method_comparison：方法对比详情
        - future_potential：未来研究潜力评估
        """
        
        try:
            response = await self.think(comparison_prompt)
            
            # 尝试解析JSON响应
            try:
                comparison_result = json.loads(response)
            except json.JSONDecodeError:
                # 如果JSON解析失败，使用文本解析
                comparison_result = self._parse_text_comparison(response)
            
            self._add_to_history("深度对比分析完成")
            return comparison_result
            
        except Exception as e:
            self._add_to_history(f"对比分析失败: {str(e)}")
            return {
                "comparison_summary": "对比分析过程中出现错误",
                "unique_contributions": [],
                "similar_works": [],
                "gaps_identified": [],
                "method_comparison": str(e)
            }
    
    def _format_sections_for_comparison(self, sections: Dict) -> str:
        """格式化论文部分用于对比"""
        formatted = []
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict):
                content_preview = section_data.get("content", "")[:200]
                word_count = section_data.get("word_count", 0)
                formatted.append(f"- {section_name}: {word_count}字，内容摘要：{content_preview}...")
        return "\n".join(formatted) if formatted else "无详细部分信息"
    
    def _format_related_papers_for_comparison(self, papers: List[Dict]) -> str:
        """格式化相关论文用于对比"""
        formatted = []
        for i, paper in enumerate(papers, 1):
            formatted.append(f"""
            论文{i}：
            标题：{paper.get('title', '')}
            摘要：{paper.get('abstract', '')[:300]}...
            相似度：{paper.get('similarity_score', 0):.3f}
            """)
        return "\n".join(formatted)
    
    def _parse_text_comparison(self, text: str) -> Dict[str, Any]:
        """解析文本格式的对比结果"""
        # 简单的文本解析逻辑
        return {
            "comparison_summary": text[:500],
            "unique_contributions": ["基于文本分析的创新点"],
            "similar_works": ["相关研究工作"],
            "gaps_identified": ["研究空白识别"]
        }
    
    async def _create_analysis_report(self, paper: Dict, parsed_content: Dict, 
                                    related_papers: List[Dict], comparison_result: Dict,
                                    user_id: str = None) -> Dict[str, Any]:
        """创建分析报告 - 利用深度解析结果"""
        
        # 提取关键信息
        sections_info = parsed_content.get("sections", {})
        key_terms = parsed_content.get("key_terms", [])
        insights = parsed_content.get("insights", {})
        parsing_time = parsed_content.get("parsing_time", "0.00s")
        
        # 构建高质量的分析report prompt
        report_prompt = f"""
        基于以下深度论文分析结果，生成一份高质量的学术分析报告：
        
        **论文基本信息**
        标题：{paper.get('title', '')}
        作者：{', '.join(paper.get('authors', []))}
        关键词：{', '.join(paper.get('keywords', []))}
        
        **论文摘要**
        {paper.get('abstract', '')}
        
        **提取的关键术语（按重要性排序）**
        {self._format_key_terms(key_terms[:10])}
        
        **论文结构分析**
        {self._format_sections_info(sections_info)}
        
        **创新指标**
        {json.dumps(insights, ensure_ascii=False, indent=2)}
        
        **对比分析结果**
        {json.dumps(comparison_result, ensure_ascii=False, indent=2)}
        
        **相关研究数量**：{len(related_papers)} 篇
        
        请生成包含以下部分的详细分析报告（JSON格式）：
        
        1. **Summary** - 论文主要贡献、研究方法和关键发现的全面概述
        2. **Innovation** - 相比相关论文的具体创新点和技术突破
        3. **Technical Novelty** - 技术方法的创新程度评估
        4. **Experimental Validation** - 实验设计的完整性和有效性评估
        5. **Limitation** - 当前研究存在的局限性和改进空间
        6. **Future Ideas** - 基于分析建议的未来研究方向和扩展工作
        7. **Impact Potential** - 该研究的潜在应用价值和影响力
        
        返回JSON格式，每个字段是字符串或列表。
        """
        
        try:
            response = await self.think(report_prompt)
            
            # 尝试解析JSON响应
            try:
                report = json.loads(response)
            except json.JSONDecodeError:
                # 如果JSON解析失败，生成默认报告
                report = self._generate_default_report(paper, parsed_content, comparison_result)
            
            # 添加元数据和分析信息
            report.update({
                "paper_id": paper.get("id"),
                "paper_title": paper.get("title", ""),
                "generated_for_user_id": user_id,
                "generated_at": datetime.now().isoformat(),
                "related_papers_count": len(related_papers),
                "parsing_details": {
                    "sections_identified": len(sections_info),
                    "key_terms_extracted": len(key_terms),
                    "parsing_method": parsed_content.get("parsing_method", "unknown"),
                    "parsing_time": parsing_time
                },
                "analysis_method": "deepened_miner_agent",
                "model_version": "v2.1"
            })
            
            self._add_to_history("深度分析报告生成完成")
            return report
            
        except Exception as e:
            self._add_to_history(f"生成分析报告失败: {str(e)}")
            return self._generate_default_report(paper, parsed_content, comparison_result)
    
    def _format_key_terms(self, key_terms: List) -> str:
        """格式化关键术语"""
        if not key_terms:
            return "暂无关键术语"
        formatted = []
        for i, (term, score) in enumerate(key_terms, 1):
            formatted.append(f"{i}. {term} (重要性: {score:.2f})")
        return "\n".join(formatted)
    
    def _format_sections_info(self, sections: Dict) -> str:
        """格式化部分信息"""
        if not sections:
            return "未识别出结构化部分"
        formatted = []
        for name, section_data in sections.items():
            if isinstance(section_data, dict):
                word_count = section_data.get("word_count", 0)
                formatted.append(f"- **{name}**: {word_count} 字")
        return "\n".join(formatted) if formatted else "部分结构信息不可用"
    
    async def _convert_paper_to_dict(self, paper: ResearchPaper) -> Dict[str, Any]:
        """将 ResearchPaper 对象转换为字典"""
        sections_dict = {}
        for name, section in paper.sections.items():
            sections_dict[name] = {
                "content": section.content,
                "word_count": section.word_count
            }
        
        # 提取创新洞察
        insights = self.paper_parser.extract_innovation_insights(paper)
        
        return {
            "title": paper.metadata.title,
            "abstract": paper.metadata.abstract,
            "authors": paper.metadata.authors,
            "keywords": paper.metadata.keywords,
            "sections": sections_dict,
            "key_terms": paper.key_terms,
            "insights": insights,
            "word_count": paper.total_word_count,
            "page_count": paper.page_count,
            "parsing_method": paper.parsing_method,
            "parsing_time": paper.parsing_time
        }
    
    def _generate_default_report(self, paper: Dict, parsed_content: Dict, comparison_result: Dict) -> Dict[str, Any]:
        """生成默认报告"""
        # 从解析内容中提取关键信息
        key_terms = parsed_content.get("key_terms", [])
        insights = parsed_content.get("insights", {})
        
        innovation_points = [
            f"关键术语: {', '.join([t[0] for t in key_terms[:5]])}" if key_terms else "需要进一步分析的创新点",
            insights.get("methodology_novelty", "") or "方法论研究",
        ]
        
        return {
            "summary": f"本文 '{paper.get('title', '')}' 主要研究了相关领域的问题。共有 {len(parsed_content.get('sections', {}))} 个主要部分，总字数约 {parsed_content.get('word_count', 0)}。",
            "innovation_points": [p for p in innovation_points if p],
            "limitations": ["需要与相关工作进行详细对比分析"],
            "future_ideas": ["建议的未来研究方向"],
            "paper_id": paper.get("id"),
            "generated_at": datetime.now().isoformat(),
            "analysis_method": "deepened_analysis"
        }
    
    async def _save_analysis_report(self, paper_id: str, report: Dict, user_id: str = None) -> str:
        """保存分析报告到数据库"""
        try:
            report_id = await db_manager.create_analysis_report(
                paper_id=paper_id,
                summary=report.get("summary", ""),
                innovation_point=json.dumps(report.get("innovation_points", []), ensure_ascii=False),
                limitation=json.dumps(report.get("limitations", []), ensure_ascii=False),
                future_idea=json.dumps(report.get("future_ideas", []), ensure_ascii=False),
                vector_ids=report.get("vector_ids", {}),
                user_id=user_id
            )
            
            self._add_to_history(f"分析报告已保存: {report_id}")
            return report_id
            
        except Exception as e:
            self._add_to_history(f"保存分析报告失败: {str(e)}")
            return ""
    
    async def _update_vector_store(self, paper_id: str, paper: Dict, parsed_content: Dict, user_id: str = None):
        """更新向量库 - 利用深度解析结果"""
        try:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            
            # 从深度解析中提取关键内容
            sections = parsed_content.get("sections", {})
            key_terms = parsed_content.get("key_terms", [])
            insights = parsed_content.get("insights", {})
            keywords = parsed_content.get("keywords", [])
            
            # 组合多个层次的内容用于向量化
            content_parts = [
                title,
                abstract,
                ", ".join([t[0] for t in key_terms[:10]]),  # 添加关键词
                ", ".join(keywords),  # 添加元数据中的关键词
                insights.get("technical_highlights", "")  # 添加技术亮点
            ]
            
            # 从所有部分提取有用文本
            for section_name, section_data in sections.items():
                if isinstance(section_data, dict):
                    section_content = section_data.get("content", "")
                    if section_content:
                        content_parts.append(f"[{section_name}] {section_content[:200]}")
            
            content = " ".join([p for p in content_parts if p])
            
            # 添加到L2用户库（个人知识库）
            if user_id:
                # 构建更丰富的元数据
                metadata = {
                    "authors": paper.get("authors", []),
                    "sections_identified": list(sections.keys()),
                    "word_count": parsed_content.get("word_count", 0),
                    "page_count": parsed_content.get("page_count", 0),
                    "key_terms_count": len(key_terms),
                    "analysis_date": datetime.now().isoformat(),
                    "parsing_method": parsed_content.get("parsing_method", "unknown"),
                    "has_insights": bool(insights),
                    "innovation_indicators": insights.get("innovation_indicators", [])
                }
                
                await vector_store_manager.add_to_l2(
                    user_id=user_id,
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    content=content,
                    metadata=metadata
                )
                
                self._add_to_history(f"论文已添加到用户向量库（L2）: {user_id}，包含{len(key_terms)}个关键词")
            
            # 同时添加到全局库（L1）以便跨用户搜索
            await vector_store_manager.add_to_l1(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                content=content[:2000],  # 限制L1的内容长度
                metadata={
                    "authors": paper.get("authors", []),
                    "sections": list(sections.keys()),
                    "key_terms": [t[0] for t in key_terms[:5]]
                }
            )
            
            self._add_to_history(f"论文已同步到全局向量库（L1）")
            
        except Exception as e:
            self._add_to_history(f"更新向量库失败: {str(e)}")
    
    # 工具方法
    async def _parse_pdf(self, file_path: str) -> Dict:
        """解析PDF工具"""
        return await self._extract_structured_content(file_path)
    
    async def _search_memory(self, query: str, user_id: str = None) -> List[Dict]:
        """搜索记忆库工具"""
        try:
            results = await vector_store_manager.hybrid_search(
                query=query,
                user_id=user_id,
                top_k=5
            )
            return [{"id": r["id"], "score": r["score"], "payload": r["payload"]} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def _compare_papers(self, current_paper: Dict, related_papers: List[Dict]) -> Dict:
        """对比论文工具"""
        return await self._perform_comparison_analysis(current_paper, related_papers)
    
    async def _generate_report(self, paper_info: Dict, analysis_result: Dict) -> Dict:
        """生成报告工具"""
        return await self._create_analysis_report(
            paper_info, 
            analysis_result.get("parsed_content", {}),
            analysis_result.get("related_papers", []),
            analysis_result.get("comparison_result", {})
        )