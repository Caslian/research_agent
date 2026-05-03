"""
LLM 适配器 - 基于 LangChain 框架
支持 OpenAI 兼容 API（OpenAI / ModelScope / DashScope / SiliconFlow / vLLM 等）
"""

import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from core.config import get_config

logger = logging.getLogger(__name__)


class LLMAdapter:
    """LLM 适配器，基于 LangChain 框架"""
    
    def __init__(self):
        """初始化 LLM 适配器"""
        self.config = get_config()
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化 LangChain LLM"""
        try:
            # LangChain OpenAI 兼容客户端
            # 支持 base_url 参数，可直连任何 OpenAI 兼容端点
            # 包括：ModelScope API / DashScope / SiliconFlow / 自建 vLLM 等
            self.llm = ChatOpenAI(
                model=self.config.llm.model_name,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout,
                streaming=False  # 禁用流式以保持兼容
            )
            logger.info(
                f"LLM 初始化成功: {self.config.llm.model_name} "
                f"(base_url={self.config.llm.base_url or 'OpenAI 默认'})"
            )
        except ImportError as e:
            logger.error(f"langchain-openai 未安装: {str(e)}")
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )
        except Exception as e:
            logger.error(f"LLM 初始化失败: {str(e)}")
            raise
    
    def _format_messages(self, prompt) -> List[HumanMessage]:
        """将提示词格式化为 LangChain 消息列表"""
        if isinstance(prompt, str):
            return [HumanMessage(content=prompt)]
        elif isinstance(prompt, list):
            # 处理消息列表格式 [{"role": "user", "content": "..."}]
            messages = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=content))
                elif hasattr(msg, 'content'):
                    # 已经是 Message 对象
                    messages.append(msg)
            return messages if messages else [HumanMessage(content=str(prompt))]
        else:
            return [HumanMessage(content=str(prompt))]
    
    def _extract_content(self, response) -> str:
        """从响应中提取文本内容"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        elif isinstance(response, ChatResult):
            # 处理 ChatResult
            if response.generations:
                return response.generations[0].text
        # 尝试迭代
        try:
            content = str(response)
            return content
        except:
            return str(response)
    
    async def ainvoke(self, prompt, **kwargs) -> str:
        """异步调用 LLM"""
        try:
            messages = self._format_messages(prompt)
            import asyncio
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"LLM 异步调用失败: {str(e)}")
            raise
    
    def invoke(self, prompt, **kwargs) -> str:
        """同步调用 LLM"""
        try:
            messages = self._format_messages(prompt)
            response = self.llm.invoke(messages)
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"LLM 同步调用失败: {str(e)}")
            raise
    
    def batch(self, prompts: List[str]) -> List[str]:
        """批量调用 LLM"""
        try:
            messages_list = [self._format_messages(p) for p in prompts]
            responses = self.llm.batch(messages_list)
            return [self._extract_content(r) for r in responses]
        except Exception as e:
            logger.error(f"LLM 批量调用失败: {str(e)}")
            raise


# 全局 LLM 适配器实例
_llm_adapter = None

def get_llm_adapter() -> LLMAdapter:
    """获取全局 LLM 适配器实例"""
    global _llm_adapter
    if _llm_adapter is None:
        _llm_adapter = LLMAdapter()
    return _llm_adapter