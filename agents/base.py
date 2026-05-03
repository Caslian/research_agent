"""
InnoCore AI 基础智能体类 - 基于 LangChain 1.x 框架
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import logging

# LangChain Core 组件 (v1.x)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import Tool

# LangChain v1.x Agent API
from langchain.agents import create_agent

# LangGraph 内存管理 (v1.x 替代 ConversationBufferMemory)
from langgraph.checkpoint.memory import InMemorySaver

from core.config import get_config
from core.llm_adapter import get_llm_adapter
from core.exceptions import AgentException, TimeoutException

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """基础智能体抽象类 - LangChain 1.x 实现"""
    
    def __init__(self, name: str, llm=None, 
                 max_steps: int = None, timeout: int = None):
        self.name = name
        self.config = get_config()
        self.llm = llm or get_llm_adapter()
        
        self.max_steps = max_steps or self.config.agent_max_steps
        self.timeout = timeout or self.config.agent_timeout
        
        self.history: List[str] = []
        self.tools: List[Tool] = []
        self.tools_dict = {}  # 保留工具字典用于兼容
        self.state = "idle"
        self.created_at = datetime.now()
        
        # LangChain v1.x 组件
        # InMemorySaver 替代了旧版 ConversationBufferMemory
        self.checkpointer = InMemorySaver()
        
        # LangGraph 编译图 (替代 AgentExecutor)
        self.agent_graph = None
        
    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体任务"""
        pass
    
    def add_tool(self, tool_name: str, tool_func: Callable, description: str = ""):
        """添加工具 - LangChain v1.x Tool 格式"""
        # 创建 LangChain Tool
        tool = Tool(
            name=tool_name,
            description=description or f"Tool: {tool_name}",
            func=tool_func
        )
        self.tools.append(tool)
        
        # 同时保存到字典以保持兼容性
        self.tools_dict[tool_name] = {
            "function": tool_func,
            "description": description
        }
    
    def get_tools_description(self) -> str:
        """获取工具描述"""
        if not self.tools:
            return "暂无可用工具"
        
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    async def call_tool(self, tool_name: str, tool_input: Any) -> Any:
        """调用工具"""
        if tool_name not in self.tools_dict:
            raise AgentException(f"工具 '{tool_name}' 不存在")
        
        try:
            tool_func = self.tools_dict[tool_name]["function"]
            if asyncio.iscoroutinefunction(tool_func):
                result = await asyncio.wait_for(
                    tool_func(tool_input), 
                    timeout=self.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool_func, tool_input),
                    timeout=self.timeout
                )
            
            self._add_to_history(f"Tool {tool_name} called with input: {tool_input}")
            self._add_to_history(f"Tool {tool_name} result: {result}")
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutException(f"工具 '{tool_name}' 执行超时")
        except Exception as e:
            raise AgentException(f"工具 '{tool_name}' 执行失败: {str(e)}")
    
    async def think(self, prompt: str, context: Dict = None) -> str:
        """调用LLM进行思考 - 使用 LangChain"""
        try:
            # 构建消息列表
            messages = []
            
            # 添加系统提示
            system_prompt = self._get_system_prompt()
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # 添加上下文信息
            if context:
                context_str = json.dumps(context, ensure_ascii=False, indent=2)
                full_prompt = f"上下文信息:\n{context_str}\n\n任务:\n{prompt}"
            else:
                full_prompt = prompt
            
            # 添加历史记录
            if self.history:
                history_str = "\n".join(self.history[-10:])
                full_prompt += f"\n\n历史记录:\n{history_str}"
            
            messages.append(HumanMessage(content=full_prompt))
            
            # 调用 LangChain LLM
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=self.timeout
            )
            
            response_text = response if isinstance(response, str) else str(response)
            
            self._add_to_history(f"LLM prompt: {prompt}")
            self._add_to_history(f"LLM response: {response_text}")
            
            return response_text
            
        except asyncio.TimeoutError:
            raise TimeoutException("LLM思考超时")
        except Exception as e:
            raise AgentException(f"LLM思考失败: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词 - 子类可重写"""
        return f"你是{self.name}智能体，请根据用户需求完成任务。"
    
    def _build_agent_graph(self, system_prompt: str = None):
        """构建 LangChain v1.x Agent (返回 CompiledStateGraph)"""
        if not self.tools:
            logger.warning(f"Agent {self.name} 没有注册工具，无法创建 Agent")
            return None
        
        try:
            # 获取底层 LLM (ChatModel)
            llm = self.llm.llm if hasattr(self.llm, 'llm') else self.llm
            
            # 使用 LangChain v1.x create_agent
            # 替代旧版 create_tool_calling_agent + AgentExecutor
            self.agent_graph = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=system_prompt or self._get_system_prompt(),
                checkpointer=self.checkpointer,
            )
            
            return self.agent_graph
            
        except Exception as e:
            logger.error(f"构建 Agent 失败: {str(e)}")
            return None
    
    async def run_with_tools(self, input_text: str) -> str:
        """使用工具执行任务 - LangChain v1.x 方式"""
        if not self.agent_graph:
            self._build_agent_graph()
        
        if not self.agent_graph:
            # 如果没有工具/图，直接调用 LLM
            return await self.think(input_text)
        
        try:
            # LangChain v1.x agent 使用 LangGraph StateGraph
            # invoke 接受 {"messages": [...]} 格式
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.agent_graph.invoke,
                    {
                        "messages": [
                            {"role": "user", "content": input_text}
                        ]
                    }
                ),
                timeout=self.timeout
            )
            
            # 从结果中提取最后一条 AI 消息
            messages = result.get("messages", [])
            last_ai_msg = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and getattr(msg, 'type', '') in ('ai', 'AIMessage'):
                    last_ai_msg = msg.content
                    break
            
            return last_ai_msg or str(result)
            
        except asyncio.TimeoutError:
            raise TimeoutException("Agent执行超时")
        except Exception as e:
            logger.error(f"Agent执行失败: {str(e)}")
            # 降级为直接LLM调用
            return await self.think(input_text)
    
    def _add_to_history(self, message: str):
        """添加到历史记录"""
        timestamp = datetime.now().isoformat()
        self.history.append(f"[{timestamp}] {message}")
        
        # 限制历史记录长度
        if len(self.history) > 100:
            self.history = self.history[-50:]
    
    def get_history(self, limit: int = 10) -> List[str]:
        """获取历史记录"""
        return self.history[-limit:]
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        # 重新创建 checkpointer 以清除 LangGraph 状态
        self.checkpointer = InMemorySaver()
    
    def set_state(self, state: str):
        """设置智能体状态"""
        self.state = state
        logger.info(f"Agent {self.name} state changed to: {state}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "history_count": len(self.history),
            "tools_count": len(self.tools),
            "max_steps": self.max_steps,
            "timeout": self.timeout
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        required_fields = self.get_required_fields()
        
        for field in required_fields:
            if field not in input_data:
                raise AgentException(f"缺少必需字段: {field}")
        
        return True
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """获取必需的输入字段"""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state='{self.state}')"
    
    def __repr__(self) -> str:
        return self.__str__()
