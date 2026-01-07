"""Tool implementations for agent"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import requests
from datetime import datetime

# Forward declaration to avoid circular import
if False:
    from multi_agent import AgentManager


class Tool(ABC):
    """Base class for tools"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get tool description"""
        pass


class WebSearchTool(Tool):
    """Web search tool"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search the web
        
        Args:
            query: Search query
            num_results: Number of results
            
        Returns:
            Search results
        """
        # This is a placeholder - implement with actual API
        return {
            "query": query,
            "results": [],
            "status": "API key not configured"
        }
    
    def get_description(self) -> str:
        return "Search the web for information"


class CalculatorTool(Tool):
    """Calculator tool"""
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate mathematical expression
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculation result
        """
        try:
            # Safe evaluation
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": lambda x: x**0.5
            })
            return {
                "expression": expression,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "status": "error"
            }
    
    def get_description(self) -> str:
        return "Perform mathematical calculations"


class KnowledgeBaseTool(Tool):
    """Knowledge base query tool"""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
    
    def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query knowledge base
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            Knowledge base results
        """
        context = self.rag_pipeline.retrieve_context(query, top_k=top_k)
        
        return {
            "query": query,
            "results": [
                {
                    "content": content,
                    "similarity": similarity
                }
                for content, similarity in context.retrieved_entries
            ],
            "status": "success"
        }
    
    def get_description(self) -> str:
        return "Query the agent's knowledge base"


class ToolManager:
    """Manages available tools"""
    
    def __init__(self, rag_pipeline=None):
        self.tools = {}
        self.rag_pipeline = rag_pipeline
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        self.register_tool("calculator", CalculatorTool())
        
        if self.rag_pipeline:
            self.register_tool("knowledge_base", KnowledgeBaseTool(self.rag_pipeline))
    
    def register_tool(self, name: str, tool: Tool):
        """Register a tool"""
        self.tools[name] = tool
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool
        
        Args:
            tool_name: Name of tool
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            result = self.tools[tool_name].execute(**kwargs)
            result["tool"] = tool_name
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception as e:
            return {
                "tool": tool_name,
                "error": str(e),
                "status": "error"
            }
    
    def get_tools_description(self) -> Dict[str, str]:
        """Get descriptions of all tools"""
        return {
            name: tool.get_description()
            for name, tool in self.tools.items()
        }
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())

