"""AI Agent with Long-term Memory Package"""

from .config import Config, get_config, set_config
from .agent import AIAgent
from .vector_db import VectorDB, KnowledgeEntry
from .embeddings import EmbeddingManager
from .rag_pipeline import RAGPipeline, RAGContext
from .knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from .tools import ToolManager, Tool
from .ui import AgentUI, launch_ui

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "AIAgent",
    "VectorDB",
    "KnowledgeEntry",
    "EmbeddingManager",
    "RAGPipeline",
    "RAGContext",
    "KnowledgeExtractor",
    "ExtractedKnowledge",
    "ToolManager",
    "Tool",
    "AgentUI",
    "launch_ui",
]

