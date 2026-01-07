"""Configuration management for AI Agent"""
import os
from pathlib import Path
from typing import Optional, List
import yaml
from pydantic import BaseModel, Field, validator


class AgentConfig(BaseModel):
    """Agent personality and behavior configuration"""
    name: str = "Luna"
    age: int = 25
    gender: str = "female"
    language: str = "vi"
    personality: str = "friendly"
    speaking_style: str = "natural"
    role: str = "general"  # system role for the agent


class OpenAIConfig(BaseModel):
    """OpenAI API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9


class VectorDBConfig(BaseModel):
    """Vector Database configuration"""
    type: str = "faiss"
    embedding_model: str = "text-embedding-3-small"
    dimension: int = 1536
    db_path: str = "./data/vector_db"
    index_type: str = "Flat"


class KnowledgeConfig(BaseModel):
    """Knowledge management configuration"""
    max_entries: int = 10000
    similarity_threshold: float = 0.85
    min_confidence: float = 0.6
    retention_days: int = 365


class RAGConfig(BaseModel):
    """RAG pipeline configuration"""
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "./logs/agent.log"


class MultiAgentConfig(BaseModel):
    """Multi-agent configuration"""
    agents_root: str = "./data/agents"
    shared_db_path: str = "./data/shared_vector_db"
    shared_enabled: bool = True
    share_policy: str = "private"  # private | shared | both
    # Session controls
    max_rounds: int = 12
    turn_history_window: int = 6
    max_reply_chars: int = 2000
    max_transcript_chars: int = 150000
    # Auto finalization controls
    auto_finalize: bool = False
    min_rounds_before_finalize_attempt: int = 2
    max_auto_rounds: int = 20
    enforce_timeframe_days: int = 10


class DatabaseConfig(BaseModel):
    """External database/action configuration"""
    allow_actions: bool = False
    default_action: str = "none"  # none | mongodb | postgresql
    mongo_uri: str = ""
    mongo_db: str = ""
    mongo_collection: str = "content_plans"
    postgres_dsn: str = ""  # e.g. postgresql://user:pass@host:5432/dbname


class Config(BaseModel):
    """Main configuration class"""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Replace environment variables
        data = cls._replace_env_vars(data)
        return cls(**data)
    
    @staticmethod
    def _replace_env_vars(data: dict) -> dict:
        """Recursively replace ${VAR} with environment variables"""
        if isinstance(data, dict):
            return {k: Config._replace_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._replace_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.getenv(var_name, data)
        return data
    
    def ensure_directories(self):
        """Create necessary directories"""
        Path(self.vector_db.db_path).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_file).parent.mkdir(parents=True, exist_ok=True)
        # Multi-agent dirs
        if hasattr(self, "multi_agent"):
            Path(self.multi_agent.agents_root).mkdir(parents=True, exist_ok=True)
            Path(self.multi_agent.shared_db_path).mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        # Load .env if available (optional dependency)
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(override=False)
        except Exception:
            pass

        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        _config = Config.from_yaml(config_path)

        # Allow MongoDB URL to be configured via .env
        # User expects: MONGODB_URL=mongodb://localhost:27017/
        if getattr(_config, "database", None):
            if not _config.database.mongo_uri:
                # Support common typos/variants: MONGODB_URL, MONGODB_URI, MONGGODB_URL
                _config.database.mongo_uri = os.getenv(
                    "MONGODB_URL",
                    os.getenv("MONGODB_URI", os.getenv("MONGGODB_URL", ""))
                )

        _config.ensure_directories()
    return _config


def set_config(config: Config):
    """Set global config instance"""
    global _config
    _config = config

