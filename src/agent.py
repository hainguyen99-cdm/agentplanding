"""Main AI Agent with OpenAI integration and long-term memory"""
import logging
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from config import get_config
from rag_pipeline import RAGPipeline
from knowledge_extractor import KnowledgeExtractor


class AIAgent:
    """
    AI Agent with:
    - OpenAI integration
    - Long-term memory (Vector DB)
    - RAG pipeline
    - Configurable personality
    """
    
    def __init__(self, config_path: str = "config.yaml", agent_id: Optional[str] = None, role: Optional[str] = None, vector_db: Optional[object] = None):
        """Initialize agent
        Args:
            config_path: Path to configuration file
            agent_id: Optional identifier for this agent (multi-agent)
            role: Optional system role override
            vector_db: Optional injected VectorDB/CompositeVectorDB
        """
        from config import set_config, Config
        
        # Load configuration
        config = Config.from_yaml(config_path)
        config.ensure_directories()  # Create directories first
        set_config(config)
        
        self.config = config
        self.agent_id = agent_id
        if role:
            self.config.agent.role = role
        self.client = OpenAI(api_key=config.openai.api_key)
        
        # Setup logging (after directories are created)
        self._setup_logging()
        
        # Initialize components (inject vector db if provided)
        self.rag_pipeline = RAGPipeline(vector_db=vector_db)
        self.extractor = KnowledgeExtractor()
        
        # Conversation history
        self.conversation_history = []
        
        self.logger.info(f"Agent '{config.agent.name}' initialized" + (f" (id={self.agent_id})" if self.agent_id else ""))
    
    def _setup_logging(self):
        """Setup logging"""
        from pathlib import Path
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.logging.level)
        
        # Remove existing handlers to avoid console logging
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        
        # Create log directory if it doesn't exist
        log_file = Path(self.config.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(self.config.logging.log_file, encoding="utf-8")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Prevent double logging to console (fix Windows cp1252 Unicode issues)
        self.logger.propagate = False
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on agent configuration"""
        config = self.config.agent
        
        personality_desc = {
            "friendly": "thân thiện, vui vẻ, dễ gần",
            "professional": "chuyên nghiệp, trang trọng, chính xác",
            "casual": "thoải mái, không chính thức, gần gũi",
            "humorous": "hài hước, vui nhộn, thích đùa"
        }.get(config.personality, "thân thiện")
        
        speaking_style_desc = {
            "natural": "tự nhiên và lưu loát",
            "formal": "trang trọng và chuẩn mực",
            "casual": "thân mật và không chính thức",
            "poetic": "thơ mộng và uyển chuyển"
        }.get(config.speaking_style, "tự nhiên")
        
        return f"""Bạn là {config.name}, một AI assistant.

Thông tin cá nhân:
- Vai trò: {config.role}
- Tuổi: {config.age}
- Giới tính: {config.gender}
- Ngôn ngữ chính: {config.language}

Tính cách và phong cách:
- Tính cách: {personality_desc}
- Phong cách nói chuyện: {speaking_style_desc}

Hướng dẫn:
1. Luôn trả lời bằng {config.language}
2. Giữ phong cách nhất quán
3. Sử dụng kiến thức từ cơ sở dữ liệu khi có sẵn
4. Nếu không chắc, hãy nói rõ điều đó
5. Học hỏi từ mỗi cuộc trò chuyện
6. Tôn trọng vai trò đã chỉ định khi hợp tác cùng các agent khác"""
    
    def process_message(self, user_message: str, extract_knowledge: bool = True) -> Dict:
        """
        Process user message with full pipeline
        
        Args:
            user_message: User input
            extract_knowledge: Whether to extract knowledge from user message
            
        Returns:
            Response with metadata
        """
        self.logger.info(f"Processing message: {user_message[:100]}")
        
        result = {
            "user_message": user_message,
            "knowledge_extraction": None,
            "rag_context": None,
            "response": None,
            "response_knowledge": None,
            "error": None
        }
        
        try:
            # Step 1: Extract knowledge from user message
            if extract_knowledge:
                extraction_result = self.rag_pipeline.process_new_information(
                    user_message,
                    source="user_chat"
                )
                result["knowledge_extraction"] = extraction_result
                self.logger.info(f"Extracted {len(extraction_result['extracted'])} knowledge entries")
            
            # Step 2: Retrieve relevant context (RAG)
            rag_result = self.rag_pipeline.process_query_with_rag(user_message)
            result["rag_context"] = rag_result["context"]
            
            # Step 3: Generate response with RAG context
            response = self._generate_response(
                user_message,
                rag_result["rag_prompt"]
            )
            result["response"] = response
            
            # Step 4: Extract knowledge from response
            response_extraction = self.rag_pipeline.process_new_information(
                response,
                source="agent_response"
            )
            result["response_knowledge"] = response_extraction
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            self.logger.info("Message processed successfully")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Error processing message: {e}")
        
        return result
    
    def _generate_response(self, user_message: str, rag_prompt: str) -> str:
        """
        Generate response using OpenAI
        
        Args:
            user_message: User message
            rag_prompt: Prompt with RAG context
            
        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-4:],  # Keep last 2 exchanges
            {"role": "user", "content": rag_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=messages,
                temperature=self.config.openai.temperature,
                max_tokens=self.config.openai.max_tokens,
                top_p=self.config.openai.top_p
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def add_knowledge(
        self,
        content: str,
        source: str = "manual",
        confidence: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Manually add knowledge entry
        
        Args:
            content: Knowledge content
            source: Source
            confidence: Confidence score
            
        Returns:
            (success, entry_id or error_message)
        """
        return self.rag_pipeline.vector_db.add_entry(
            content=content,
            source=source,
            confidence=confidence
        )
    
    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.rag_pipeline.get_knowledge_stats()
    
    def clear_knowledge(self):
        """Clear all knowledge"""
        self.rag_pipeline.clear_knowledge()
        self.logger.info("Knowledge base cleared")
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def update_config(self, **kwargs):
        """Update agent configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.agent, key):
                setattr(self.config.agent, key, value)
        self.logger.info(f"Configuration updated: {kwargs}")
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            "agent": self.config.agent.dict(),
            "openai": {
                "model": self.config.openai.model,
                "temperature": self.config.openai.temperature,
                "max_tokens": self.config.openai.max_tokens
            },
            "knowledge": {
                "max_entries": self.config.knowledge.max_entries,
                "similarity_threshold": self.config.knowledge.similarity_threshold,
                "min_confidence": self.config.knowledge.min_confidence
            }
        }

