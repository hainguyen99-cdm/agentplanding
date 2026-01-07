"""RAG (Retrieval-Augmented Generation) Pipeline"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from vector_db import VectorDB
from knowledge_extractor import KnowledgeExtractor
from config import get_config


@dataclass
class RAGContext:
    """Context retrieved for RAG"""
    query: str
    retrieved_entries: List[Tuple[str, float]]  # (content, similarity)
    context_text: str
    top_k: int


class RAGPipeline:
    """
    RAG Pipeline: Extract → Judge → Embed → Store → Retrieve
    """
    
    def __init__(self, vector_db: Optional[object] = None):
        self.vector_db = vector_db or VectorDB()
        self.extractor = KnowledgeExtractor()
        self.config = get_config()
    
    def process_new_information(self, text: str, source: str = "chat") -> Dict:
        """
        Process new information through full pipeline
        
        Pipeline: Extract → Judge → Embed → Store
        
        Args:
            text: New information
            source: Source of information
            
        Returns:
            Processing result with statistics
        """
        result = {
            "input": text,
            "source": source,
            "extracted": [],
            "stored": [],
            "rejected": [],
            "errors": []
        }
        
        # Step 1: Extract knowledge entries
        extractions = self.extractor.extract_from_text(text, source)
        result["extracted"] = [e.content for e in extractions]
        
        if not extractions:
            return result
        
        # Step 2-5: Judge, Embed, Store for each extraction
        for extraction in extractions:
            try:
                # Step 2: Judge
                should_store, confidence, reason = self.extractor.judge_knowledge(
                    extraction.content
                )
                
                # Fallback: if judge says no but extraction suggested storing and confidence is sufficient
                if not should_store:
                    fallback_reason = None
                    if extraction.should_store and extraction.confidence >= self.config.knowledge.min_confidence:
                        should_store = True
                        confidence = max(extraction.confidence, confidence)
                        fallback_reason = "Fallback to extraction.should_store"
                    elif "Error in judgment" in reason and extraction.should_store:
                        should_store = True
                        fallback_reason = "Judge error; fallback to extraction"
                
                if not should_store:
                    result["rejected"].append({
                        "content": extraction.content,
                        "reason": reason,
                        "confidence": confidence
                    })
                    continue
                
                # Step 3-4: Embed & Store
                success, entry_id = self.vector_db.add_entry(
                    content=extraction.content,
                    source=source,
                    confidence=max(extraction.confidence, confidence),
                    metadata={
                        "extraction_reason": extraction.reason,
                        "judgment_reason": reason
                    }
                )
                
                if success:
                    result["stored"].append({
                        "entry_id": entry_id,
                        "content": extraction.content,
                        "confidence": max(extraction.confidence, confidence)
                    })
                else:
                    result["rejected"].append({
                        "content": extraction.content,
                        "reason": entry_id,  # entry_id contains error message
                        "confidence": confidence
                    })
            except Exception as e:
                result["errors"].append({
                    "content": extraction.content,
                    "error": str(e)
                })
        
        return result
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> RAGContext:
        """
        Retrieve relevant context for a query
        
        Step: Retrieve
        
        Args:
            query: User query
            top_k: Number of entries to retrieve
            
        Returns:
            RAG context with retrieved entries
        """
        if top_k is None:
            top_k = self.config.rag.top_k
        
        # Retrieve similar entries
        retrieved = self.vector_db.find_similar(query, top_k=top_k)
        
        # Build context text
        context_parts = []
        for i, (content, similarity) in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] (relevance: {similarity:.2f}) {content}")
        
        context_text = "\n".join(context_parts) if context_parts else "No relevant knowledge found."
        
        return RAGContext(
            query=query,
            retrieved_entries=retrieved,
            context_text=context_text,
            top_k=top_k
        )
    
    def build_rag_prompt(self, query: str, context: RAGContext) -> str:
        """
        Build prompt with RAG context
        
        Args:
            query: User query
            context: RAG context
            
        Returns:
            Prompt with context
        """
        config = self.config
        
        agent_info = f"""Bạn là {config.agent.name}, một AI assistant.
- Tuổi: {config.agent.age}
- Giới tính: {config.agent.gender}
- Phong cách nói chuyện: {config.agent.speaking_style}
- Tính cách: {config.agent.personality}

Sử dụng kiến thức dưới đây để trả lời câu hỏi:

{context.context_text}

Nếu kiến thức có sẵn không đủ, hãy nói rõ điều đó."""
        
        return f"""{agent_info}

Câu hỏi: {query}

Trả lời:"""
    
    def process_query_with_rag(self, query: str) -> Dict:
        """
        Process user query with RAG
        
        Args:
            query: User query
            
        Returns:
            Processing result with context and recommendations
        """
        # Retrieve context
        context = self.retrieve_context(query)
        
        return {
            "query": query,
            "context": {
                "retrieved_count": len(context.retrieved_entries),
                "entries": [
                    {
                        "content": content,
                        "similarity": similarity
                    }
                    for content, similarity in context.retrieved_entries
                ],
                "context_text": context.context_text
            },
            "rag_prompt": self.build_rag_prompt(query, context)
        }
    
    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.vector_db.get_stats()
    
    def clear_knowledge(self):
        """Clear all knowledge"""
        self.vector_db.clear()

