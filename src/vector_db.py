"""Vector Database management using FAISS"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from embeddings import EmbeddingManager


class KnowledgeEntry:
    """Represents a single knowledge entry"""
    
    def __init__(
        self,
        content: str,
        embedding: np.ndarray,
        entry_id: str,
        source: str = "chat",
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
        created_at: Optional[str] = None
    ):
        self.content = content
        self.embedding = embedding
        self.entry_id = entry_id
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "entry_id": self.entry_id,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class VectorDB:
    """FAISS-based Vector Database for knowledge management"""
    
    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        self.config = config
        self.embedding_manager = EmbeddingManager()
        self.db_path = Path(db_path or config.vector_db.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.db_path / "faiss_index.bin"
        self.metadata_path = self.db_path / "metadata.json"
        
        self.index = None
        self.metadata_list = []
        self.entry_map = {}  # entry_id -> index mapping
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if self.index_path.exists() and self.metadata_path.exists():
            self._load_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        dimension = self.config.vector_db.dimension
        index_type = self.config.vector_db.index_type
        
        # Create FAISS index
        self.index = faiss.index_factory(dimension, index_type)
        self.metadata_list = []
        self.entry_map = {}
    
    def _load_index(self):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata_list = data.get("metadata", [])
                self.entry_map = data.get("entry_map", {})
        except Exception as e:
            print(f"Error loading index: {e}. Creating new index.")
            self._create_new_index()
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": self.metadata_list,
                    "entry_map": self.entry_map
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def find_similar(self, text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find similar entries in the database
        
        Args:
            text: Query text
            top_k: Number of results to return
            
        Returns:
            List of (content, similarity_score) tuples
        """
        if top_k is None:
            top_k = self.config.rag.top_k
        
        if len(self.metadata_list) == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_manager.embed_text(text)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata_list)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                similarity = 1 - (distances[0][i] / 2)  # Convert distance to similarity
                results.append((metadata["content"], float(similarity)))
        
        return results
    
    def is_duplicate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is duplicate of existing entry
        
        Args:
            text: Text to check
            
        Returns:
            (is_duplicate, similar_entry_id)
        """
        similar_entries = self.find_similar(text, top_k=1)
        
        if similar_entries:
            content, similarity = similar_entries[0]
            if similarity >= self.config.knowledge.similarity_threshold:
                # Find entry_id for this content
                for entry_id, idx in self.entry_map.items():
                    if idx < len(self.metadata_list) and self.metadata_list[idx]["content"] == content:
                        return True, entry_id
        
        return False, None
    
    def add_entry(
        self,
        content: str,
        source: str = "chat",
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Add new knowledge entry
        
        Args:
            content: Knowledge content
            source: Source of knowledge (chat, tool, etc.)
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            
        Returns:
            (success, entry_id or error_message)
        """
        # Check confidence threshold
        if confidence < self.config.knowledge.min_confidence:
            return False, f"Confidence {confidence} below threshold {self.config.knowledge.min_confidence}"
        
        # Check for duplicates
        is_dup, dup_id = self.is_duplicate(content)
        if is_dup:
            return False, f"Duplicate of entry {dup_id}"
        
        # Check max entries
        if len(self.metadata_list) >= self.config.knowledge.max_entries:
            return False, "Database full"
        
        # Generate embedding
        embedding = self.embedding_manager.embed_text(content)
        
        # Create entry
        entry_id = f"entry_{len(self.metadata_list)}_{int(datetime.now().timestamp())}"
        entry = KnowledgeEntry(
            content=content,
            embedding=embedding,
            entry_id=entry_id,
            source=source,
            confidence=confidence,
            metadata=metadata
        )
        
        # Add to FAISS
        embedding_reshaped = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding_reshaped)
        
        # Add metadata
        idx = len(self.metadata_list)
        self.metadata_list.append(entry.to_dict())
        self.entry_map[entry_id] = idx
        
        # Save
        self._save_index()
        
        return True, entry_id
    
    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Get entry by ID"""
        if entry_id not in self.entry_map:
            return None
        
        idx = self.entry_map[entry_id]
        if idx < len(self.metadata_list):
            return self.metadata_list[idx]
        
        return None
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry by ID"""
        if entry_id not in self.entry_map:
            return False
        
        # Mark as deleted instead of removing (to maintain indices)
        idx = self.entry_map[entry_id]
        if idx < len(self.metadata_list):
            self.metadata_list[idx]["deleted"] = True
            self._save_index()
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        active_entries = sum(1 for m in self.metadata_list if not m.get("deleted", False))
        
        return {
            "total_entries": len(self.metadata_list),
            "active_entries": active_entries,
            "index_size": self.index.ntotal if self.index else 0,
            "db_path": str(self.db_path)
        }
    
    def clear(self):
        """Clear all data"""
        self._create_new_index()
        self._save_index()

