"""Embedding generation and management"""
import os
from typing import List, Optional
import numpy as np
from openai import OpenAI
from config import get_config


class EmbeddingManager:
    """Manages text embeddings using OpenAI API"""
    
    def __init__(self):
        config = get_config()
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = config.vector_db.embedding_model
        self.dimension = config.vector_db.dimension
        self._cache = {}
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Clean text
        text = text.strip()
        if not text:
            return np.zeros(self.dimension)
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # L2-normalize for stable similarity under L2 distance
            norm = np.linalg.norm(embedding) + 1e-12
            embedding = embedding / norm
            self._cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Matrix of embeddings (n_texts, dimension)
        """
        embeddings = []
        
        # Separate cached and uncached texts
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text = text.strip()
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch embed uncached texts
        if uncached_texts:
            try:
                response = self.client.embeddings.create(
                    input=uncached_texts,
                    model=self.model
                )
                
                for idx, data in enumerate(response.data):
                    embedding = np.array(data.embedding, dtype=np.float32)
                    self._cache[uncached_texts[idx]] = embedding
                    embeddings.insert(uncached_indices[idx], embedding)
            except Exception as e:
                print(f"Error embedding texts: {e}")
                for _ in uncached_texts:
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between -1 and 1
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()

