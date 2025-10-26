"""
Shared reranker service using BGE-reranker-large.
Implements singleton pattern to avoid loading model multiple times.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Any
import torch


class SharedReranker:
    """Singleton reranker service using BGE-reranker-large."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name: str = "BAAI/bge-reranker-large", device: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = None):
        # Only initialize once
        if self._initialized:
            return
        
        self.model_name = model_name
        # Default to CPU to avoid GPU memory issues when embedder is also loaded
        self.device = device or 'cpu'
        
        print(f"Loading BGE-reranker-large on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        
        self._initialized = True
        print(f"✅ BGE-reranker-large loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        text_key: str = 'text'
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of top documents to return
            text_key: Key in document dict containing text
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc.get(text_key, '')] for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by score (descending) and return top k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
    
    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Score query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        scores = self.model.predict(query_doc_pairs)
        return scores.tolist()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
