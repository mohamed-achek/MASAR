"""
Shared reranker service using BGE-reranker-large.
Implements singleton pattern to avoid loading model multiple times.

Supports two modes:
- Local mode (default): Loads model locally in the backend pod
- Distributed mode: Uses HTTP client to call reranker microservice

Set USE_RERANKER_SERVICE=true to enable distributed mode.
"""

import os
import torch
from typing import List, Tuple, Dict, Any

# Check if we should use the microservice
USE_RERANKER_SERVICE = os.getenv("USE_RERANKER_SERVICE", "false").lower() == "true"


class SharedReranker:
    """
    Singleton reranker service using BGE-reranker-large.
    
    Automatically switches between local and distributed mode based on
    USE_RERANKER_SERVICE environment variable.
    """
    
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
        self._use_service = USE_RERANKER_SERVICE
        
        if self._use_service:
            # Distributed mode - use HTTP client
            print(f"🔗 Using reranker microservice (USE_RERANKER_SERVICE=true)")
            from utils.reranker_client import SyncRerankerClient
            self._client = SyncRerankerClient()
            self.device = "remote"
            self._initialized = True
            print(f"✅ Reranker client configured")
        else:
            # Local mode - load model directly
            from sentence_transformers import CrossEncoder
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
        
        # Use microservice if configured
        if self._use_service:
            return self._client.rerank(query, documents, top_k=top_k, text_key=text_key)
        
        # Local mode - run model directly
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
        # Use microservice if configured
        if self._use_service:
            return self._client.score_pairs(query_doc_pairs)
        
        # Local mode
        scores = self.model.predict(query_doc_pairs)
        return scores.tolist()
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        if self._use_service:
            return self._client.is_ready()
        return self._initialized
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, '_use_service') and self._use_service:
            return  # No cleanup needed for service client
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
