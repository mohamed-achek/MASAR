"""
HTTP client for the Reranker microservice.
Provides async interface for document reranking when running in distributed mode.
"""

import os
import httpx
from typing import List, Dict, Any, Tuple, Optional


# Service configuration
RERANKER_URL = os.getenv("RERANKER_SERVICE_URL", "http://masar-reranker:8002")
RERANKER_TIMEOUT = float(os.getenv("RERANKER_TIMEOUT", "30.0"))


class RerankerClient:
    """Async HTTP client for the reranker microservice."""
    
    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = base_url or RERANKER_URL
        self.timeout = timeout or RERANKER_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        text_key: str = 'text'
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance via the reranker service.
        
        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of top documents to return
            text_key: Key in document dict containing text
            
        Returns:
            List of reranked documents with scores
        """
        client = await self._get_client()
        
        # Convert documents to service format
        doc_list = []
        for doc in documents:
            doc_list.append({
                "text": doc.get(text_key, ''),
                "metadata": {k: v for k, v in doc.items() if k != text_key}
            })
        
        response = await client.post(
            "/rerank",
            json={
                "query": query,
                "documents": doc_list,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert back to original format
        results = []
        for item in data["results"]:
            result = item["metadata"].copy()
            result[text_key] = item["text"]
            result["rerank_score"] = item["rerank_score"]
            results.append(result)
        
        return results
    
    async def score_pairs(
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
        client = await self._get_client()
        
        response = await client.post(
            "/score",
            json={"pairs": query_doc_pairs}
        )
        response.raise_for_status()
        
        return response.json()["scores"]
    
    async def health_check(self) -> dict:
        """Check reranker service health."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def is_ready(self) -> bool:
        """Check if reranker service is ready."""
        try:
            client = await self._get_client()
            response = await client.get("/ready")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class SyncRerankerClient:
    """Synchronous HTTP client for the reranker microservice."""
    
    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = base_url or RERANKER_URL
        self.timeout = timeout or RERANKER_TIMEOUT
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        text_key: str = 'text'
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance via the reranker service."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            # Convert documents to service format
            doc_list = []
            for doc in documents:
                doc_list.append({
                    "text": doc.get(text_key, ''),
                    "metadata": {k: v for k, v in doc.items() if k != text_key}
                })
            
            response = client.post(
                "/rerank",
                json={
                    "query": query,
                    "documents": doc_list,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert back to original format
            results = []
            for item in data["results"]:
                result = item["metadata"].copy()
                result[text_key] = item["text"]
                result["rerank_score"] = item["rerank_score"]
                results.append(result)
            
            return results
    
    def score_pairs(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post(
                "/score",
                json={"pairs": query_doc_pairs}
            )
            response.raise_for_status()
            return response.json()["scores"]
    
    def health_check(self) -> dict:
        """Check reranker service health."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()
    
    def is_ready(self) -> bool:
        """Check if reranker service is ready."""
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                response = client.get("/ready")
                return response.status_code == 200
        except Exception:
            return False


# Global client instance for convenience
_reranker_client: Optional[RerankerClient] = None


def get_reranker_client() -> RerankerClient:
    """Get the global reranker client instance."""
    global _reranker_client
    if _reranker_client is None:
        _reranker_client = RerankerClient()
    return _reranker_client
