"""
HTTP client for the Embedder microservice.
Provides async interface for embedding generation when running in distributed mode.
"""

import os
import httpx
import numpy as np
from typing import List, Optional
from functools import lru_cache


# Service configuration
EMBEDDER_URL = os.getenv("EMBEDDER_SERVICE_URL", "http://masar-embedder:8001")
EMBEDDER_TIMEOUT = float(os.getenv("EMBEDDER_TIMEOUT", "60.0"))


class EmbedderClient:
    """Async HTTP client for the embedder microservice."""
    
    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = base_url or EMBEDDER_URL
        self.timeout = timeout or EMBEDDER_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Generate embeddings for texts via the embedder service.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2 normalize embeddings
            max_length: Maximum token length
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        client = await self._get_client()
        
        response = await client.post(
            "/embed",
            json={
                "texts": texts,
                "normalize": normalize,
                "max_length": max_length
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return np.array(data["embeddings"])
    
    async def health_check(self) -> dict:
        """Check embedder service health."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def is_ready(self) -> bool:
        """Check if embedder service is ready."""
        try:
            client = await self._get_client()
            response = await client.get("/ready")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_info(self) -> dict:
        """Get model information."""
        client = await self._get_client()
        response = await client.get("/info")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class SyncEmbedderClient:
    """Synchronous HTTP client for the embedder microservice."""
    
    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = base_url or EMBEDDER_URL
        self.timeout = timeout or EMBEDDER_TIMEOUT
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        max_length: int = 512
    ) -> np.ndarray:
        """Generate embeddings for texts via the embedder service."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post(
                "/embed",
                json={
                    "texts": texts,
                    "normalize": normalize,
                    "max_length": max_length
                }
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embeddings"])
    
    def health_check(self) -> dict:
        """Check embedder service health."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()
    
    def is_ready(self) -> bool:
        """Check if embedder service is ready."""
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                response = client.get("/ready")
                return response.status_code == 200
        except Exception:
            return False


# Global client instance for convenience
_embedder_client: Optional[EmbedderClient] = None


def get_embedder_client() -> EmbedderClient:
    """Get the global embedder client instance."""
    global _embedder_client
    if _embedder_client is None:
        _embedder_client = EmbedderClient()
    return _embedder_client
