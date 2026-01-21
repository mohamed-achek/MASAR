"""
Shared utility modules for the MASAR project.

Note: Heavy imports (torch, transformers) are done lazily to support
lightweight deployments that use microservices instead.
"""

# Don't auto-import heavy modules - use explicit imports when needed:
# from utils.embedding_utils import SharedEmbeddingService
# from utils.reranker_utils import SharedReranker
# from utils.embedder_client import EmbedderClient, SyncEmbedderClient
# from utils.reranker_client import RerankerClient, SyncRerankerClient

__all__ = []
