"""
Retrieval Module - Query processing and result ranking

This module handles:
- Query normalization and expansion
- Vector search with FAISS
- Metadata filtering
- Cross-encoder reranking
- Result fusion and deduplication
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import faiss

# Check if we should use the microservices
USE_EMBEDDER_SERVICE = os.getenv("USE_EMBEDDER_SERVICE", "false").lower() == "true"
USE_RERANKER_SERVICE = os.getenv("USE_RERANKER_SERVICE", "false").lower() == "true"


# ============================================================================
# QUERY ENCODER
# ============================================================================

class QueryEncoder:
    """Encode queries using BGE-M3 (local or remote)."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu"
    ):
        """
        Initialize query encoder.
        
        Args:
            model_name: BGE-M3 model name
            device: Device to use
        """
        self.model_name = model_name
        self._use_service = USE_EMBEDDER_SERVICE
        
        if self._use_service:
            # Use embedder microservice
            print(f"🔗 Using embedder microservice for query encoding")
            from utils.embedder_client import SyncEmbedderClient
            self._client = SyncEmbedderClient()
            self.device = "remote"
            print(f"✅ Query encoder (remote) configured")
        else:
            # Load model locally
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.device = torch.device("cpu")
            
            print(f"Loading query encoder: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Query encoder loaded on {self.device}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, query: str) -> np.ndarray:
        """
        Encode query to embedding.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding (1D numpy array)
        """
        if self._use_service:
            # Use remote embedder service
            embeddings = self._client.encode([query])
            return np.array(embeddings[0], dtype=np.float32)
        
        # Tokenize
        encoded = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embedding
        with torch.no_grad():
            model_output = self.model(**encoded)
            embedding = self.mean_pooling(model_output, encoded['attention_mask'])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = embedding.cpu().numpy().astype(np.float32)[0]
        
        return embedding_np


# ============================================================================
# CROSS-ENCODER RERANKER
# ============================================================================

class Reranker:
    """BGE reranker for result refinement (local or remote)."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu"
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: BGE reranker model name
            device: Device to use
        """
        self.model_name = model_name
        self._use_service = USE_RERANKER_SERVICE
        
        if self._use_service:
            # Use reranker microservice
            print(f"🔗 Using reranker microservice")
            from utils.reranker_client import SyncRerankerClient
            self._client = SyncRerankerClient()
            print(f"✅ Reranker (remote) configured")
        else:
            # Load model locally
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            print(f"Loading reranker: {model_name}")
            self.model = CrossEncoder(model_name, device=device)
            print(f"✅ Reranker loaded on {device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Return top K results (None = all)
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        if self._use_service:
            # Use remote reranker service - store original index in metadata
            doc_objs = [{"text": doc, "metadata": {"_idx": i}} for i, doc in enumerate(documents)]
            results = self._client.rerank(query, doc_objs, top_k=len(documents) if top_k is None else top_k)
            # Convert results to (index, score) format using stored index
            scored = []
            for r in results:
                idx = r.get("_idx", -1)
                if idx >= 0:
                    scored.append((idx, r.get("rerank_score", 0.0)))
            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            if top_k:
                return scored[:top_k]
            return scored
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        # Return top K
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """Vector retrieval using FAISS indices."""
    
    def __init__(
        self,
        index_dir: Path,
        embeddings_dir: Path,
        encoder: QueryEncoder,
        reranker: Reranker
    ):
        """
        Initialize retriever.
        
        Args:
            index_dir: Directory with FAISS indices
            embeddings_dir: Directory with metadata
            encoder: Query encoder
            reranker: Reranker model
        """
        self.encoder = encoder
        self.reranker = reranker
        
        # Load FAISS indices
        print("Loading FAISS indices...")
        self.chunk_index = faiss.read_index(str(index_dir / "chunk_index.faiss"))
        self.table_index = faiss.read_index(str(index_dir / "table_index.faiss"))
        
        # Load metadata
        print("Loading metadata...")
        with open(embeddings_dir / "chunk_metadata.json", 'r') as f:
            self.chunk_metadata = json.load(f)
        
        with open(embeddings_dir / "table_metadata.json", 'r') as f:
            self.table_metadata = json.load(f)
        
        print("✅ Retriever initialized")
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text.
        
        Args:
            query: Raw query
            
        Returns:
            Normalized query
        """
        # Basic normalization
        query = query.strip()
        query = ' '.join(query.split())  # Normalize whitespace
        return query
    
    def vector_search(
        self,
        query: str,
        k: int = 20,
        search_chunks: bool = True,
        search_tables: bool = True,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search.
        
        Args:
            query: Query text
            k: Number of results per index
            search_chunks: Search chunk index
            search_tables: Search table index
            metadata_filters: Metadata filters (e.g., {"program": "CS"})
            
        Returns:
            List of search results
        """
        # Encode query
        query_embedding = self.encoder.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        results = []
        
        # Search chunk index
        if search_chunks:
            distances, indices = self.chunk_index.search(query_embedding, k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    result = {
                        "type": "chunk",
                        "score": float(score),
                        "metadata": self.chunk_metadata[idx],
                        "text": self.chunk_metadata[idx]['text']
                    }
                    
                    # Apply metadata filters
                    if metadata_filters:
                        if self._matches_filters(result['metadata'], metadata_filters):
                            results.append(result)
                    else:
                        results.append(result)
        
        # Search table index
        if search_tables:
            distances, indices = self.table_index.search(query_embedding, k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < len(self.table_metadata):
                    metadata = self.table_metadata[idx]
                    # Try to get text from various fields
                    text = metadata.get('text_fallback') or metadata.get('text') or metadata.get('content') or str(metadata)
                    
                    result = {
                        "type": "table",
                        "score": float(score),
                        "metadata": metadata,
                        "text": text
                    }
                    
                    # Apply metadata filters
                    if metadata_filters:
                        if self._matches_filters(result['metadata'], metadata_filters):
                            results.append(result)
                    else:
                        results.append(result)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        metadata_filters: Optional[Dict] = None,
        rerank: bool = True,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with optional reranking.
        
        Args:
            query: Query text
            k: Final number of results
            metadata_filters: Metadata filters
            rerank: Apply cross-encoder reranking
            rerank_top_k: Number of results to rerank
            
        Returns:
            List of ranked search results
        """
        # Normalize query
        query = self.normalize_query(query)
        
        # Vector search
        vector_results = self.vector_search(
            query,
            k=k * 2 if rerank else k,  # Retrieve more for reranking
            metadata_filters=metadata_filters
        )
        
        # Rerank with cross-encoder
        if rerank and vector_results:
            if rerank_top_k is None:
                rerank_top_k = min(len(vector_results), k * 3)
            
            top_results = vector_results[:rerank_top_k]
            documents = [r['text'] for r in top_results]
            
            ranked = self.reranker.rerank(query, documents, top_k=k)
            
            # Reorder results based on reranking
            reranked_results = []
            for idx, score in ranked:
                result = top_results[idx].copy()
                result['rerank_score'] = float(score)
                reranked_results.append(result)
            
            return reranked_results
        
        return vector_results[:k]
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key.startswith('metadata.'):
                key = key.replace('metadata.', '')
            
            meta_obj = metadata.get('metadata', metadata)
            if meta_obj.get(key) != value:
                return False
        
        return True


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for retrieval testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vector retrieval")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory with FAISS indices")
    parser.add_argument("--embeddings-dir", type=Path, required=True, help="Directory with metadata")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing retriever...")
    encoder = QueryEncoder()
    reranker = Reranker()
    
    retriever = HybridRetriever(
        index_dir=args.index_dir,
        embeddings_dir=args.embeddings_dir,
        encoder=encoder,
        reranker=reranker
    )
    
    # Search
    print(f"\n🔍 Query: {args.query}")
    results = retriever.hybrid_search(
        query=args.query,
        k=args.k,
        rerank=not args.no_rerank
    )
    
    # Display results
    print(f"\n📊 Top {len(results)} Results:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        score = result.get('rerank_score', result['score'])
        print(f"\n{i}. [{result['type'].upper()}] Score: {score:.4f}")
        print(f"   Section: {result['metadata'].get('section_title', 'N/A')}")
        print(f"   Text: {result['text'][:200]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()
