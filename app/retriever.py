"""
Vector retrieval module for MASAR RAG system.

This module provides vector-based document retrieval with metadata filtering:
- DocumentChunk dataclass for representing document chunks
- VectorRetriever with FAISS index
- Metadata filtering support
- Cosine similarity search

Author: MASAR Team
Date: 2024
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document with metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        text: The text content of the chunk
        doc_id: Document identifier
        metadata: Additional metadata (course code, section, page, etc.)
        embedding: Optional embedding vector
        score: Optional relevance score (set during retrieval)
    """
    chunk_id: str
    text: str
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    
    def __repr__(self) -> str:
        """String representation of the chunk."""
        return f"DocumentChunk(chunk_id='{self.chunk_id}', doc_id='{self.doc_id}', score={self.score:.4f})"
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary (excluding embedding)."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'doc_id': self.doc_id,
            'metadata': self.metadata,
            'score': self.score,
        }


class VectorRetriever:
    """
    Vector-based retriever using FAISS for efficient similarity search.
    
    Supports:
    - Building FAISS index from embeddings
    - Cosine similarity search
    - Metadata filtering
    - Top-k retrieval
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the vector retriever.
        
        Args:
            embedding_dim: Dimension of embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []  # List of DocumentChunk objects
        self.chunk_id_to_idx = {}  # Map chunk_id to index
        
        if faiss is None:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
    
    def build_index(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):
        """
        Build FAISS index from document chunks and embeddings.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Numpy array of shape (num_chunks, embedding_dim)
        """
        if faiss is None:
            logger.error("Cannot build index: FAISS not installed")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match embeddings ({len(embeddings)})")
        
        # Store chunks
        self.chunks = chunks
        
        # Build chunk_id to index mapping
        self.chunk_id_to_idx = {
            chunk.chunk_id: idx
            for idx, chunk in enumerate(chunks)
        }
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.index.add(embeddings_normalized.astype('float32'))
        
        logger.info(f"✅ Built FAISS index with {self.index.ntotal} vectors (dim={self.embedding_dim})")
    
    def load_index(
        self,
        index_path: str,
        chunks: List[DocumentChunk]
    ):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to the FAISS index file
            chunks: List of DocumentChunk objects (must match index order)
        """
        if faiss is None:
            logger.error("Cannot load index: FAISS not installed")
            return
        
        try:
            self.index = faiss.read_index(index_path)
            self.chunks = chunks
            
            # Build chunk_id to index mapping
            self.chunk_id_to_idx = {
                chunk.chunk_id: idx
                for idx, chunk in enumerate(chunks)
            }
            
            logger.info(f"✅ Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")
        
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
    
    def save_index(self, index_path: str):
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save the FAISS index file
        """
        if self.index is None:
            logger.error("No index to save")
            return
        
        try:
            faiss.write_index(self.index, index_path)
            logger.info(f"✅ Saved FAISS index to {index_path}")
        
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Search for similar document chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filters (key-value pairs)
        
        Returns:
            List of DocumentChunk objects with scores
            
        Examples:
            >>> retriever = VectorRetriever()
            >>> retriever.build_index(chunks, embeddings)
            >>> results = retriever.search(query_emb, top_k=5)
            >>> results[0].text
            'BCOR 260 requires BCOR 130 as prerequisite'
            >>> results[0].score
            0.8542
        """
        if self.index is None:
            logger.error("No index built. Call build_index() first.")
            return []
        
        if query_embedding.shape[0] != self.embedding_dim:
            logger.error(f"Query embedding dimension {query_embedding.shape[0]} != index dimension {self.embedding_dim}")
            return []
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.astype('float32').reshape(1, -1)
        
        # Search FAISS index (get more than top_k for filtering)
        search_k = min(top_k * 3, self.index.ntotal) if metadata_filter else top_k
        scores, indices = self.index.search(query_normalized, search_k)
        
        # Convert to list
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        # Create result chunks
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Apply metadata filter
            if metadata_filter:
                match = all(
                    chunk.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )
                if not match:
                    continue
            
            # Create result chunk (copy with score)
            result_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_id=chunk.doc_id,
                metadata=chunk.metadata.copy(),
                score=float(score),
            )
            
            results.append(result_chunk)
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        logger.info(f"Retrieved {len(results)} chunks (top_k={top_k})")
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a chunk by its ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
        
        Returns:
            DocumentChunk object or None if not found
        """
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is None:
            return None
        
        return self.chunks[idx]
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Get all chunks in the index.
        
        Returns:
            List of all DocumentChunk objects
        """
        return self.chunks.copy()
    
    def size(self) -> int:
        """
        Get the number of chunks in the index.
        
        Returns:
            Number of chunks
        """
        return len(self.chunks)


def create_sample_chunks() -> List[DocumentChunk]:
    """
    Create sample document chunks for testing.
    
    Returns:
        List of sample DocumentChunk objects
    """
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="BCOR 260 Financial Management requires BCOR 130 Introduction to Business as a prerequisite.",
            doc_id="tbs_handbook",
            metadata={'course_code': 'BCOR 260', 'section': 'prerequisites', 'page': 42}
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            text="The Bachelor of Business Administration program consists of 120 credit hours.",
            doc_id="tbs_handbook",
            metadata={'program': 'BBA', 'section': 'curriculum', 'page': 15}
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            text="Students must maintain a GPA of 2.0 or higher to remain in good academic standing.",
            doc_id="tbs_handbook",
            metadata={'section': 'academic_policies', 'page': 8}
        ),
        DocumentChunk(
            chunk_id="chunk_4",
            text="ACCT 201 Principles of Accounting I covers fundamental accounting concepts.",
            doc_id="tbs_handbook",
            metadata={'course_code': 'ACCT 201', 'section': 'course_descriptions', 'page': 50}
        ),
    ]
    
    return chunks


if __name__ == '__main__':
    # Test the retriever module
    print("=" * 80)
    print("VECTOR RETRIEVER MODULE TEST")
    print("=" * 80)
    
    # Check if FAISS is available
    if faiss is None:
        print("\n❌ FAISS not installed. Install with: pip install faiss-cpu")
        print("Skipping tests.")
    else:
        # Test 1: Create sample chunks
        print("\nTest 1: Create sample chunks")
        chunks = create_sample_chunks()
        print(f"✅ Created {len(chunks)} sample chunks")
        for chunk in chunks:
            print(f"   - {chunk.chunk_id}: {chunk.text[:50]}...")
        
        # Test 2: Generate embeddings (dummy for testing)
        print("\nTest 2: Generate dummy embeddings")
        embedding_dim = 384
        embeddings = np.random.randn(len(chunks), embedding_dim).astype('float32')
        print(f"✅ Generated {len(embeddings)} embeddings (dim={embedding_dim})")
        
        # Test 3: Build index
        print("\nTest 3: Build FAISS index")
        retriever = VectorRetriever(embedding_dim=embedding_dim)
        retriever.build_index(chunks, embeddings)
        print(f"✅ Built index with {retriever.size()} chunks")
        
        # Test 4: Search
        print("\nTest 4: Vector search")
        query_embedding = np.random.randn(embedding_dim).astype('float32')
        results = retriever.search(query_embedding, top_k=3)
        print(f"✅ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.chunk_id} (score: {result.score:.4f})")
            print(f"      {result.text[:60]}...")
        
        # Test 5: Metadata filtering
        print("\nTest 5: Metadata filtering")
        filtered_results = retriever.search(
            query_embedding,
            top_k=2,
            metadata_filter={'section': 'prerequisites'}
        )
        print(f"✅ Retrieved {len(filtered_results)} filtered results (section='prerequisites')")
        for result in filtered_results:
            print(f"   - {result.chunk_id}: {result.metadata}")
        
        # Test 6: Get chunk by ID
        print("\nTest 6: Get chunk by ID")
        chunk = retriever.get_chunk_by_id("chunk_1")
        if chunk:
            print(f"✅ Found chunk: {chunk.chunk_id}")
            print(f"   Text: {chunk.text}")
        else:
            print("❌ Chunk not found")
    
    print("\n" + "=" * 80)
