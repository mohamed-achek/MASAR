"""
Index Module - Build FAISS indices

This module handles:
- Loading embeddings from embedding.py output
- Building FAISS vector indices (flat/IVF/HNSW)
- Saving indices and metadata mappings
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import faiss


# ============================================================================
# FAISS INDEX BUILDERS
# ============================================================================

def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build flat (exact) FAISS index using inner product.
    
    Args:
        embeddings: Normalized embeddings (N, D)
        
    Returns:
        FAISS IndexFlatIP
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(embeddings.astype('float32'))
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int = 100) -> faiss.Index:
    """
    Build IVF (inverted file) index for faster approximate search.
    
    Args:
        embeddings: Normalized embeddings (N, D)
        nlist: Number of clusters
        
    Returns:
        FAISS IndexIVFFlat
    """
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train index
    index.train(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    return index


def build_hnsw_index(embeddings: np.ndarray, m: int = 32) -> faiss.Index:
    """
    Build HNSW (hierarchical navigable small world) index.
    
    Args:
        embeddings: Normalized embeddings (N, D)
        m: Number of connections per layer
        
    Returns:
        FAISS IndexHNSWFlat
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings.astype('float32'))
    return index


# ============================================================================
# MAIN INDEXING PIPELINE
# ============================================================================

def build_indices(
    embeddings_dir: Path,
    output_dir: Path,
    faiss_type: str = "flat"
):
    """
    Build FAISS indices only (Elasticsearch removed).
    
    Args:
        embeddings_dir: Directory with embeddings from embedding.py
        output_dir: Directory to save indices
        faiss_type: Type of FAISS index (flat/ivf/hnsw)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------------------
    # Load embeddings and metadata
    # ----------------------------------------------------------------
    print("Loading embeddings and metadata...")
    
    chunk_embeddings = np.load(embeddings_dir / "chunk_embeddings.npy")
    table_embeddings = np.load(embeddings_dir / "table_embeddings.npy")
    section_embeddings = np.load(embeddings_dir / "section_embeddings.npy")
    
    with open(embeddings_dir / "chunk_metadata.json") as f:
        chunk_metadata = json.load(f)
    
    with open(embeddings_dir / "table_metadata.json") as f:
        table_metadata = json.load(f)
    
    with open(embeddings_dir / "section_metadata.json") as f:
        section_metadata = json.load(f)
    
    print(f"✅ Loaded embeddings:")
    print(f"   - Chunks: {chunk_embeddings.shape}")
    print(f"   - Tables: {table_embeddings.shape}")
    print(f"   - Sections: {section_embeddings.shape}")
    
    # ----------------------------------------------------------------
    # Build FAISS indices
    # ----------------------------------------------------------------
    print(f"\n🔹 Building FAISS indices ({faiss_type})...")
    
    # Choose index builder
    if faiss_type == "flat":
        builder = build_flat_index
    elif faiss_type == "ivf":
        builder = build_ivf_index
    elif faiss_type == "hnsw":
        builder = build_hnsw_index
    else:
        raise ValueError(f"Unknown FAISS type: {faiss_type}")
    
    # Build chunk index
    print("Building chunk index...")
    chunk_index = builder(chunk_embeddings)
    chunk_index_path = output_dir / "chunk_index.faiss"
    faiss.write_index(chunk_index, str(chunk_index_path))
    print(f"✅ Chunk index: {chunk_index.ntotal} vectors → {chunk_index_path}")
    
    # Build table index
    print("Building table index...")
    table_index = builder(table_embeddings)
    table_index_path = output_dir / "table_index.faiss"
    faiss.write_index(table_index, str(table_index_path))
    print(f"✅ Table index: {table_index.ntotal} vectors → {table_index_path}")
    
    # Build section index
    print("Building section index...")
    section_index = builder(section_embeddings)
    section_index_path = output_dir / "section_index.faiss"
    faiss.write_index(section_index, str(section_index_path))
    print(f"✅ Section index: {section_index.ntotal} vectors → {section_index_path}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'chunks': chunk_metadata,
            'tables': table_metadata,
            'sections': section_metadata
        }, f)
    print(f"✅ Metadata saved → {metadata_path}")
    
    # ----------------------------------------------------------------
    # Save index info
    # ----------------------------------------------------------------
    index_info = {
        "faiss_type": faiss_type,
        "chunk_index": str(chunk_index_path),
        "table_index": str(table_index_path),
        "section_index": str(section_index_path),
        "metadata": str(metadata_path),
        "num_chunks": len(chunk_metadata),
        "num_tables": len(table_metadata),
        "num_sections": len(section_metadata),
        "embedding_dim": chunk_embeddings.shape[1],
        "elasticsearch_enabled": False
    }
    
    info_path = output_dir / "index_info.json"
    with open(info_path, 'w') as f:
        json.dump(index_info, f, indent=2)
    
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 INDEX SUMMARY")
    print("=" * 60)
    print(f"FAISS indices created ({faiss_type}):")
    print(f"   - Chunks: {chunk_index_path}")
    print(f"   - Tables: {table_index_path}")
    print(f"   - Sections: {section_index_path}")
    print(f"   - Metadata: {metadata_path}")
    print(f"\nIndex info saved to: {info_path}")
    print("=" * 60)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for index building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS indices (FAISS-only, Elasticsearch removed)")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                       help="Directory with embeddings from embedding.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for indices")
    parser.add_argument("--faiss-type", choices=["flat", "ivf", "hnsw"], default="flat",
                       help="Type of FAISS index")
    
    args = parser.parse_args()
    
    build_indices(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        faiss_type=args.faiss_type
    )


if __name__ == "__main__":
    main()
