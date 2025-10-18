"""
Index Module - Build FAISS and Elasticsearch indices

This module handles:
- Loading embeddings from embedding.py output
- Building FAISS vector indices (flat/IVF/HNSW)
- Indexing into Elasticsearch for hybrid search
- Saving indices and metadata mappings
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import faiss

# Elasticsearch client (optional)
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    print("⚠️  Elasticsearch not installed. Install with: pip install elasticsearch")


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
# ELASTICSEARCH INDEXING
# ============================================================================

def create_es_client(
    host: str,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[Elasticsearch]:
    """
    Create Elasticsearch client.
    
    Args:
        host: ES host URL (e.g., http://localhost:9200)
        user: Username (optional)
        password: Password (optional)
        
    Returns:
        ES client or None if connection fails
    """
    if not ES_AVAILABLE:
        return None
    
    try:
        # Common connection parameters
        conn_params = {
            'verify_certs': False,
            'request_timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True
        }
        
        # Add authentication if provided
        if user and password:
            conn_params['basic_auth'] = (user, password)
        
        # Create client - try with API compatibility mode
        try:
            # Try with headers for ES 8.x compatibility
            es = Elasticsearch(
                [host],
                headers={"accept": "application/json", "content-type": "application/json"},
                **conn_params
            )
        except Exception:
            # Fallback without headers
            es = Elasticsearch([host], **conn_params)
        
        # Test connection
        info = es.info()
        print(f"✅ Connected to Elasticsearch at {host}")
        print(f"   - Cluster: {info['cluster_name']}")
        print(f"   - Version: {info['version']['number']}")
        return es
        
    except Exception as e:
        print(f"⚠️  Failed to connect to Elasticsearch: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None


def create_es_index_with_mapping(es: Elasticsearch, index_name: str, is_table: bool = False):
    """
    Create Elasticsearch index with proper mapping.
    
    Args:
        es: ES client
        index_name: Name of index
        is_table: Whether this is for table rows
    """
    # Check if index exists
    if es.indices.exists(index=index_name):
        print(f"   - Index {index_name} already exists, deleting...")
        es.indices.delete(index=index_name)
    
    # Define mapping
    if is_table:
        mapping = {
            "mappings": {
                "properties": {
                    "row_id": {"type": "keyword"},
                    "table_html": {"type": "text", "index": False},
                    "text_fallback": {"type": "text", "analyzer": "standard"},
                    "context_paragraph": {"type": "text", "analyzer": "standard"},
                    "summary": {"type": "text", "analyzer": "standard"},
                    "section_title": {"type": "keyword"},
                    "university_id": {"type": "keyword"},
                    "program": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "source_file": {"type": "keyword"},
                    "row_index": {"type": "integer"},
                    "table_index": {"type": "integer"}
                }
            }
        }
    else:
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "summary": {"type": "text", "analyzer": "standard"},
                    "section_title": {"type": "keyword"},
                    "university_id": {"type": "keyword"},
                    "program": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "source_file": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"}
                }
            }
        }
    
    # Create index
    es.indices.create(index=index_name, body=mapping)
    print(f"   ✅ Created index: {index_name}")


def index_documents_to_es(
    es: Elasticsearch,
    index_name: str,
    documents: List[Dict[str, Any]],
    id_field: str = "chunk_id"
):
    """
    Bulk index documents into Elasticsearch.
    
    Args:
        es: ES client
        index_name: Index name
        documents: List of documents to index
        id_field: Field to use as document ID
    """
    # Prepare bulk actions
    actions = []
    for doc in documents:
        action = {
            "_index": index_name,
            "_id": doc.get(id_field),
            "_source": doc
        }
        actions.append(action)
    
    # Bulk index
    success, failed = bulk(es, actions, raise_on_error=False)
    
    if failed:
        print(f"   ⚠️  {len(failed)} documents failed to index")
    
    print(f"   ✅ Indexed {success} documents into {index_name}")
    
    # Refresh index
    es.indices.refresh(index=index_name)


# ============================================================================
# MAIN INDEXING PIPELINE
# ============================================================================

def build_indices(
    embeddings_dir: Path,
    output_dir: Path,
    faiss_type: str = "flat",
    es_host: Optional[str] = None,
    es_user: Optional[str] = None,
    es_password: Optional[str] = None
):
    """
    Build FAISS and Elasticsearch indices.
    
    Args:
        embeddings_dir: Directory with embeddings from embedding.py
        output_dir: Directory to save indices
        faiss_type: Type of FAISS index (flat/ivf/hnsw)
        es_host: Elasticsearch host (optional)
        es_user: ES username (optional)
        es_password: ES password (optional)
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
    print(f"\n🔹 Building FAISS indices...")
    
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
    chunk_index = builder(chunk_embeddings)
    chunk_index_path = output_dir / "chunk_index.faiss"
    faiss.write_index(chunk_index, str(chunk_index_path))
    print(f"✅ Built {faiss_type} index with {chunk_index.ntotal} vectors")
    print(f"✅ Saved index to {chunk_index_path}")
    
    # Build table index
    table_index = builder(table_embeddings)
    table_index_path = output_dir / "table_index.faiss"
    faiss.write_index(table_index, str(table_index_path))
    print(f"✅ Built {faiss_type} index with {table_index.ntotal} vectors")
    print(f"✅ Saved index to {table_index_path}")
    
    # Build section index
    section_index = builder(section_embeddings)
    section_index_path = output_dir / "section_index.faiss"
    faiss.write_index(section_index, str(section_index_path))
    print(f"✅ Built {faiss_type} index with {section_index.ntotal} vectors")
    print(f"✅ Saved index to {section_index_path}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'chunks': chunk_metadata,
            'tables': table_metadata,
            'sections': section_metadata
        }, f)
    print(f"✅ Saved metadata to {metadata_path}")
    
    # ----------------------------------------------------------------
    # Build Elasticsearch indices (optional)
    # ----------------------------------------------------------------
    es_indexed = False
    if es_host:
        print(f"\n🔹 Building Elasticsearch indices...")
        es = create_es_client(es_host, es_user, es_password)
        
        if es:
            try:
                # Create and index chunks
                chunk_index_name = "tbs_curriculum_chunks"
                create_es_index_with_mapping(es, chunk_index_name, is_table=False)
                
                # Prepare chunk documents for ES
                chunk_docs = []
                for chunk in chunk_metadata:
                    doc = {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text", ""),
                        "summary": chunk.get("summary", ""),
                        "section_title": chunk.get("section_title", ""),
                        "university_id": chunk.get("metadata", {}).get("university_id", ""),
                        "program": chunk.get("metadata", {}).get("program", ""),
                        "year": chunk.get("metadata", {}).get("year", 0),
                        "source_file": chunk.get("metadata", {}).get("source_file", ""),
                        "chunk_type": chunk.get("metadata", {}).get("chunk_type", "")
                    }
                    chunk_docs.append(doc)
                
                index_documents_to_es(es, chunk_index_name, chunk_docs, id_field="chunk_id")
                
                # Create and index tables
                table_index_name = "tbs_curriculum_tables"
                create_es_index_with_mapping(es, table_index_name, is_table=True)
                
                # Prepare table documents for ES
                table_docs = []
                for table in table_metadata:
                    doc = {
                        "row_id": table.get("row_id"),
                        "table_html": table.get("table_html", ""),
                        "text_fallback": table.get("text_fallback", ""),
                        "context_paragraph": table.get("context_paragraph", ""),
                        "summary": table.get("summary", ""),
                        "section_title": table.get("section_title", ""),
                        "university_id": table.get("metadata", {}).get("university_id", ""),
                        "program": table.get("metadata", {}).get("program", ""),
                        "year": table.get("metadata", {}).get("year", 0),
                        "source_file": table.get("metadata", {}).get("source_file", ""),
                        "row_index": table.get("row_index", 0),
                        "table_index": table.get("table_index", 0)
                    }
                    table_docs.append(doc)
                
                index_documents_to_es(es, table_index_name, table_docs, id_field="row_id")
                
                es_indexed = True
                print(f"✅ Elasticsearch indices created successfully")
                
            except Exception as e:
                print(f"⚠️  Elasticsearch indexing failed: {e}")
                print("   Continuing with FAISS-only indices")
    
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
        "elasticsearch_enabled": es_indexed
    }
    
    if es_indexed:
        index_info["es_chunk_index"] = chunk_index_name
        index_info["es_table_index"] = table_index_name
    
    info_path = output_dir / "index_info.json"
    with open(info_path, 'w') as f:
        json.dump(index_info, f, indent=2)
    
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 INDEX SUMMARY")
    print("=" * 60)
    print(f"FAISS indices created:")
    print(f"   - Chunks: {chunk_index_path}")
    print(f"   - Tables: {table_index_path}")
    print(f"   - Sections: {section_index_path}")
    if es_indexed:
        print(f"\nElasticsearch indices created:")
        print(f"   - {chunk_index_name}: {len(chunk_docs)} documents")
        print(f"   - {table_index_name}: {len(table_docs)} documents")
    print("=" * 60)
    
    print(f"\n✅ Index info saved to {info_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for index building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS and Elasticsearch indices")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                       help="Directory with embeddings from embedding.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for indices")
    parser.add_argument("--faiss-type", choices=["flat", "ivf", "hnsw"], default="flat",
                       help="Type of FAISS index")
    parser.add_argument("--es-host", type=str,
                       help="Elasticsearch host (e.g., http://localhost:9200)")
    parser.add_argument("--es-user", type=str, help="ES username")
    parser.add_argument("--es-password", type=str, help="ES password")
    
    args = parser.parse_args()
    
    build_indices(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        faiss_type=args.faiss_type,
        es_host=args.es_host,
        es_user=args.es_user,
        es_password=args.es_password
    )


if __name__ == "__main__":
    main()