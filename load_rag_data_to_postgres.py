"""
Load existing RAG data (chunks, embeddings, indices) into PostgreSQL

This script:
1. Reads JSON chunk/metadata files
2. Stores documents and chunks in PostgreSQL
3. Stores embedding metadata
4. Updates FAISS index metadata
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from database_rag_schema import Base
from rag_data_manager import (
    store_document, store_chunks_batch, store_embeddings_metadata,
    store_faiss_index_metadata, get_document_by_source
)
from database import get_db_url


async def load_chunks_from_metadata(
    session: AsyncSession,
    metadata_file: Path
) -> Tuple[int, List[int]]:
    """
    Load chunks from metadata JSON file into database.
    
    Returns:
        Tuple of (document_id, list of chunk_db_ids)
    """
    print(f"\n📂 Loading chunks from: {metadata_file.name}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        chunks_metadata = json.load(f)
    
    if not chunks_metadata:
        print("⚠️  No chunks found in file")
        return None, []
    
    # Extract document metadata from first chunk
    first_chunk = chunks_metadata[0]
    metadata = first_chunk.get('metadata', first_chunk)
    
    source_file = metadata.get('source_file', metadata_file.stem.replace('_metadata', '') + '.md')
    university_id = metadata.get('university_id', 'Unknown')
    program = metadata.get('program', 'Unknown')
    year = metadata.get('year', 'Unknown')
    
    # Check if document already exists
    existing_doc = await get_document_by_source(session, source_file)
    
    if existing_doc:
        print(f"  📄 Document exists: {source_file} (ID: {existing_doc.id})")
        document_id = existing_doc.id
        # Check if chunks exist
        from rag_data_manager import get_chunks_by_document
        existing_chunks = await get_chunks_by_document(session, document_id)
        if existing_chunks:
            print(f"  ✓ {len(existing_chunks)} chunks already loaded, skipping...")
            return document_id, [c.id for c in existing_chunks]
    else:
        # Create document
        print(f"  📄 Creating document: {source_file}")
        document = await store_document(
            session=session,
            source_file=source_file,
            file_path=str(metadata_file.parent),
            university_id=university_id,
            program=program,
            year=year,
            file_type='md',
            num_chunks=len(chunks_metadata)
        )
        document_id = document.id
    
    # Prepare chunks data
    chunks_data = []
    for i, chunk_meta in enumerate(chunks_metadata):
        chunk_data = {
            'chunk_id': chunk_meta.get('chunk_id', f"{source_file}_{i}"),
            'text': chunk_meta.get('text', chunk_meta.get('text_fallback', '')),
            'type': chunk_meta.get('type', 'chunk'),
            'section_title': chunk_meta.get('section_title', ''),
            'summary': chunk_meta.get('summary', ''),
            'table_html': chunk_meta.get('table_html'),
            'table_description': chunk_meta.get('table_description'),
            'metadata': chunk_meta.get('metadata', {})
        }
        chunks_data.append(chunk_data)
    
    # Store chunks
    db_chunks = await store_chunks_batch(session, chunks_data, document_id)
    chunk_db_ids = [chunk.id for chunk in db_chunks]
    
    print(f"  ✅ Loaded {len(chunks_data)} chunks")
    return document_id, chunk_db_ids


async def load_embeddings_metadata(
    session: AsyncSession,
    embedding_file: Path,
    metadata_file: Path,
    chunk_db_ids: List[int],
    index_type: str = 'chunk'
):
    """Load embedding metadata into database"""
    print(f"\n📊 Loading embeddings metadata for {index_type} index")
    
    if not embedding_file.exists():
        print(f"  ⚠️  Embedding file not found: {embedding_file}")
        return
    
    # Load embeddings to get dimension
    embeddings = np.load(embedding_file)
    embedding_dim = embeddings.shape[1]
    num_embeddings = embeddings.shape[0]
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Chunks in DB: {len(chunk_db_ids)}")
    
    # Match chunk_db_ids with embeddings (assuming same order)
    if len(chunk_db_ids) != num_embeddings:
        print(f"  ⚠️  Mismatch: {len(chunk_db_ids)} chunks vs {num_embeddings} embeddings")
        print(f"  Using minimum: {min(len(chunk_db_ids), num_embeddings)}")
        min_len = min(len(chunk_db_ids), num_embeddings)
        chunk_db_ids = chunk_db_ids[:min_len]
    
    # Check if embeddings already exist
    from rag_data_manager import get_embedding_by_chunk
    first_embedding = await get_embedding_by_chunk(session, chunk_db_ids[0])
    if first_embedding:
        print(f"  ✓ Embeddings already loaded for this index, skipping...")
        return
    
    # Store embedding metadata
    await store_embeddings_metadata(
        session=session,
        chunk_ids=chunk_db_ids,
        model_name='BAAI/bge-m3',
        embedding_dim=embedding_dim,
        numpy_file=str(embedding_file),
        faiss_start_id=0
    )


async def load_faiss_index_info(
    session: AsyncSession,
    index_file: Path,
    index_name: str,
    num_vectors: int,
    dimension: int = 1024
):
    """Store FAISS index metadata"""
    print(f"\n🔍 Loading FAISS index metadata: {index_name}")
    
    if not index_file.exists():
        print(f"  ⚠️  Index file not found: {index_file}")
        return
    
    # Check if already exists
    from rag_data_manager import get_faiss_index
    existing = await get_faiss_index(session, index_name)
    if existing:
        print(f"  ✓ Index metadata already exists, skipping...")
        return
    
    await store_faiss_index_metadata(
        session=session,
        index_name=index_name,
        file_path=str(index_file),
        dimension=dimension,
        num_vectors=num_vectors,
        index_type='flat',
        model_name='BAAI/bge-m3'
    )


async def main():
    """Main data loading pipeline"""
    print("=" * 80)
    print("LOADING RAG DATA INTO POSTGRESQL")
    print("=" * 80)
    
    # Get database URL
    database_url = get_db_url()
    print(f"\n📡 Database: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    # Create engine and session
    engine = create_async_engine(database_url, echo=False)
    
    # Create tables
    print("\n🔧 Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created/verified")
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # Data paths
    base_dir = Path("data/final")
    embeddings_dir = base_dir / "embeddings"
    indices_dir = base_dir / "indices"
    
    if not embeddings_dir.exists():
        print(f"\n❌ Error: Embeddings directory not found: {embeddings_dir}")
        return
    
    async with async_session_maker() as session:
        # 1. Load chunks from metadata files
        print("\n" + "=" * 80)
        print("STEP 1: Loading Chunks from Metadata")
        print("=" * 80)
        
        chunk_metadata_file = embeddings_dir / "chunk_metadata.json"
        table_metadata_file = embeddings_dir / "table_metadata.json"
        
        all_chunk_db_ids = []
        all_table_db_ids = []
        
        if chunk_metadata_file.exists():
            doc_id, chunk_ids = await load_chunks_from_metadata(session, chunk_metadata_file)
            if chunk_ids:
                all_chunk_db_ids.extend(chunk_ids)
        else:
            print(f"⚠️  Chunk metadata not found: {chunk_metadata_file}")
        
        if table_metadata_file.exists():
            doc_id, table_ids = await load_chunks_from_metadata(session, table_metadata_file)
            if table_ids:
                all_table_db_ids.extend(table_ids)
        else:
            print(f"⚠️  Table metadata not found: {table_metadata_file}")
        
        # 2. Load embeddings metadata
        print("\n" + "=" * 80)
        print("STEP 2: Loading Embeddings Metadata")
        print("=" * 80)
        
        if all_chunk_db_ids and (embeddings_dir / "chunk_embeddings.npy").exists():
            await load_embeddings_metadata(
                session=session,
                embedding_file=embeddings_dir / "chunk_embeddings.npy",
                metadata_file=chunk_metadata_file,
                chunk_db_ids=all_chunk_db_ids,
                index_type='chunk'
            )
        
        if all_table_db_ids and (embeddings_dir / "table_embeddings.npy").exists():
            await load_embeddings_metadata(
                session=session,
                embedding_file=embeddings_dir / "table_embeddings.npy",
                metadata_file=table_metadata_file,
                chunk_db_ids=all_table_db_ids,
                index_type='table'
            )
        
        # 3. Load FAISS index metadata
        print("\n" + "=" * 80)
        print("STEP 3: Loading FAISS Index Metadata")
        print("=" * 80)
        
        if (indices_dir / "chunk_index.faiss").exists():
            chunk_embeddings = np.load(embeddings_dir / "chunk_embeddings.npy")
            await load_faiss_index_info(
                session=session,
                index_file=indices_dir / "chunk_index.faiss",
                index_name='chunk_index',
                num_vectors=chunk_embeddings.shape[0],
                dimension=chunk_embeddings.shape[1]
            )
        
        if (indices_dir / "table_index.faiss").exists():
            table_embeddings = np.load(embeddings_dir / "table_embeddings.npy")
            await load_faiss_index_info(
                session=session,
                index_file=indices_dir / "table_index.faiss",
                index_name='table_index',
                num_vectors=table_embeddings.shape[0],
                dimension=table_embeddings.shape[1]
            )
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        from rag_data_manager import get_document_stats
        stats = await get_document_stats(session)
        
        print(f"✅ Total documents: {stats['total_documents']}")
        print(f"✅ Total chunks: {stats['total_chunks']}")
        print(f"✅ Documents by university: {stats['documents_by_university']}")
        print("=" * 80)
        print("✅ Data loading complete!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
