"""
RAG Data Manager - Interface for storing and retrieving RAG data from PostgreSQL

This module provides:
- Functions to store chunks, embeddings, and documents
- Functions to query data for RAG operations
- Sync/async support
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from database_rag_schema import Document, Chunk, Embedding, FAISSIndex, QueryHistory, RetrievedChunk

# ============================================================================
# DOCUMENT OPERATIONS
# ============================================================================

async def store_document(
    session: AsyncSession,
    source_file: str,
    file_path: str,
    university_id: str,
    program: str,
    year: str,
    **kwargs
) -> Document:
    """
    Store a document in the database.
    
    Args:
        session: Database session
        source_file: Source filename
        file_path: Path to file
        university_id: University ID
        program: Program name
        year: Year
        **kwargs: Additional metadata
        
    Returns:
        Created Document object
    """
    document = Document(
        source_file=source_file,
        file_path=file_path,
        file_type=kwargs.get('file_type', 'md'),
        university_id=university_id,
        program=program,
        year=year,
        file_size=kwargs.get('file_size'),
        num_pages=kwargs.get('num_pages'),
        num_chunks=kwargs.get('num_chunks', 0),
        meta_data=kwargs.get('metadata', {})
    )
    
    session.add(document)
    await session.commit()
    await session.refresh(document)
    
    print(f"✅ Stored document: {source_file} (ID: {document.id})")
    return document


async def get_document_by_source(session: AsyncSession, source_file: str) -> Optional[Document]:
    """Get document by source filename"""
    result = await session.execute(
        select(Document).where(Document.source_file == source_file)
    )
    return result.scalar_one_or_none()


async def get_documents(
    session: AsyncSession,
    university_id: Optional[str] = None,
    program: Optional[str] = None,
    year: Optional[str] = None
) -> List[Document]:
    """Get documents with optional filters"""
    query = select(Document)
    
    filters = []
    if university_id:
        filters.append(Document.university_id == university_id)
    if program:
        filters.append(Document.program == program)
    if year:
        filters.append(Document.year == year)
    
    if filters:
        query = query.where(and_(*filters))
    
    result = await session.execute(query)
    return result.scalars().all()


# ============================================================================
# CHUNK OPERATIONS
# ============================================================================

async def store_chunks_batch(
    session: AsyncSession,
    chunks_data: List[Dict[str, Any]],
    document_id: int
) -> List[Chunk]:
    """
    Store multiple chunks in batch.
    
    Args:
        session: Database session
        chunks_data: List of chunk dictionaries
        document_id: Parent document ID
        
    Returns:
        List of created Chunk objects
    """
    chunks = []
    
    for i, chunk_data in enumerate(chunks_data):
        chunk = Chunk(
            chunk_id=chunk_data['chunk_id'],
            document_id=document_id,
            text=chunk_data['text'],
            chunk_type=chunk_data.get('type', 'text'),
            section_title=chunk_data.get('section_title', ''),
            chunk_index=i,
            summary=chunk_data.get('summary', ''),
            table_html=chunk_data.get('table_html'),
            table_description=chunk_data.get('table_description'),
            meta_data=chunk_data.get('metadata', {})
        )
        chunks.append(chunk)
        session.add(chunk)
    
    await session.commit()
    
    # Refresh all chunks to get IDs
    for chunk in chunks:
        await session.refresh(chunk)
    
    print(f"✅ Stored {len(chunks)} chunks for document ID {document_id}")
    return chunks


async def get_chunk_by_chunk_id(session: AsyncSession, chunk_id: str) -> Optional[Chunk]:
    """Get chunk by chunk_id string"""
    result = await session.execute(
        select(Chunk).where(Chunk.chunk_id == chunk_id)
    )
    return result.scalar_one_or_none()


async def get_chunks_by_document(session: AsyncSession, document_id: int) -> List[Chunk]:
    """Get all chunks for a document"""
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    return result.scalars().all()


async def search_chunks_by_text(
    session: AsyncSession,
    search_text: str,
    limit: int = 10
) -> List[Chunk]:
    """Simple text search in chunks (PostgreSQL full-text search)"""
    result = await session.execute(
        select(Chunk)
        .where(Chunk.text.ilike(f'%{search_text}%'))
        .limit(limit)
    )
    return result.scalars().all()


# ============================================================================
# EMBEDDING OPERATIONS
# ============================================================================

async def store_embeddings_metadata(
    session: AsyncSession,
    chunk_ids: List[int],
    model_name: str,
    embedding_dim: int,
    numpy_file: str,
    faiss_start_id: int = 0
) -> List[Embedding]:
    """
    Store embedding metadata (actual vectors stored in FAISS/numpy).
    
    Args:
        session: Database session
        chunk_ids: List of chunk database IDs
        model_name: Embedding model name
        embedding_dim: Embedding dimension
        numpy_file: Path to numpy file
        faiss_start_id: Starting position in FAISS index
        
    Returns:
        List of created Embedding objects
    """
    embeddings = []
    
    for i, chunk_id in enumerate(chunk_ids):
        embedding = Embedding(
            chunk_id=chunk_id,
            model_name=model_name,
            embedding_dim=embedding_dim,
            faiss_index_id=faiss_start_id + i,
            numpy_file=numpy_file
        )
        embeddings.append(embedding)
        session.add(embedding)
    
    await session.commit()
    
    for embedding in embeddings:
        await session.refresh(embedding)
    
    print(f"✅ Stored {len(embeddings)} embedding metadata entries")
    return embeddings


async def get_embedding_by_chunk(session: AsyncSession, chunk_id: int) -> Optional[Embedding]:
    """Get embedding metadata for a chunk"""
    result = await session.execute(
        select(Embedding).where(Embedding.chunk_id == chunk_id)
    )
    return result.scalar_one_or_none()


# ============================================================================
# FAISS INDEX OPERATIONS
# ============================================================================

async def store_faiss_index_metadata(
    session: AsyncSession,
    index_name: str,
    file_path: str,
    dimension: int,
    num_vectors: int,
    index_type: str = 'flat',
    model_name: str = 'BAAI/bge-m3',
    **kwargs
) -> FAISSIndex:
    """Store FAISS index metadata"""
    faiss_index = FAISSIndex(
        index_name=index_name,
        index_type=index_type,
        file_path=file_path,
        dimension=dimension,
        num_vectors=num_vectors,
        model_name=model_name,
        index_params=kwargs.get('index_params', {})
    )
    
    session.add(faiss_index)
    await session.commit()
    await session.refresh(faiss_index)
    
    print(f"✅ Stored FAISS index metadata: {index_name}")
    return faiss_index


async def get_faiss_index(session: AsyncSession, index_name: str) -> Optional[FAISSIndex]:
    """Get FAISS index metadata by name"""
    result = await session.execute(
        select(FAISSIndex).where(FAISSIndex.index_name == index_name)
    )
    return result.scalar_one_or_none()


# ============================================================================
# QUERY HISTORY OPERATIONS
# ============================================================================

async def log_query(
    session: AsyncSession,
    question: str,
    answer: str,
    num_sources: int,
    response_time_ms: int,
    user_id: Optional[int] = None,
    **kwargs
) -> QueryHistory:
    """Log a RAG query"""
    query_history = QueryHistory(
        user_id=user_id,
        question=question,
        answer=answer,
        num_sources=num_sources,
        response_time_ms=response_time_ms,
        llm_model=kwargs.get('llm_model', 'mistral'),
        embedding_model=kwargs.get('embedding_model', 'BAAI/bge-m3'),
        question_type=kwargs.get('question_type', 'qa'),
        meta_filters=kwargs.get('metadata_filters', {})
    )
    
    session.add(query_history)
    await session.commit()
    await session.refresh(query_history)
    
    return query_history


async def get_query_history(
    session: AsyncSession,
    user_id: Optional[int] = None,
    limit: int = 100
) -> List[QueryHistory]:
    """Get query history with optional user filter"""
    query = select(QueryHistory).order_by(QueryHistory.timestamp.desc()).limit(limit)
    
    if user_id:
        query = query.where(QueryHistory.user_id == user_id)
    
    result = await session.execute(query)
    return result.scalars().all()


# ============================================================================
# ANALYTICS
# ============================================================================

async def get_query_stats(session: AsyncSession) -> Dict[str, Any]:
    """Get query statistics"""
    # Total queries
    total_result = await session.execute(select(func.count(QueryHistory.id)))
    total_queries = total_result.scalar()
    
    # Average response time
    avg_time_result = await session.execute(
        select(func.avg(QueryHistory.response_time_ms))
    )
    avg_response_time = avg_time_result.scalar() or 0
    
    # Most common question types
    type_result = await session.execute(
        select(QueryHistory.question_type, func.count(QueryHistory.id))
        .group_by(QueryHistory.question_type)
    )
    question_types = dict(type_result.all())
    
    return {
        "total_queries": total_queries,
        "avg_response_time_ms": round(avg_response_time, 2),
        "question_types": question_types
    }


async def get_document_stats(session: AsyncSession) -> Dict[str, Any]:
    """Get document statistics"""
    # Total documents
    doc_count_result = await session.execute(select(func.count(Document.id)))
    total_documents = doc_count_result.scalar()
    
    # Total chunks
    chunk_count_result = await session.execute(select(func.count(Chunk.id)))
    total_chunks = chunk_count_result.scalar()
    
    # Documents by university
    uni_result = await session.execute(
        select(Document.university_id, func.count(Document.id))
        .group_by(Document.university_id)
    )
    docs_by_university = dict(uni_result.all())
    
    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "documents_by_university": docs_by_university
    }
