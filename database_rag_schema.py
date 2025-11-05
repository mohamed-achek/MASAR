"""
PostgreSQL schema for RAG system data storage

This module defines tables for:
- Documents and chunks
- Embeddings metadata
- FAISS index metadata
- Query history and feedback
"""

from sqlalchemy import Column, Integer, String, Text, Float, JSON, DateTime, ForeignKey, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# ============================================================================
# DOCUMENT TABLES
# ============================================================================

class Document(Base):
    """Store source documents (PDFs, MD files)"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=True)  # Keep for backward compatibility
    file_content = Column(LargeBinary, nullable=True)  # Store actual file content
    file_type = Column(String(50))  # 'pdf', 'md', etc.
    university_id = Column(String(50), nullable=False, index=True)
    program = Column(String(255), nullable=False, index=True)
    year = Column(String(10), index=True)
    
    # Metadata
    file_size = Column(Integer)  # bytes
    num_pages = Column(Integer)
    num_chunks = Column(Integer)
    processed_date = Column(DateTime, default=datetime.utcnow)
    
    # Additional metadata as JSON
    meta_data = Column('metadata', JSON)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, source='{self.source_file}')>"


class Chunk(Base):
    """Store document chunks with embeddings"""
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(255), nullable=False, unique=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)
    
    # Chunk content
    text = Column(Text, nullable=False)
    chunk_type = Column(String(50))  # 'text', 'table', 'table_chunk'
    section_title = Column(String(500), index=True)
    
    # Positioning
    chunk_index = Column(Integer)  # Position in document
    start_char = Column(Integer)
    end_char = Column(Integer)
    
    # AI-generated content
    summary = Column(Text)
    
    # Table-specific fields
    table_html = Column(Text)  # If chunk is a table
    table_description = Column(Text)  # LLM-generated description
    
    # Metadata
    meta_data = Column('metadata', JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, chunk_id='{self.chunk_id}')>"


# ============================================================================
# EMBEDDING TABLES
# ============================================================================

class Embedding(Base):
    """Store embedding vectors (or reference to them)"""
    __tablename__ = 'embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False, unique=True, index=True)
    
    # Embedding metadata (actual vectors stored in FAISS/numpy files)
    model_name = Column(String(100), nullable=False)  # e.g., 'BAAI/bge-m3'
    embedding_dim = Column(Integer, nullable=False)  # e.g., 1024
    
    # Reference to external storage
    faiss_index_id = Column(Integer)  # Position in FAISS index
    numpy_file = Column(String(255))  # Path to numpy file
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("Chunk", back_populates="embedding")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, chunk_id={self.chunk_id}, dim={self.embedding_dim})>"


class FAISSIndex(Base):
    """Store FAISS index metadata"""
    __tablename__ = 'faiss_indices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    index_name = Column(String(100), nullable=False, unique=True)  # 'chunk_index', 'table_index'
    index_type = Column(String(50))  # 'flat', 'ivf', 'hnsw'
    
    # Index info
    file_path = Column(String(500), nullable=False)  # Path to .faiss file
    dimension = Column(Integer, nullable=False)
    num_vectors = Column(Integer, nullable=False)
    
    # Build info
    model_name = Column(String(100))
    build_date = Column(DateTime, default=datetime.utcnow)
    
    # Index parameters
    index_params = Column(JSON)
    
    def __repr__(self):
        return f"<FAISSIndex(name='{self.index_name}', vectors={self.num_vectors})>"


# ============================================================================
# QUERY HISTORY & FEEDBACK
# ============================================================================

class QueryHistory(Base):
    """Store user queries for analytics and improvement"""
    __tablename__ = 'query_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True)  # Reference to users table
    
    # Query
    question = Column(Text, nullable=False)
    question_type = Column(String(50))  # 'qa', 'comparison', 'summary'
    
    # Response
    answer = Column(Text)
    num_sources = Column(Integer)
    response_time_ms = Column(Integer)
    
    # Model info
    llm_model = Column(String(100))
    embedding_model = Column(String(100))
    
    # Metadata
    meta_filters = Column('metadata_filters', JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    feedback = relationship("QueryFeedback", back_populates="query", uselist=False)
    retrieved_chunks = relationship("RetrievedChunk", back_populates="query")
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, question='{self.question[:50]}...')>"


class QueryFeedback(Base):
    """Store user feedback on answers"""
    __tablename__ = 'query_feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey('query_history.id'), nullable=False, unique=True, index=True)
    
    # Feedback
    rating = Column(Integer)  # 1-5 stars
    helpful = Column(Boolean)
    comments = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    query = relationship("QueryHistory", back_populates="feedback")
    
    def __repr__(self):
        return f"<QueryFeedback(query_id={self.query_id}, rating={self.rating})>"


class RetrievedChunk(Base):
    """Track which chunks were retrieved for each query"""
    __tablename__ = 'retrieved_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey('query_history.id'), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False, index=True)
    
    # Retrieval scores
    rank = Column(Integer)  # Position in results (1, 2, 3...)
    vector_score = Column(Float)
    bm25_score = Column(Float)
    combined_score = Column(Float)
    rerank_score = Column(Float)
    
    # Relationships
    query = relationship("QueryHistory", back_populates="retrieved_chunks")
    chunk = relationship("Chunk")
    
    def __repr__(self):
        return f"<RetrievedChunk(query_id={self.query_id}, chunk_id={self.chunk_id}, rank={self.rank})>"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    print("✅ All RAG tables created successfully")


def drop_tables(engine):
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(engine)
    print("⚠️  All RAG tables dropped")
