"""
MASAR RAG System - Application Pipeline Components

This package provides modular components for the MASAR RAG system:
- input_handler: Question record creation with preprocessing
- preprocessing: Text normalization and PII masking
- intent_ner: Intent classification and entity extraction
- embeddings_service: Embedding generation and storage
- retriever: Vector-based document retrieval
- reranker: Cross-encoder reranking

Author: MASAR Team
Date: 2024
"""

from . import preprocessing
from . import intent_ner
from . import embeddings_service
from . import retriever
from . import reranker
from . import input_handler

__all__ = [
    'preprocessing',
    'intent_ner',
    'embeddings_service',
    'retriever',
    'reranker',
    'input_handler',
]

__version__ = '1.0.0'
