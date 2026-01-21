"""
MASAR RAG System - Application Pipeline Components

This package provides modular components for the MASAR RAG system:
- input_handler: Question record creation with preprocessing
- preprocessing: Text normalization and PII masking
- intent_ner: Intent classification and entity extraction
- retriever: Vector-based document retrieval

Note: Embedding and reranking services are now in utils/ as shared singletons

Author: MASAR Team
Date: 2024
"""

from . import preprocessing
from . import intent_ner
from . import retriever
from . import input_handler

__all__ = [
    'preprocessing',
    'intent_ner',
    'retriever',
    'input_handler',
]

__version__ = '1.0.0'
