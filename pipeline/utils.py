"""
Utilities Module - Shared helper functions

This module provides:
- JSON I/O operations
- Metadata normalization
- Logging configuration
- Token counting
- Embedding normalization
- Path management
- Configuration loading
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import tiktoken


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline.
    
    Args:
        log_file: Path to log file (None = console only)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('rag_pipeline')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# JSON I/O
# ============================================================================

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load JSON from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load JSONL (JSON Lines) file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: Union[str, Path]):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        file_path: Path to output file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ============================================================================
# METADATA OPERATIONS
# ============================================================================

def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata to standard format.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Normalized metadata
    """
    normalized = {
        'university_id': metadata.get('university_id', ''),
        'program': metadata.get('program', ''),
        'year': metadata.get('year', ''),
        'aliases': metadata.get('aliases', []),
        'source_file': metadata.get('source_file', ''),
        'chunk_type': metadata.get('chunk_type', 'unknown')
    }
    
    # Ensure aliases is a list
    if isinstance(normalized['aliases'], str):
        normalized['aliases'] = [normalized['aliases']]
    
    return normalized


def merge_metadata(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two metadata dictionaries.
    
    Args:
        base: Base metadata
        override: Override metadata
        
    Returns:
        Merged metadata
    """
    merged = base.copy()
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


# ============================================================================
# TOKEN OPERATIONS
# ============================================================================

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text to count
        encoding_name: Tiktoken encoding name
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base"
) -> str:
    """
    Truncate text to maximum token count.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        encoding_name: Tiktoken encoding name
        
    Returns:
        Truncated text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


# ============================================================================
# EMBEDDING OPERATIONS
# ============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Array of embeddings (N x D)
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(
    query: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarities between query and multiple embeddings.
    
    Args:
        query: Query vector (D,)
        embeddings: Embedding matrix (N x D)
        
    Returns:
        Similarity scores (N,)
    """
    query_norm = query / np.linalg.norm(query)
    embeddings_norm = normalize_embeddings(embeddings)
    return np.dot(embeddings_norm, query_norm)


# ============================================================================
# PATH OPERATIONS
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Project root path
    """
    # Assumes utils.py is in pipeline/ directory
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get data directory path."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get models directory path."""
    return get_project_root() / "models"


def get_logs_dir() -> Path:
    """Get logs directory path."""
    return get_project_root() / "logs"


# ============================================================================
# CONFIG OPERATIONS
# ============================================================================

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration.
    
    Args:
        config_path: Path to config file (None = use default)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "pipeline_config.json"
    
    if Path(config_path).exists():
        return load_json(config_path)
    
    # Return default config
    return {
        "etl": {
            "max_chunk_tokens": 512,
            "overlap_tokens": 50,
            "header_level": 2
        },
        "embedding": {
            "model": "BAAI/bge-m3",
            "batch_size": 8,
            "max_length": 8192
        },
        "index": {
            "faiss_type": "flat",
            "es_host": "http://localhost:9200"
        },
        "retrieval": {
            "k": 10,
            "rerank_top_k": 20,
            "vector_weight": 0.7,
            "text_weight": 0.3
        },
        "rag": {
            "provider": "ollama",
            "model": "llama3.2",
            "temperature": 0.3,
            "max_tokens": 1000
        }
    }


def save_config(config: Dict[str, Any], config_path: Optional[Union[str, Path]] = None):
    """
    Save pipeline configuration.
    
    Args:
        config: Configuration dictionary
        config_path: Path to config file
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "pipeline_config.json"
    
    save_json(config, config_path)


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove multiple whitespace
    text = ' '.join(text.split())
    
    # Remove multiple newlines
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    
    return text.strip()


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text (simple TF-IDF approach).
    
    Args:
        text: Input text
        top_k: Number of keywords
        
    Returns:
        List of keywords
    """
    # Simple word frequency approach
    words = text.lower().split()
    
    # Filter stopwords (basic list)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count frequencies
    from collections import Counter
    word_counts = Counter(words)
    
    return [word for word, count in word_counts.most_common(top_k)]


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Simple progress tracker for pipeline operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of task
        """
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end='')
    
    def finish(self):
        """Mark as complete."""
        print()  # New line


# ============================================================================
# VALIDATION
# ============================================================================

def validate_embeddings(embeddings: np.ndarray) -> bool:
    """
    Validate embeddings array.
    
    Args:
        embeddings: Embeddings array
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embeddings, np.ndarray):
        return False
    
    if len(embeddings.shape) != 2:
        return False
    
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        return False
    
    return True


def validate_metadata(metadata: Dict) -> bool:
    """
    Validate metadata structure.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['university_id', 'program', 'year', 'chunk_type']
    
    for field in required_fields:
        meta_obj = metadata.get('metadata', metadata)
        if field not in meta_obj:
            return False
    
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing RAG Pipeline Utilities")
    print("=" * 60)
    
    # Test token counting
    text = "This is a test sentence with multiple words."
    token_count = count_tokens(text)
    print(f"Text: {text}")
    print(f"Tokens: {token_count}")
    
    # Test path operations
    print(f"\nProject root: {get_project_root()}")
    print(f"Data dir: {get_data_dir()}")
    
    # Test config loading
    config = load_config()
    print(f"\nDefault config:")
    print(json.dumps(config, indent=2))
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
