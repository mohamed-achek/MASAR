"""
Performance optimization utilities.
"""

from functools import lru_cache
from typing import Dict, List, Any, TypeVar, Callable
import time
from pathlib import Path
import hashlib

T = TypeVar('T')


def dict_lookup_optimized(items: List[Dict[str, Any]], key_field: str = 'id') -> Dict[Any, Dict[str, Any]]:
    """
    Convert list of dicts to lookup dictionary for O(1) access.
    
    Before (O(n²)):
        for result in results:
            for metadata in all_metadata:
                if result['id'] == metadata['id']:
                    result['metadata'] = metadata
    
    After (O(n)):
        metadata_dict = dict_lookup_optimized(all_metadata, 'id')
        for result in results:
            result['metadata'] = metadata_dict.get(result['id'], {})
    
    Args:
        items: List of dictionaries
        key_field: Field name to use as dictionary key
        
    Returns:
        Dictionary mapping key_field values to full dictionaries
    """
    return {item[key_field]: item for item in items if key_field in item}


def batch_iterator(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split list into batches for more efficient processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timing_decorator
        def slow_function():
            time.sleep(1)
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️  {func.__name__} took {end - start:.2f}s")
        return result
    return wrapper


def cache_to_disk(cache_dir: Path = None):
    """
    Decorator to cache function results to disk.
    
    Args:
        cache_dir: Directory to store cache files
    
    Usage:
        @cache_to_disk()
        def expensive_computation(x):
            return x ** 2
    """
    if cache_dir is None:
        cache_dir = Path(".cache")
    
    cache_dir.mkdir(exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_str = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            cache_file = cache_dir / f"{cache_key}.cache"
            
            # Check if cached result exists
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Compute and cache result
            result = func(*args, **kwargs)
            
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        
        return wrapper
    return decorator


@lru_cache(maxsize=1)
def is_model_cached(model_name: str, cache_dir: Path = None) -> bool:
    """
    Check if a HuggingFace model is cached locally.
    
    Args:
        model_name: Name of the model (e.g., "BAAI/bge-m3")
        cache_dir: Cache directory (default: ~/.cache/huggingface)
        
    Returns:
        True if model is cached, False otherwise
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    model_path = cache_dir / f"models--{model_name.replace('/', '--')}"
    return model_path.exists()


def memory_efficient_file_reader(file_path: Path, chunk_size: int = 8192):
    """
    Read large files in chunks to avoid memory issues.
    
    Args:
        file_path: Path to file
        chunk_size: Size of each chunk in bytes
        
    Yields:
        File chunks
    """
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def deduplicate_list(items: List[T], key_func: Callable[[T], Any] = None) -> List[T]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        items: List of items
        key_func: Optional function to extract comparison key
        
    Returns:
        List with duplicates removed
    """
    if key_func is None:
        # Simple case: items are hashable
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        # Complex case: use key function
        seen = set()
        result = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


class PerformanceMonitor:
    """
    Context manager to monitor performance of code blocks.
    
    Usage:
        with PerformanceMonitor("Database query"):
            results = db.query(...)
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"🔄 Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            print(f"✅ Completed: {self.name} ({duration:.2f}s)")
        else:
            print(f"❌ Failed: {self.name} ({duration:.2f}s)")
        
        return False


def profile_memory():
    """
    Get current memory usage of the process.
    
    Returns:
        Dictionary with memory info in MB
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
    except ImportError:
        return {"error": "psutil not installed"}


# Example usage
if __name__ == "__main__":
    # Example 1: Dict lookup optimization
    metadata = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
        {"id": 3, "name": "Item 3"},
    ]
    
    metadata_dict = dict_lookup_optimized(metadata, 'id')
    print(f"Lookup result: {metadata_dict.get(2)}")
    
    # Example 2: Performance monitoring
    with PerformanceMonitor("Sleep test"):
        time.sleep(0.1)
    
    # Example 3: Memory profiling
    mem = profile_memory()
    print(f"Memory usage: {mem}")
