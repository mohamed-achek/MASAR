"""
Streaming JSON utilities for efficient large file processing.
Handles reading and writing large JSON files without loading everything into memory.
"""

import json
from typing import Iterator, Dict, Any, List
from pathlib import Path


def stream_write_chunks(chunks: Iterator[Dict[str, Any]], output_file: Path, chunk_key: str = "chunks"):
    """
    Write chunks to JSON file in streaming fashion.
    
    Args:
        chunks: Iterator/generator of chunk dictionaries
        output_file: Path to output JSON file
        chunk_key: Key name for chunks array in JSON
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{')
        f.write(f'"{chunk_key}": [\n')
        
        first = True
        count = 0
        
        for chunk in chunks:
            if not first:
                f.write(',\n')
            else:
                first = False
            
            # Write chunk without extra indentation to save space
            json.dump(chunk, f, ensure_ascii=False)
            count += 1
        
        f.write(f'\n],\n"total_{chunk_key}": {count}\n')
        f.write('}')
    
    return count


def stream_write_full_etl_output(
    text_chunks: Iterator[Dict[str, Any]],
    table_items: Iterator[Dict[str, Any]],
    output_file: Path
):
    """
    Write full ETL output with both text chunks and tables in streaming fashion.
    
    Args:
        text_chunks: Iterator of text chunk dictionaries
        table_items: Iterator of table dictionaries
        output_file: Path to output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        
        # Write chunks
        f.write('"chunks": [\n')
        first = True
        chunk_count = 0
        
        for chunk in text_chunks:
            if not first:
                f.write(',\n')
            else:
                first = False
            json.dump(chunk, f, ensure_ascii=False)
            chunk_count += 1
        
        f.write('\n],\n')
        
        # Write table items
        f.write('"table_rows": [\n')
        first = True
        table_count = 0
        
        for table in table_items:
            if not first:
                f.write(',\n')
            else:
                first = False
            json.dump(table, f, ensure_ascii=False)
            table_count += 1
        
        f.write('\n],\n')
        
        # Write totals
        f.write(f'"total_chunks": {chunk_count},\n')
        f.write(f'"total_table_rows": {table_count}\n')
        f.write('}')
    
    return chunk_count, table_count


def stream_read_chunks(input_file: Path, chunk_key: str = "chunks") -> Iterator[Dict[str, Any]]:
    """
    Read chunks from JSON file in streaming fashion using ijson.
    
    Args:
        input_file: Path to input JSON file
        chunk_key: Key name for chunks array in JSON
        
    Yields:
        Individual chunk dictionaries
    """
    try:
        import ijson
        
        with open(input_file, 'rb') as f:
            # Stream parse the chunks array
            parser = ijson.items(f, f'{chunk_key}.item')
            for chunk in parser:
                yield chunk
                
    except ImportError:
        # Fallback to standard json if ijson not available
        print("⚠️  ijson not installed, falling back to standard json.load()")
        print("   Install ijson for better performance: pip install ijson")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for chunk in data.get(chunk_key, []):
                yield chunk


def batch_process_json(
    input_file: Path,
    output_file: Path,
    processor_func,
    batch_size: int = 100,
    chunk_key: str = "chunks"
):
    """
    Process large JSON file in batches without loading all into memory.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        processor_func: Function that takes a chunk and returns processed chunk
        batch_size: Number of chunks to process at once
        chunk_key: Key name for chunks array in JSON
    """
    def process_batches():
        batch = []
        
        for chunk in stream_read_chunks(input_file, chunk_key):
            batch.append(processor_func(chunk))
            
            if len(batch) >= batch_size:
                yield from batch
                batch = []
        
        # Yield remaining items
        if batch:
            yield from batch
    
    count = stream_write_chunks(process_batches(), output_file, chunk_key)
    return count


def get_json_size_info(file_path: Path) -> Dict[str, Any]:
    """
    Get size information about a JSON file without loading it fully.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with file size and estimated chunk count
    """
    import os
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Quick estimate of chunk count by counting opening braces
    # This is much faster than parsing the whole file
    chunk_count_estimate = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first 1000 lines to estimate
        for i, line in enumerate(f):
            if '"chunk_id"' in line or '"id"' in line:
                chunk_count_estimate += 1
            if i >= 1000:
                # If file is large, break early and extrapolate
                if file_size_mb > 10:
                    total_lines = sum(1 for _ in open(file_path))
                    chunk_count_estimate = int(chunk_count_estimate * (total_lines / 1000))
                break
    
    return {
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "estimated_chunks": chunk_count_estimate
    }


# Example usage functions

def example_streaming_write():
    """Example of streaming write."""
    def chunk_generator():
        for i in range(1000):
            yield {
                "chunk_id": f"chunk_{i}",
                "text": f"This is chunk {i}",
                "metadata": {"index": i}
            }
    
    output_path = Path("output_streamed.json")
    count = stream_write_chunks(chunk_generator(), output_path)
    print(f"✅ Wrote {count} chunks to {output_path}")


def example_streaming_read():
    """Example of streaming read."""
    input_path = Path("output_streamed.json")
    
    count = 0
    for chunk in stream_read_chunks(input_path):
        count += 1
        if count <= 5:  # Print first 5
            print(f"Chunk {count}: {chunk['chunk_id']}")
    
    print(f"✅ Read {count} chunks from {input_path}")


def example_batch_process():
    """Example of batch processing."""
    input_path = Path("input.json")
    output_path = Path("output_processed.json")
    
    def uppercase_text(chunk):
        chunk['text'] = chunk['text'].upper()
        return chunk
    
    count = batch_process_json(input_path, output_path, uppercase_text, batch_size=50)
    print(f"✅ Processed {count} chunks")


if __name__ == "__main__":
    print("Streaming JSON Utilities")
    print("Import this module to use the functions")
