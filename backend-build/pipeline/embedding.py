"""
Embedding Module - Generate embeddings using BGE-M3

This module handles:
- Loading BGE-M3 model (BAAI/bge-m3)
- Generating dense embeddings for chunks and table rows
- Creating hierarchical embeddings (section summaries)
- Handling both text and table content
- Batch processing for efficiency
- Hybrid CPU/GPU mode with memory capping
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================

def set_gpu_memory_limit(gpu_memory_gb: float = 2.5):
    """
    Set GPU memory limit to prevent OOM errors.
    
    Args:
        gpu_memory_gb: Maximum GPU memory to use in GB
    """
    if torch.cuda.is_available():
        # Set per-process GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(
            gpu_memory_gb / torch.cuda.get_device_properties(0).total_memory * 1024**3
        )
        print(f"⚙️  GPU memory limit set to {gpu_memory_gb} GB")
        
        # Enable memory efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# MODEL SETUP
# ============================================================================

class BGE_M3_Embedder:
    """BGE-M3 embedding model wrapper with hybrid CPU/GPU support."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 8192,
        gpu_memory_limit: Optional[float] = None
    ):
        """
        Initialize BGE-M3 model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            gpu_memory_limit: GPU memory limit in GB (None = no limit)
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        
        # Set GPU memory limit if using CUDA
        if self.device.type == "cuda" and gpu_memory_limit is not None:
            set_gpu_memory_limit(gpu_memory_limit)
        
        print(f"Loading BGE-M3 model: {model_name} on {self.device}")
        
        if self.device.type == "cuda":
            # Check available GPU memory
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / (1024**3)
            print(f"   - GPU: {gpu_props.name}")
            print(f"   - Total GPU memory: {total_memory:.2f} GB")
            if gpu_memory_limit:
                print(f"   - Memory limit: {gpu_memory_limit} GB")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with memory optimization for GPU
            if self.device.type == "cuda":
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use FP16 for GPU to save memory
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Adjust batch size for GPU memory
            if self.device.type == "cuda" and gpu_memory_limit:
                # Reduce batch size for safety with limited GPU memory
                batch_size = min(batch_size, 4)
                print(f"   - Batch size reduced to {batch_size} for GPU memory safety")
            
            self.batch_size = batch_size
            self.max_length = max_length
            
            print(f"✅ Model loaded successfully")
            print(f"   - Device: {self.device}")
            print(f"   - Data type: {next(self.model.parameters()).dtype}")
            print(f"   - Batch size: {self.batch_size}")
            print(f"   - Max length: {self.max_length}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ GPU out of memory! Falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✅ Model loaded on CPU as fallback")
            else:
                raise
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to model output.
        
        Args:
            model_output: Model output with last_hidden_state
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (N x embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = tqdm(range(num_batches), desc="Encoding") if show_progress else range(num_batches)
        
        for i in iterator:
            batch_texts = texts[i * self.batch_size : (i + 1) * self.batch_size]
            
            try:
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded)
                    embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to numpy (always float32 for consistency)
                    embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                    all_embeddings.append(embeddings_np)
                
                # Clear GPU cache if using CUDA
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device.type == "cuda":
                    print(f"\n⚠️  GPU OOM on batch {i}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Retry with CPU for this batch
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        model_output = self.model.cpu()(**encoded)
                        embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        embeddings_np = embeddings.numpy().astype(np.float32)
                        all_embeddings.append(embeddings_np)
                    
                    # Move model back to GPU
                    self.model = self.model.to(self.device)
                else:
                    raise
        
        # Concatenate all batches
        return np.vstack(all_embeddings)


# ============================================================================
# HIERARCHICAL EMBEDDINGS
# ============================================================================

def create_section_summaries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create hierarchical section summaries from chunks.
    
    Groups chunks by section_title and creates summary embeddings.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of section summary dictionaries
    """
    # Group by section
    sections = {}
    for chunk in chunks:
        section_title = chunk.get('section_title', 'Unknown')
        if section_title not in sections:
            sections[section_title] = []
        sections[section_title].append(chunk)
    
    # Create summaries
    section_summaries = []
    for section_title, section_chunks in sections.items():
        # Concatenate all summaries in section
        combined_text = " ".join([c['summary'] for c in section_chunks])
        
        section_summaries.append({
            'section_id': f"section_{len(section_summaries)}",
            'section_title': section_title,
            'text': combined_text,
            'num_chunks': len(section_chunks),
            'metadata': section_chunks[0]['metadata'] if section_chunks else {}
        })
    
    return section_summaries


# ============================================================================
# MAIN EMBEDDING PIPELINE
# ============================================================================

def generate_embeddings_from_json(
    input_json: Path,
    output_dir: Path,
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    batch_size: int = 8,
    gpu_memory_limit: Optional[float] = None,
    include_hierarchical: bool = True
) -> Dict[str, Path]:
    """
    Generate embeddings from ETL JSON output.
    
    Args:
        input_json: Path to ETL JSON file
        output_dir: Directory to save embeddings
        model_name: BGE-M3 model name
        device: Device to use ('cpu', 'cuda', or None for auto)
        batch_size: Batch size for encoding
        gpu_memory_limit: GPU memory limit in GB
        include_hierarchical: Generate hierarchical section embeddings
        
    Returns:
        Dictionary with paths to saved files
    """
    # Load JSON data
    print(f"Loading data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    
    # Handle both standard (table_rows) and advanced (table_chunks) formats
    if 'table_rows' in data and data['table_rows']:
        table_items = data['table_rows']
        # Detect format by checking first item's structure
        first_item = table_items[0]
        if 'metadata' in first_item and first_item['metadata'].get('chunk_type') == 'table_chunk':
            # Advanced format: unified table chunks with contextual descriptions
            table_format = 'advanced'
            print(f"✅ Loaded {len(chunks)} chunks and {len(table_items)} table chunks (advanced format)")
        elif 'context_paragraph' in first_item:
            # Standard format: row-by-row table processing
            table_format = 'standard'
            print(f"✅ Loaded {len(chunks)} chunks and {len(table_items)} table rows (standard format)")
        else:
            # Fallback: use text field if available
            table_format = 'advanced'
            print(f"✅ Loaded {len(chunks)} chunks and {len(table_items)} table items (detected format)")
    elif 'table_chunks' in data:
        # Explicit advanced format key
        table_items = data['table_chunks']
        table_format = 'advanced'
        print(f"✅ Loaded {len(chunks)} chunks and {len(table_items)} table chunks (advanced format)")
    else:
        # Fallback: no table data
        table_items = []
        table_format = 'none'
        print(f"✅ Loaded {len(chunks)} chunks (no table data)")
    
    # Initialize embedder
    embedder = BGE_M3_Embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        gpu_memory_limit=gpu_memory_limit
    )
    
    output_files = {}
    
    # ----------------------------------------------------------------
    # Embed paragraph chunks
    # ----------------------------------------------------------------
    print("\n🔹 Embedding paragraph chunks...")
    chunk_texts = [c['text'] for c in chunks]
    chunk_embeddings = embedder.encode(chunk_texts)
    
    # Save chunk embeddings
    chunk_emb_path = output_dir / "chunk_embeddings.npy"
    np.save(chunk_emb_path, chunk_embeddings)
    print(f"✅ Saved chunk embeddings: {chunk_emb_path}")
    output_files['chunk_embeddings'] = chunk_emb_path
    
    # Save chunk metadata
    chunk_meta_path = output_dir / "chunk_metadata.json"
    with open(chunk_meta_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved chunk metadata: {chunk_meta_path}")
    output_files['chunk_metadata'] = chunk_meta_path
    
    # ----------------------------------------------------------------
    # Embed table items (format-agnostic)
    # ----------------------------------------------------------------
    if table_items:
        if table_format == 'standard':
            # Standard format: prepend context to text fallback
            print("\n🔹 Embedding table rows (standard format)...")
            table_texts = [
                f"{row['context_paragraph']} {row['text_fallback']}" 
                for row in table_items
            ]
        elif table_format == 'advanced':
            # Advanced format: use the combined chunk text directly
            print("\n🔹 Embedding table chunks (advanced format)...")
            table_texts = [chunk['text'] for chunk in table_items]
        else:
            table_texts = []
        
        if table_texts:
            table_embeddings = embedder.encode(table_texts)
            
            # Save table embeddings
            table_emb_path = output_dir / "table_embeddings.npy"
            np.save(table_emb_path, table_embeddings)
            print(f"✅ Saved table embeddings: {table_emb_path}")
            output_files['table_embeddings'] = table_emb_path
            
            # Save table metadata
            table_meta_path = output_dir / "table_metadata.json"
            with open(table_meta_path, 'w', encoding='utf-8') as f:
                json.dump(table_items, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved table metadata: {table_meta_path}")
            output_files['table_metadata'] = table_meta_path
    else:
        print("\n⚠️  No table data to embed")
    
    # ----------------------------------------------------------------
    # Hierarchical section embeddings
    # ----------------------------------------------------------------
    if include_hierarchical:
        print("\n🔹 Creating hierarchical section embeddings...")
        section_summaries = create_section_summaries(chunks)
        section_texts = [s['text'] for s in section_summaries]
        section_embeddings = embedder.encode(section_texts)
        
        # Save section embeddings
        section_emb_path = output_dir / "section_embeddings.npy"
        np.save(section_emb_path, section_embeddings)
        print(f"✅ Saved section embeddings: {section_emb_path}")
        output_files['section_embeddings'] = section_emb_path
        
        # Save section metadata
        section_meta_path = output_dir / "section_metadata.json"
        with open(section_meta_path, 'w', encoding='utf-8') as f:
            json.dump(section_summaries, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved section metadata: {section_meta_path}")
        output_files['section_metadata'] = section_meta_path
    
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"Chunk embeddings shape: {chunk_embeddings.shape}")
    print(f"Table embeddings shape: {table_embeddings.shape}")
    if include_hierarchical:
        print(f"Section embeddings shape: {section_embeddings.shape}")
    print(f"Total embeddings: {chunk_embeddings.shape[0] + table_embeddings.shape[0]}")
    print(f"Embedding dimension: {chunk_embeddings.shape[1]}")
    print(f"Device used: {embedder.device}")
    print("=" * 60)
    
    return output_files


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for embedding generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BGE-M3 embeddings")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON from ETL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Model name")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", 
                       help="Device to use (auto=detect)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--gpu-memory-limit", type=float, default=2.5,
                       help="GPU memory limit in GB (default: 2.5)")
    parser.add_argument("--no-hierarchical", action="store_true", 
                       help="Disable hierarchical embeddings")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = None if args.device == "auto" else args.device
    
    # Generate embeddings
    output_files = generate_embeddings_from_json(
        input_json=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        gpu_memory_limit=args.gpu_memory_limit if device == "cuda" or device is None else None,
        include_hierarchical=not args.no_hierarchical
    )
    
    print("\n✅ All embeddings generated successfully!")


if __name__ == "__main__":
    main()