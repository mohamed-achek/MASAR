"""
Shared embedding service using BGE-M3 model.
Implements singleton pattern to avoid loading model multiple times.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np


class SharedEmbeddingService:
    """Singleton embedding service using BGE-M3."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name: str = "BAAI/bge-m3", device: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        # Only initialize once
        if self._initialized:
            return
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading BGE-M3 model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._initialized = True
        print(f"✅ BGE-M3 model loaded successfully")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = self.model(**encoded)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling over token embeddings."""
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        
        # Sum attention mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
