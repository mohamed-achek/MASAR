"""
Embedding Microservice - Dedicated pod for BGE-M3 model.
Separates heavy embedding model from main API server to optimize memory usage.
"""

import os
import torch
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel


# ============== Configuration ==============

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
PORT = int(os.getenv("PORT", "8001"))


# ============== Request/Response Models ==============

class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str] = Field(..., min_length=1, description="List of texts to embed")
    normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings")
    max_length: int = Field(default=512, le=8192, description="Maximum sequence length")


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used")
    dimension: int = Field(..., description="Embedding dimension")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str


# ============== Embedding Service ==============

class EmbeddingService:
    """Service class for managing BGE-M3 model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self._initialized = False
    
    def load_model(self):
        """Load the embedding model."""
        if self._initialized:
            return
        
        print(f"🔄 Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._initialized = True
        print(f"✅ Model loaded successfully on {self.device}")
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        max_length: int = MAX_LENGTH
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        if not self._initialized:
            raise RuntimeError("Model not loaded")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch_texts = texts[i:i + MAX_BATCH_SIZE]
                
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
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            return 0
        return self.model.config.hidden_size
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============== Global Service Instance ==============

embedding_service = EmbeddingService()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    embedding_service.load_model()
    yield
    # Shutdown
    embedding_service.cleanup()


app = FastAPI(
    title="Masar Embedding Service",
    description="Dedicated microservice for BGE-M3 embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if embedding_service._initialized else "loading",
        model_loaded=embedding_service._initialized,
        device=embedding_service.device,
        model_name=embedding_service.model_name
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if not embedding_service._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    Generate embeddings for a list of texts.
    
    - **texts**: List of text strings to embed
    - **normalize**: Whether to L2 normalize the embeddings (default: True)
    - **max_length**: Maximum token length per text (default: 512)
    """
    if not embedding_service._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        embeddings = embedding_service.encode(
            texts=request.texts,
            normalize=request.normalize,
            max_length=request.max_length
        )
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=embedding_service.model_name,
            dimension=embedding_service.embedding_dim
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model_name": embedding_service.model_name,
        "device": embedding_service.device,
        "embedding_dimension": embedding_service.embedding_dim,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_length": MAX_LENGTH
    }


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
