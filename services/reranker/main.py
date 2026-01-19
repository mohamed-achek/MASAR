"""
Reranker Microservice - Dedicated pod for BGE-Reranker-Large model.
Separates heavy reranking model from main API server to optimize memory usage.
"""

import os
import torch
from typing import List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


# ============== Configuration ==============

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU for reranker
PORT = int(os.getenv("PORT", "8002"))


# ============== Request/Response Models ==============

class Document(BaseModel):
    """Document model for reranking."""
    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class RerankRequest(BaseModel):
    """Request model for reranking."""
    query: str = Field(..., min_length=1, description="Search query")
    documents: List[Document] = Field(..., min_length=1, description="List of documents to rerank")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top documents to return")


class RerankResponse(BaseModel):
    """Response model for reranking."""
    results: List[Dict[str, Any]] = Field(..., description="Reranked documents with scores")
    model: str = Field(..., description="Model name used")


class ScorePairsRequest(BaseModel):
    """Request for scoring query-document pairs."""
    pairs: List[Tuple[str, str]] = Field(..., description="List of (query, document) pairs")


class ScorePairsResponse(BaseModel):
    """Response for pair scoring."""
    scores: List[float] = Field(..., description="Relevance scores for each pair")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str


# ============== Reranker Service ==============

class RerankerService:
    """Service class for managing BGE-Reranker model."""
    
    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self._initialized = False
    
    def load_model(self):
        """Load the reranker model."""
        if self._initialized:
            return
        
        print(f"🔄 Loading {self.model_name} on {self.device}...")
        
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        self._initialized = True
        print(f"✅ Model loaded successfully on {self.device}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance."""
        if not self._initialized:
            raise RuntimeError("Model not loaded")
        
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc.text] for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Build results with scores
        results = []
        for doc, score in zip(documents, scores):
            result = {
                "text": doc.text,
                "metadata": doc.metadata,
                "rerank_score": float(score)
            }
            results.append(result)
        
        # Sort by score (descending) and return top k
        results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs."""
        if not self._initialized:
            raise RuntimeError("Model not loaded")
        
        scores = self.model.predict(pairs)
        return scores.tolist()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============== Global Service Instance ==============

reranker_service = RerankerService()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    reranker_service.load_model()
    yield
    # Shutdown
    reranker_service.cleanup()


app = FastAPI(
    title="Masar Reranker Service",
    description="Dedicated microservice for BGE-Reranker-Large",
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
        status="healthy" if reranker_service._initialized else "loading",
        model_loaded=reranker_service._initialized,
        device=reranker_service.device,
        model_name=reranker_service.model_name
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if not reranker_service._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on query relevance.
    
    - **query**: The search query
    - **documents**: List of documents with text and optional metadata
    - **top_k**: Number of top results to return (default: 5)
    """
    if not reranker_service._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        results = reranker_service.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        return RerankResponse(
            results=results,
            model=reranker_service.model_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.post("/score", response_model=ScorePairsResponse)
async def score_pairs(request: ScorePairsRequest):
    """
    Score query-document pairs.
    
    - **pairs**: List of (query, document) tuples to score
    """
    if not reranker_service._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        scores = reranker_service.score_pairs(request.pairs)
        return ScorePairsResponse(scores=scores)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model_name": reranker_service.model_name,
        "device": reranker_service.device
    }


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
