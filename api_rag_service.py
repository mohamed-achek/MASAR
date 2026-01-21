"""
RAG Service - Handles RAG query processing
Uses embedder and reranker microservices
"""
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Global RAG pipeline
rag_pipeline = None


# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    hostname: str
    components: dict
    version: str


# Auth dependency (JWT verification only, no database)
async def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    """Verify JWT token and extract user email"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return email
    except JWTError:
        raise credentials_exception


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global rag_pipeline
    
    print("🔄 Initializing RAG Service...")
    
    # Configure Python path
    import sys
    project_root = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.join(project_root, "pipeline")
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    
    print(f"📂 Python path configured:")
    print(f"   - Project root: {project_root}")
    print(f"   - Pipeline dir: {pipeline_dir}")
    
    # Import RAG modules
    print("🔄 Importing RAG modules...")
    try:
        from pipeline.retrieve import QueryEncoder, HybridRetriever
        from pipeline.rag_new import RAGPipeline
        
        print("✅ RAG modules imported successfully")
        print(f"   - QueryEncoder: {QueryEncoder}")
        print(f"   - HybridRetriever: {HybridRetriever}")
        print(f"   - RAGPipeline: {RAGPipeline}")
    except Exception as e:
        print(f"❌ Error importing RAG modules: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  RAG Service will not be available")
        yield
        return
    
    # Initialize RAG system
    print("🔄 Initializing RAG system...")
    try:
        # Check environment variables
        use_embedder_service = os.getenv("USE_EMBEDDER_SERVICE", "true").lower() == "true"
        use_reranker_service = os.getenv("USE_RERANKER_SERVICE", "true").lower() == "true"
        
        if use_embedder_service:
            print("🔗 Using embedder microservice for query encoding")
        if use_reranker_service:
            print("🔗 Using reranker microservice")
        
        # Initialize query encoder
        encoder = QueryEncoder()
        print("✅ Query encoder (remote) configured")
        
        # Initialize retriever
        retriever = HybridRetriever(query_encoder=encoder)
        print("✅ Retriever initialized")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(retriever=retriever)
        print("✅ RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        rag_pipeline = None
        print("⚠️  RAG Service will not be available")
    
    print("✅ RAG Service initialized")
    
    yield
    
    print("🔄 Shutting down RAG Service...")
    rag_pipeline = None


# Create FastAPI app
app = FastAPI(
    title="Masar RAG Service",
    description="RAG query processing microservice",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import socket
    import httpx
    
    components = {
        "rag_pipeline": "available" if rag_pipeline else "unavailable",
        "embedder": "unknown",
        "reranker": "unknown"
    }
    
    # Check embedder service
    try:
        embedder_url = os.getenv("EMBEDDER_SERVICE_URL", "http://masar-embedder:8001")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{embedder_url}/health")
            if resp.status_code == 200:
                components["embedder"] = "connected"
            else:
                components["embedder"] = f"error: status {resp.status_code}"
    except Exception as e:
        components["embedder"] = f"error: {str(e)[:30]}"
    
    # Check reranker service
    try:
        reranker_url = os.getenv("RERANKER_SERVICE_URL", "http://masar-reranker:8002")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{reranker_url}/health")
            if resp.status_code == 200:
                components["reranker"] = "connected"
            else:
                components["reranker"] = f"error: status {resp.status_code}"
    except Exception as e:
        components["reranker"] = f"error: {str(e)[:30]}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "components": components,
        "version": "1.0.0"
    }


# RAG query endpoint
@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user_email: str = Depends(get_current_user_email)
):
    """Process RAG query"""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not available"
        )
    
    try:
        result = await rag_pipeline.query(
            question=request.question,
            top_k=request.top_k
        )
        
        response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "metadata": {
                "question": request.question,
                "top_k": request.top_k,
                "user": user_email,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if request.include_context:
            response["metadata"]["context"] = result.get("context", "")
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
