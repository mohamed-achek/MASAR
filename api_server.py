"""
FastAPI Backend for Masar Application

Provides authentication and RAG query endpoints
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import sys
import os

# Import database module
from database import (
    get_db,
    init_db,
    create_default_user,
    get_user_by_email,
    create_user as db_create_user,
    User as DBUser
)

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(project_root, 'pipeline')

# Add both directories to path
for path in [project_root, pipeline_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"📂 Python path configured:")
print(f"   - Project root: {project_root}")
print(f"   - Pipeline dir: {pipeline_dir}")

# Try to import RAG system from pipeline
RAG_AVAILABLE = False
rag_pipeline = None
QueryEncoder = None
Reranker = None
HybridRetriever = None
RAGPipeline = None
LLMInterface = None

try:
    from pathlib import Path
    print("🔄 Importing RAG modules...")
    
    # Import with full context
    import sys
    sys.path.insert(0, pipeline_dir)
    
    from retrieve import QueryEncoder as QE, Reranker as RR, HybridRetriever as HR
    from rag_new import RAGPipeline as RP, LLMInterface as LI
    
    QueryEncoder = QE
    Reranker = RR
    HybridRetriever = HR
    RAGPipeline = RP
    LLMInterface = LI
    
    RAG_AVAILABLE = True
    print("✅ RAG modules imported successfully")
    print(f"   - QueryEncoder: {QueryEncoder}")
    print(f"   - HybridRetriever: {HybridRetriever}")
    print(f"   - RAGPipeline: {RAGPipeline}")
except Exception as e:
    print(f"❌ Error importing RAG system: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    RAG_AVAILABLE = False

# Import new app pipeline modules
APP_PIPELINE_AVAILABLE = False
vector_retriever = None
embedding_service = None
app_reranker = None

try:
    print("🔄 Importing app pipeline modules...")
    from app.input_handler import create_question_record
    from app.retriever import VectorRetriever, DocumentChunk
    
    # Use shared utilities instead of duplicates
    from utils.embedding_utils import SharedEmbeddingService
    from utils.reranker_utils import SharedReranker
    
    APP_PIPELINE_AVAILABLE = True
    print("✅ App pipeline modules imported successfully")
except Exception as e:
    print(f"❌ Error importing app pipeline modules: {e}")
    import traceback
    traceback.print_exc()
    APP_PIPELINE_AVAILABLE = False

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing - using argon2 as it's more modern and doesn't have bcrypt's issues
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Initialize FastAPI app
app = FastAPI(title="Masar API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG system on application startup."""
    global rag_pipeline, vector_retriever, embedding_service, app_reranker
    
    # Initialize database
    try:
        print("🔄 Initializing database...")
        init_db()
        create_default_user()
        print("✅ Database ready")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
    
    if RAG_AVAILABLE:
        try:
            from pathlib import Path
            
            # Paths to your pipeline indices and embeddings
            index_dir = Path("data/final/indices")
            embeddings_dir = Path("data/final/embeddings")
            
            if not index_dir.exists():
                print(f"⚠️  Index directory not found: {index_dir}")
                print(f"   Run: python pipeline/index.py to create indices")
                return
            
            if not embeddings_dir.exists():
                print(f"⚠️  Embeddings directory not found: {embeddings_dir}")
                print(f"   Run: python pipeline/embedding.py to create embeddings")
                return
            
            print("🔄 Initializing RAG system...")
            
            # Initialize query encoder
            encoder = QueryEncoder()
            
            # Initialize reranker
            reranker = Reranker()
            
            # Initialize retriever with Elasticsearch host
            # HybridRetriever will create its own ES connection internally
            retriever = HybridRetriever(
                index_dir=index_dir,
                embeddings_dir=embeddings_dir,
                encoder=encoder,
                reranker=reranker,
                es_host="http://localhost:9200",  # Pass host, not client
                es_user=None,  # No authentication in dev mode
                es_password=None
            )
            
            # Initialize LLM interface
            llm = LLMInterface(
                provider="ollama",
                model="llama3.2",
                api_key=None
            )
            
            # Create RAG pipeline
            rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)
            
            # Check if ES is available in retriever
            es_available = retriever.es is not None
            search_mode = "Hybrid (FAISS + Elasticsearch)" if es_available else "FAISS only"
            print("✅ RAG system initialized successfully")
            print(f"   - Index dir: {index_dir}")
            print(f"   - Embeddings dir: {embeddings_dir}")
            print(f"   - Search mode: {search_mode}")
            print(f"   - LLM: Ollama (llama3.2)")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            import traceback
            traceback.print_exc()
            rag_pipeline = None
    else:
        print("⚠️  RAG modules not available")
    
    # Initialize new app pipeline components
    if APP_PIPELINE_AVAILABLE:
        try:
            print("🔄 Initializing app pipeline components...")
            
            # Initialize shared embedding service (singleton)
            embedding_service = SharedEmbeddingService.get_instance()
            
            # Initialize vector retriever (will be populated with data later)
            vector_retriever = VectorRetriever(embedding_dim=embedding_service.embedding_dim)
            
            # Initialize shared reranker (singleton)
            app_reranker = SharedReranker.get_instance()
            
            print("✅ App pipeline components initialized")
            print(f"   - Embedding model: {embedding_service.model_name}")
            print(f"   - Embedding dim: {embedding_service.embedding_dim}")
            print(f"   - Reranker model: {app_reranker.model_name}")
            print(f"   - Memory optimization: Using shared singleton instances")
        
        except Exception as e:
            print(f"❌ Error initializing app pipeline: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  App pipeline modules not available")


# ==================== Models ====================

class User(BaseModel):
    email: EmailStr
    name: str
    hashed_password: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class RAGQuery(BaseModel):
    question: str
    language: str = "auto"
    top_k: int = 3

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    model: str

class QuestionRequest(BaseModel):
    """Request model for new pipeline question processing"""
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    top_k: int = 5
    rerank: bool = True

class QuestionResponse(BaseModel):
    """Response model for new pipeline question processing"""
    question_id: str
    timestamp: str
    original_question: str
    processed_question: str
    intent: str
    intent_confidence: float
    language: str
    results: List[Dict[str, Any]]
    processing_time: float


# ==================== Authentication Functions ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password using argon2"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token"""
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
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    db_user = get_user_by_email(db, email=email)
    if db_user is None:
        raise credentials_exception
    
    # Convert DB user to Pydantic User model
    user = User(
        email=db_user.email,
        name=db_user.full_name,
        hashed_password=db_user.hashed_password
    )
    
    return user


# ==================== Authentication Endpoints ====================

@app.post("/api/auth/signup", response_model=Token)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """Sign up a new user"""
    # Check if user already exists
    existing_user = get_user_by_email(db, email=user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user in database
    hashed_password = get_password_hash(user_data.password)
    new_user = db_create_user(
        db=db,
        email=user_data.email,
        full_name=user_data.name,
        hashed_password=hashed_password
    )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": new_user.email,
            "name": new_user.full_name
        }
    }

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    # Get user from database
    db_user = get_user_by_email(db, email=user_data.email)
    
    if not db_user or not verify_password(user_data.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is disabled
    if db_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": db_user.email,
            "name": db_user.full_name
        }
    }

@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return {
        "email": current_user.email,
        "name": current_user.name
    }


# ==================== RAG Endpoints ====================

@app.get("/api/rag/status")
async def rag_status():
    """Check if RAG system is available"""
    return {
        "available": rag_pipeline is not None,
        "message": "RAG system ready" if rag_pipeline else "RAG system not initialized"
    }

@app.post("/api/rag/query", response_model=RAGResponse)
async def query_rag(
    query: RAGQuery,
    current_user: User = Depends(get_current_user)
):
    """Query the RAG system"""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not available"
        )
    
    try:
        # Use your RAGPipeline's answer_question method
        result = rag_pipeline.answer_question(
            question=query.question,
            k=query.top_k
        )
        
        # Debug: print the result
        print(f"DEBUG - RAG result keys: {result.keys()}")
        print(f"DEBUG - Number of citations: {len(result.get('citations', []))}")
        
        # Transform citations to sources format
        sources = []
        for citation in result.get("citations", []):
            source = {
                "id": citation.get("id", ""),
                "text": citation.get("text", ""),
                "university": citation.get("university_id", ""),
                "program": citation.get("program", ""),
                "section": citation.get("section", ""),
                "type": citation.get("type", "chunk"),
                "score": citation.get("score", 0.0),
                "year": citation.get("year", ""),
                "source_file": citation.get("source_file", "")
            }
            sources.append(source)
        
        print(f"DEBUG - Number of sources: {len(sources)}")
        if sources:
            print(f"DEBUG - First source: {sources[0]}")
        
        # Transform result to match RAGResponse model
        response = {
            "question": query.question,
            "answer": result.get("answer", ""),
            "sources": sources,  # Map citations to sources with full text
            "model": "llama3.2"
        }
        
        print(f"DEBUG - Response dict: {response}")
        
        return RAGResponse(**response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# ==================== New Pipeline Endpoint ====================

@app.post("/api/questions", response_model=QuestionResponse)
async def process_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process a question using the new pipeline components.
    
    Flow:
    1. Input handling: Create question record with preprocessing
    2. Embeddings: Generate embedding for processed question
    3. Retrieval: Search vector store with optional metadata filtering
    4. Reranking: Rerank results using cross-encoder (optional)
    
    This endpoint demonstrates the new modular pipeline architecture.
    """
    import time
    start_time = time.time()
    
    if not APP_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="App pipeline not available. Install required packages: pip install langdetect sentence-transformers faiss-cpu"
        )
    
    if not embedding_service or not embedding_service.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not initialized"
        )
    
    try:
        # Step 1: Create question record (input handling + preprocessing + intent/NER)
        from app.input_handler import create_question_record
        
        question_record = create_question_record(
            question=request.question,
            user_id=request.user_id or current_user.email,
            session_id=request.session_id,
            metadata=request.metadata,
            preprocess_text=True,
            analyze_intent=True,
            mask_pii=True
        )
        
        # Step 2: Generate embedding for processed question
        query_embedding = embedding_service.get_embedding(
            question_record['processed_question']
        )
        
        if query_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding"
            )
        
        # Step 3: Retrieve documents
        # NOTE: For this demo, we'll use sample chunks if no index is loaded
        # In production, you would load the index during startup
        results = []
        
        if vector_retriever and vector_retriever.size() > 0:
            # Use existing index
            retrieved_chunks = vector_retriever.search(
                query_embedding=query_embedding,
                top_k=request.top_k * 2 if request.rerank else request.top_k,  # Get more for reranking
                metadata_filter=request.metadata  # Optional metadata filtering
            )
            
            # Step 4: Rerank (optional)
            if request.rerank and app_reranker and app_reranker.model:
                # Convert chunks to dicts for reranking
                chunks_as_dicts = [chunk.to_dict() for chunk in retrieved_chunks]
                
                # Use SharedReranker's rerank method
                reranked_dicts = app_reranker.rerank(
                    query=question_record['processed_question'],
                    documents=chunks_as_dicts,
                    top_k=request.top_k,
                    text_key='text'
                )
                results = reranked_dicts
            else:
                # Just take top_k
                retrieved_chunks = retrieved_chunks[:request.top_k]
                # Convert chunks to dict format
                results = [chunk.to_dict() for chunk in retrieved_chunks]
        else:
            # Return empty results with warning
            results = []
            print("⚠️  No vector index loaded. Returning empty results.")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = QuestionResponse(
            question_id=question_record['question_id'],
            timestamp=question_record['timestamp'],
            original_question=question_record['original_question'],
            processed_question=question_record['processed_question'],
            intent=question_record['intent'],
            intent_confidence=question_record['intent_confidence'],
            language=question_record['preprocessing'].get('language', 'unknown'),
            results=results,
            processing_time=processing_time
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


# ==================== Health Check ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "rag_available": rag_pipeline is not None
    }


# ==================== Root ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Masar API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/auth/*",
            "rag": "/api/rag/*",
            "questions": "/api/questions (new pipeline)",
            "health": "/api/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
