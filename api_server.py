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
import sys
import os

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

# Mock database (replace with real database in production)
fake_users_db = {}

# Initialize RAG system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on application startup."""
    global rag_pipeline
    
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
            
            # Initialize Elasticsearch client
            try:
                from elasticsearch import Elasticsearch
                es_client = Elasticsearch(
                    ["http://localhost:9200"],
                    verify_certs=False,
                    request_timeout=30
                )
                # Test connection
                es_info = es_client.info()
                print(f"✅ Connected to Elasticsearch v{es_info['version']['number']}")
            except Exception as es_error:
                print(f"⚠️  Elasticsearch connection failed: {es_error}")
                print("   Continuing with FAISS-only mode...")
                es_client = None
            
            # Initialize retriever
            retriever = HybridRetriever(
                index_dir=index_dir,
                embeddings_dir=embeddings_dir,
                encoder=encoder,
                reranker=reranker,
                es_client=es_client
            )
            
            # Initialize LLM interface
            llm = LLMInterface(
                provider="ollama",
                model="llama3.2",
                api_key=None
            )
            
            # Create RAG pipeline
            rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)
            
            search_mode = "Hybrid (FAISS + Elasticsearch)" if es_client else "FAISS only"
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

def get_user(email: str) -> Optional[User]:
    """Get user from database"""
    if email in fake_users_db:
        user_dict = fake_users_db[email]
        return User(**user_dict)
    return None

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
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
    
    user = get_user(email=email)
    if user is None:
        raise credentials_exception
    return user


# ==================== Authentication Endpoints ====================

@app.post("/api/auth/signup", response_model=Token)
async def signup(user_data: UserCreate):
    """Sign up a new user"""
    # Check if user already exists
    if user_data.email in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = {
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hashed_password
    }
    fake_users_db[user_data.email] = user
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": user_data.email,
            "name": user_data.name
        }
    }

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login user"""
    user = get_user(user_data.email)
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": user.email,
            "name": user.name
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
        print(f"DEBUG - RAG result: {result}")
        
        # Transform citations to sources format
        sources = []
        for citation in result.get("citations", []):
            sources.append({
                "id": citation.get("id", ""),
                "text": citation.get("text", ""),
                "university": citation.get("university_id", ""),
                "program": citation.get("program", ""),
                "section": citation.get("section", ""),
                "score": citation.get("score", 0.0)
            })
        
        # Transform result to match RAGResponse model
        response = {
            "question": query.question,
            "answer": result.get("answer", ""),
            "sources": sources,  # Map citations to sources
            "confidence": result.get("confidence", 0.0),
            "language": query.language,
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
            "health": "/api/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
