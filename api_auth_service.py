"""
Auth Service - Handles authentication and user management
Lightweight microservice without heavy ML dependencies
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext
from jose import JWTError, jwt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    async_init_db as init_db, get_db, User, UserCRUD, AsyncSessionLocal
)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Security
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    hostname: str
    components: dict
    version: str


# Auth functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
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
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = await UserCRUD.get_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    print("🔄 Initializing Auth Service...")
    
    # Initialize database
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Initializing database... (attempt {attempt}/{max_retries})")
            await init_db()
            print("✅ PostgreSQL database initialized")
            
            # Verify default user exists
            async with AsyncSessionLocal() as session:
                admin = await UserCRUD.get_by_email(session, "admin@masar.tn")
                if admin:
                    print("✅ Default user exists: admin@masar.tn")
                else:
                    # Create default admin user
                    hashed_pw = get_password_hash("admin123")
                    admin = await UserCRUD.create(
                        session,
                        email="admin@masar.tn",
                        username="admin",
                        hashed_password=hashed_pw,
                        full_name="Admin User"
                    )
                    print("✅ Created default admin user: admin@masar.tn")
            
            print("✅ Database ready")
            break
        except Exception as e:
            print(f"❌ Database initialization failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                print("❌ Max retries reached. Starting without database.")
            else:
                import asyncio
                await asyncio.sleep(2)
    
    print("✅ Auth Service initialized")
    
    yield
    
    print("🔄 Shutting down Auth Service...")


# Create FastAPI app
app = FastAPI(
    title="Masar Auth Service",
    description="Authentication and user management microservice",
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
@app.get("/api/auth/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import socket
    from database import engine
    from sqlalchemy import text
    
    components = {
        "database": "unknown"
    }
    
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["database"] = "connected"
    except Exception as e:
        components["database"] = f"error: {str(e)[:50]}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "components": components,
        "version": "1.0.0"
    }


# Auth endpoints
@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    existing_user = await UserCRUD.get_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user with hashed password
    hashed_password = get_password_hash(user_data.password)
    username = user_data.email.split("@")[0]  # Use email prefix as username
    user = await UserCRUD.create(
        db,
        email=user_data.email,
        username=username,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    return user


@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    # form_data.username is the email
    user = await UserCRUD.get_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user


@app.get("/api/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)"""
    from sqlalchemy import select
    
    result = await db.execute(select(User))
    users = result.scalars().all()
    return users


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
