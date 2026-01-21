"""
Lightweight Authentication Microservice for MASAR
Handles user authentication, JWT tokens, and user management.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import asyncpg
from contextlib import asynccontextmanager

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://masar:masar123@masar-postgresql:5432/masar")

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None


# Pydantic models
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection pool lifecycle."""
    global db_pool
    print("🔄 Connecting to database...")
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    print("✅ Database connection pool created")
    
    # Create default admin user if not exists
    try:
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1",
                "admin@masar.tn"
            )
            if not user:
                hashed_password = pwd_context.hash("admin123")
                await conn.execute(
                    """INSERT INTO users (email, full_name, hashed_password, is_active)
                       VALUES ($1, $2, $3, $4)""",
                    "admin@masar.tn", "Admin User", hashed_password, True
                )
                print("✅ Default admin user created")
    except Exception as e:
        print(f"⚠️  Could not create default user: {e}")
    
    yield
    
    print("🔄 Closing database connection pool...")
    await db_pool.close()
    print("✅ Database connection pool closed")


# FastAPI app
app = FastAPI(
    title="Masar Auth Service",
    description="Authentication microservice for MASAR platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user from JWT token."""
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
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, full_name, is_active, created_at FROM users WHERE email = $1",
            token_data.email
        )
    
    if user is None:
        raise credentials_exception
    
    return User(**dict(user))


# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return {
        "status": "healthy" if db_status == "connected" else "unhealthy",
        "service": "auth",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/auth/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """Register a new user."""
    async with db_pool.acquire() as conn:
        # Check if user already exists
        existing_user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            user.email
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = await conn.fetchrow(
            """INSERT INTO users (email, full_name, hashed_password, is_active)
               VALUES ($1, $2, $3, $4)
               RETURNING id, email, full_name, is_active, created_at""",
            user.email, user.full_name, hashed_password, True
        )
    
    return User(**dict(new_user))


@app.post("/api/auth/login", response_model=Token)
async def login(user_login: UserLogin):
    """Login and get access token."""
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, hashed_password, is_active FROM users WHERE email = $1",
            user_login.email
        )
    
    if not user or not verify_password(user_login.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login."""
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, hashed_password, is_active FROM users WHERE email = $1",
            form_data.username
        )
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user


@app.post("/api/auth/verify")
async def verify_token(token: str = Depends(oauth2_scheme)):
    """Verify a JWT token and return user info."""
    user = await get_current_user(token)
    return {
        "valid": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
