"""
PostgreSQL database configuration and utilities.
Async PostgreSQL database using asyncpg and SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from datetime import datetime
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://masar_user:masar_password@localhost:5432/masar_db"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for models
Base = declarative_base()


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryHistory(Base):
    """Store user query history."""
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text)
    num_sources = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session.
    
    Usage in FastAPI:
        @app.get("/")
        async def route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ PostgreSQL database initialized")


async def close_db():
    """Close database connections."""
    await engine.dispose()
    print("✅ PostgreSQL connections closed")


# ============================================================================
# MIGRATION FROM SQLITE
# ============================================================================

async def migrate_from_sqlite(sqlite_path: str = "masar.db"):
    """
    Migrate data from SQLite to PostgreSQL.
    
    Args:
        sqlite_path: Path to SQLite database file
    """
    import sqlite3
    from sqlalchemy import select
    
    # Check if SQLite file exists
    if not os.path.exists(sqlite_path):
        print(f"⚠️  SQLite file not found: {sqlite_path}")
        return
    
    print(f"🔄 Migrating data from {sqlite_path} to PostgreSQL...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    
    # Initialize PostgreSQL
    await init_db()
    
    async with AsyncSessionLocal() as session:
        try:
            # Migrate users
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            
            for row in users:
                # Check if user already exists
                result = await session.execute(
                    select(User).where(User.email == row['email'])
                )
                existing_user = result.scalar_one_or_none()
                
                if not existing_user:
                    user = User(
                        id=row['id'],
                        email=row['email'],
                        username=row['username'],
                        hashed_password=row['hashed_password'],
                        full_name=row.get('full_name'),
                        is_active=bool(row.get('is_active', 1)),
                        created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else datetime.utcnow()
                    )
                    session.add(user)
            
            await session.commit()
            print(f"✅ Migrated {len(users)} users")
            
        except Exception as e:
            print(f"⚠️  Migration error: {e}")
            await session.rollback()
    
    sqlite_conn.close()
    print("✅ Migration completed")


# ============================================================================
# CRUD OPERATIONS
# ============================================================================

from sqlalchemy import select

class UserCRUD:
    """CRUD operations for User model."""
    
    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> User | None:
        """Get user by email."""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_username(db: AsyncSession, username: str) -> User | None:
        """Get user by username."""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Get user by ID."""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create(db: AsyncSession, email: str, username: str, hashed_password: str, full_name: str = None) -> User:
        """Create new user."""
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    
    @staticmethod
    async def update(db: AsyncSession, user_id: int, **kwargs) -> User | None:
        """Update user."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            user.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(user)
        
        return user


class QueryHistoryCRUD:
    """CRUD operations for QueryHistory model."""
    
    @staticmethod
    async def create(db: AsyncSession, user_id: int, query: str, answer: str = None, num_sources: int = 0) -> QueryHistory:
        """Create query history entry."""
        history = QueryHistory(
            user_id=user_id,
            query=query,
            answer=answer,
            num_sources=num_sources
        )
        db.add(history)
        await db.commit()
        await db.refresh(history)
        return history
    
    @staticmethod
    async def get_user_history(db: AsyncSession, user_id: int, limit: int = 50):
        """Get user's query history."""
        result = await db.execute(
            select(QueryHistory)
            .where(QueryHistory.user_id == user_id)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
