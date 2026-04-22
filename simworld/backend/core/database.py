"""Async SQLAlchemy engine, session factory, and Base declarative class.

Auto-detects database: uses PostgreSQL if available, falls back to SQLite.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_engine_url() -> str:
    """Determine the database URL, falling back to SQLite if PG is unreachable."""
    url = settings.DATABASE_URL

    # If using asyncpg (PostgreSQL), test connectivity isn't easy at import time.
    # If the URL contains 'asyncpg', check if asyncpg is even installed.
    if "asyncpg" in url:
        try:
            import asyncpg  # noqa: F401
            return url
        except ImportError:
            logger.warning("asyncpg not installed — falling back to SQLite")

    if "aiosqlite" in url or "sqlite" in url:
        return url

    # Default fallback to SQLite
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_root), "data")
    os.makedirs(data_dir, exist_ok=True)
    sqlite_url = f"sqlite+aiosqlite:///{data_dir}/simworld.db"
    logger.info("Using SQLite: %s", sqlite_url)
    return sqlite_url


_db_url = _get_engine_url()
_is_sqlite = "sqlite" in _db_url

engine_kwargs = {
    "echo": settings.LOG_LEVEL == "DEBUG",
}

if not _is_sqlite:
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 10
    engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(_db_url, **engine_kwargs)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (dev convenience — prefer Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the connection pool."""
    await engine.dispose()
