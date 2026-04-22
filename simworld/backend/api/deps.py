"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis import get_redis


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async for session in get_db():
        yield session


async def get_redis_client():
    """Return the Redis client."""
    return await get_redis()
