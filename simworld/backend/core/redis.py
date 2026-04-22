"""Redis async client — with in-memory fallback when Redis is unavailable.

If redis package is installed and server is reachable, uses real Redis.
Otherwise, provides an in-memory dict-based fallback so the app still runs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _MemoryCache:
    """Simple in-memory fallback when Redis is not available."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._channels: dict[str, list] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def publish(self, channel: str, message: str) -> int:
        logger.debug("MemoryCache publish: %s -> %s", channel, message[:50])
        return 0

    async def aclose(self) -> None:
        self._store.clear()

    async def ping(self) -> bool:
        return True

    # ── Set operations (for training_active tracking) ─────────────
    async def sadd(self, key: str, *values: str) -> int:
        if key not in self._store:
            self._store[key] = set()
        s = self._store[key]
        if not isinstance(s, set):
            s = set()
            self._store[key] = s
        added = 0
        for v in values:
            if v not in s:
                s.add(v)
                added += 1
        return added

    async def srem(self, key: str, *values: str) -> int:
        s = self._store.get(key)
        if not isinstance(s, set):
            return 0
        removed = 0
        for v in values:
            if v in s:
                s.discard(v)
                removed += 1
        return removed

    async def sismember(self, key: str, value: str) -> bool:
        s = self._store.get(key)
        if not isinstance(s, set):
            return False
        return value in s

    async def smembers(self, key: str) -> set:
        s = self._store.get(key)
        if not isinstance(s, set):
            return set()
        return s.copy()


def _create_client() -> Any:
    """Try to create a real Redis client, fall back to in-memory."""
    try:
        import redis.asyncio as aioredis
        from core.config import get_settings
        settings = get_settings()

        client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=50,
        )
        logger.info("Redis client created (url=%s)", settings.REDIS_URL)
        return client
    except (ImportError, Exception) as exc:
        logger.warning("Redis unavailable (%s) — using in-memory cache", exc)
        return _MemoryCache()


redis_client = _create_client()


async def get_redis() -> Any:
    """Return the shared Redis client (or memory fallback)."""
    return redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection pool."""
    try:
        await redis_client.aclose()
    except Exception:
        pass


async def publish_message(channel: str, message: str) -> int:
    """Publish a message to a Redis channel. Returns subscriber count."""
    try:
        return await redis_client.publish(channel, message)
    except Exception:
        return 0


async def get_cached(key: str) -> str | None:
    """Retrieve a cached value by key."""
    try:
        return await redis_client.get(key)
    except Exception:
        return None


async def set_cached(key: str, value: str, ttl: int = 3600) -> None:
    """Store a value in cache with a TTL in seconds."""
    try:
        await redis_client.set(key, value, ex=ttl)
    except Exception:
        pass
