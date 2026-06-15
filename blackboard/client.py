"""
blackboard/client.py — Async Redis connection factory.

Provides a lazy singleton async Redis client shared across all components.
Uses redis.asyncio for non-blocking I/O compatible with asyncio-based agents.
"""

import asyncio
import time
from config import REDIS_URL

# Try to use redis.asyncio when available; otherwise fall back to a tiny
# in-memory async-compatible Redis-like client for testing and demos.
try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    aioredis = None


class _InMemoryRedis:
    def __init__(self):
        self._store: dict[str, str] = {}
        # key -> monotonic deadline (seconds). Absent = no expiry.
        self._expiry: dict[str, float] = {}
        self._pubsubs: dict[str, list[asyncio.Queue]] = {}

    def pubsub(self):
        return _InMemoryPubSub(self)

    def _is_expired(self, key: str) -> bool:
        deadline = self._expiry.get(key)
        if deadline is not None and time.monotonic() >= deadline:
            # Lazy eviction on access — mirrors Redis TTL semantics closely
            # enough for crash-safety (a dead owner's lock disappears).
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._expiry[key] = time.monotonic() + ttl

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        # Honour NX against a *live* key (an expired key must not block SETNX).
        if nx and key in self._store and not self._is_expired(key):
            return False
        self._store[key] = value
        if ex is not None:
            self._expiry[key] = time.monotonic() + ex
        else:
            self._expiry.pop(key, None)
        return True

    async def get(self, key: str):
        if self._is_expired(key):
            return None
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    async def expire(self, key: str, seconds: int) -> None:
        # Renew/extend TTL (used by the owner heartbeat to keep its lock alive).
        if key in self._store:
            self._expiry[key] = time.monotonic() + seconds

    async def flushdb(self) -> None:
        # Clear all keys/TTLs (mirrors Redis FLUSHDB). Pub/Sub subscriptions
        # are left intact, matching real Redis semantics.
        self._store.clear()
        self._expiry.clear()

    async def publish(self, channel: str, message: str) -> int:
        queues = self._pubsubs.get(channel, [])
        for q in queues:
            await q.put({"data": message})
        return len(queues)

    async def aclose(self) -> None:
        self._store.clear()
        self._pubsubs.clear()


class _InMemoryPubSub:
    def __init__(self, redis: _InMemoryRedis):
        self._redis = redis
        self._queues: dict[str, asyncio.Queue] = {}

    async def subscribe(self, channel: str) -> None:
        q = asyncio.Queue()
        self._queues[channel] = q
        self._redis._pubsubs.setdefault(channel, []).append(q)

    async def unsubscribe(self, channel: str) -> None:
        q = self._queues.get(channel)
        if not q:
            return
        lst = self._redis._pubsubs.get(channel, [])
        if q in lst:
            lst.remove(q)
        del self._queues[channel]

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 1.0):
        # Wait for a message with timeout
        try:
            msg = await asyncio.wait_for(self._queues[next(iter(self._queues))].get(), timeout=timeout)
            return msg
        except Exception:
            return None

    async def aclose(self) -> None:
        for ch in list(self._queues.keys()):
            await self.unsubscribe(ch)


_redis_client = None


async def get_redis():
    """Return a shared async Redis client, or an in-memory fallback if Redis is unreachable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if aioredis is None:
        _redis_client = _InMemoryRedis()
        return _redis_client

    # Attempt to create a real Redis client and ping it; fall back on failure
    client = aioredis.from_url(REDIS_URL, decode_responses=True, encoding="utf-8")
    try:
        await client.ping()
        _redis_client = client
    except Exception:
        _redis_client = _InMemoryRedis()
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            # In-memory client has aclose too; ignore errors
            pass
        _redis_client = None
