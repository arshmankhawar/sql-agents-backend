"""
blackboard/query_registry.py — Atomic Query Registry with In-Flight Coordination.

This is the heart of the Blackboard system. It handles:

  1. Atomic ownership registration (Redis SETNX) — exactly one agent owns a query.
  2. In-flight tracking with TTL — if the owning agent crashes, the lock auto-expires.
  3. Subscriber notification via Redis Pub/Sub — waiting agents are woken when results arrive.
  4. Heartbeat renewal — owner agents periodically refresh their TTL.

Redis key scheme:
    registry:<query_hash>       → JSON metadata (status, owner, created_at)
    pubsub channel: query_done:<query_hash>

Status values:
    "running"   → owned by an agent, executing the DB query
    "completed" → result is available in the result cache
"""

import asyncio
import json
import logging
import time
from typing import Any

from blackboard.client import get_redis
from blackboard.result_cache import cache_get, cache_set
from config import QUERY_TTL_SECONDS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _registry_key(query_hash: str) -> str:
    return f"registry:{query_hash}"

def _pubsub_channel(query_hash: str) -> str:
    return f"query_done:{query_hash}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def try_claim_query(query_hash: str, agent_id: str) -> bool:
    """
    Attempt to atomically claim ownership of a query using Redis SETNX.

    The entry is created with a TTL so a crashed agent cannot block others
    indefinitely. Only one agent will succeed; all others return False.

    Args:
        query_hash: SHA-256 hash of the normalised SQL.
        agent_id:   Unique identifier of the claiming agent.

    Returns:
        True  → this agent is the owner and must execute the query.
        False → another agent already owns it; caller should subscribe and wait.
    """
    redis = await get_redis()
    key = _registry_key(query_hash)
    metadata = json.dumps({
        "status": "running",
        "owner": agent_id,
        "created_at": time.time(),
    })
    # NX = only set if key does Not eXist  (atomic compare-and-set)
    # EX = expire after QUERY_TTL_SECONDS  (crash-safety)
    acquired = await redis.set(key, metadata, nx=True, ex=QUERY_TTL_SECONDS)
    if acquired:
        logger.info("[Registry] Agent %s CLAIMED  hash=%s", agent_id, query_hash[:12])
    else:
        logger.info("[Registry] Agent %s WAITING  hash=%s (already owned)", agent_id, query_hash[:12])
    return bool(acquired)


async def renew_heartbeat(query_hash: str) -> None:
    """
    Refresh the TTL of a running registry entry.

    Should be called periodically by the owning agent during long queries
    to prevent the TTL from expiring while the query is still executing.
    """
    redis = await get_redis()
    key = _registry_key(query_hash)
    await redis.expire(key, QUERY_TTL_SECONDS)


async def complete_query(query_hash: str, result: list[dict[str, Any]]) -> None:
    """
    Mark a query as completed: persist the result and notify all subscribers.

    Steps:
      1. Store result in the shared result cache (with its own long TTL).
      2. Update registry entry status to "completed".
      3. Publish to the Pub/Sub channel so waiting agents wake up.
      4. Delete the short-lived registry key (result cache is the durable store).

    Args:
        query_hash: SHA-256 hash of the normalised SQL.
        result:     List of row dicts from the database.
    """
    redis = await get_redis()

    # 1. Persist result in the result cache (long TTL)
    await cache_set(query_hash, result)

    # 2. Publish completion signal — all subscribers will receive this
    channel = _pubsub_channel(query_hash)
    await redis.publish(channel, json.dumps({"status": "completed", "query_hash": query_hash}))

    # 3. Remove the short-lived registry key — result cache is now the source of truth
    await redis.delete(_registry_key(query_hash))

    logger.info("[Registry] COMPLETED hash=%s  rows=%d", query_hash[:12], len(result))


async def wait_for_result(query_hash: str, timeout: float = 120.0) -> list[dict[str, Any]] | None:
    """
    Subscribe to the Pub/Sub channel and block until the owning agent completes.

    First checks the result cache in case the result arrived before we subscribed
    (handles the race between completion and subscription setup).

    Args:
        query_hash: SHA-256 hash of the normalised SQL.
        timeout:    Maximum seconds to wait before giving up.

    Returns:
        List of row dicts on success, or None on timeout.
    """
    # Fast path: result may already be in cache (completed just before we subscribed)
    cached = await cache_get(query_hash)
    if cached is not None:
        logger.info("[Registry] Fast-path cache hit for hash=%s", query_hash[:12])
        return cached

    redis = await get_redis()
    channel = _pubsub_channel(query_hash)

    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)

    logger.info("[Registry] Agent subscribed to channel=%s (timeout=%.0fs)", channel, timeout)

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Poll with a short timeout so we don't block forever
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is not None:
                # Woken by the owning agent — fetch result from cache
                result = await cache_get(query_hash)
                if result is not None:
                    logger.info("[Registry] Subscriber received result for hash=%s", query_hash[:12])
                    return result
            # Also poll the cache directly in case we missed the pub/sub message
            cached = await cache_get(query_hash)
            if cached is not None:
                return cached
            await asyncio.sleep(0.1)

        logger.warning("[Registry] TIMEOUT waiting for hash=%s", query_hash[:12])
        return None
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()


async def get_registry_status(query_hash: str) -> dict | None:
    """Return the raw registry entry for a query, or None if not present."""
    redis = await get_redis()
    raw = await redis.get(_registry_key(query_hash))
    return json.loads(raw) if raw else None
