"""
blackboard/result_cache.py — Shared Result Cache backed by Redis.

Stores completed SQL query results so agents and the Plot Agent can
consume them without re-executing the same database query.

Cache key scheme:
    result:<query_hash>

Value is a JSON-encoded string containing the query result rows.
"""

import json
import logging
from typing import Any

from blackboard.client import get_redis
from config import RESULT_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


async def cache_set(query_hash: str, result: list[dict[str, Any]] | dict[str, Any]) -> None:
    """
    Store query results in the shared cache.

    Args:
        query_hash: SHA-256 hash of the normalised SQL query.
        result: List of row dicts returned by the database.
    """
    redis = await get_redis()
    key = f"result:{query_hash}"
    # Normalize payload to a result dict with a 'rows' key so consumers
    # (PlotAgent, subscribers) can rely on a consistent structure.
    if isinstance(result, list):
        payload_obj = {"rows": result}
    else:
        payload_obj = result

    payload = json.dumps(payload_obj, default=str)
    await redis.setex(key, RESULT_CACHE_TTL_SECONDS, payload)
    logger.debug("Cache SET  key=%s  rows=%d  ttl=%ds", key, len(result), RESULT_CACHE_TTL_SECONDS)


async def cache_get(query_hash: str) -> dict[str, Any] | None:
    """
    Retrieve previously cached query results, or None if not present.

    Args:
        query_hash: SHA-256 hash of the normalised SQL query.

    Returns:
        List of row dicts, or None on cache miss.
    """
    redis = await get_redis()
    key = f"result:{query_hash}"
    raw = await redis.get(key)
    if raw is None:
        logger.debug("Cache MISS key=%s", key)
        return None
    logger.debug("Cache HIT  key=%s", key)
    return json.loads(raw)


async def cache_delete(query_hash: str) -> None:
    """Remove a cached result (useful for testing or explicit invalidation)."""
    redis = await get_redis()
    await redis.delete(f"result:{query_hash}")
