"""
agents/base_agent.py — Blackboard-Aware Query Execution Base.

Implements the core deduplication + coordination flow that ALL SQL agents use:

  1. Normalise the SQL and compute its hash.
  2. Check the shared result cache → cache hit → return immediately.
  3. Atomically try to claim ownership via SETNX.
     a. Owner → execute the query → publish result → return.
     b. Subscriber → wait on Pub/Sub channel → receive result → return.

This guarantees:
  - Identical concurrent queries result in exactly ONE PostgreSQL execution.
  - All other agents receive the same result without touching the DB.
  - A crashed owner's slot expires via TTL, preventing permanent stalls.
"""

import asyncio
import logging
import time
import hashlib
from typing import Any

from blackboard.query_registry import (
    complete_query,
    renew_heartbeat,
    try_claim_query,
    wait_for_result,
)
from blackboard.result_cache import cache_get
from db.pool import execute_query
from utils.sql_normalizer import normalize_and_hash

logger = logging.getLogger(__name__)

# How often (seconds) an owner agent renews its TTL during a long query
_HEARTBEAT_INTERVAL = 10.0


async def run_with_blackboard(
    agent_id: str,
    task_description: str,
    sql: str,
    domain: str = "default",
) -> dict[str, Any]:
    """
    Execute a SQL query using full Blackboard coordination.

    Args:
        agent_id:         Unique identifier for this agent instance (e.g. "sql_agent_a").
        task_description: Human-readable label for logging.
        sql:              The SQL query string to execute.

    Returns:
        A result dict with:
            {
                "agent_id":    str,
                "task":        str,
                "sql":         str,
                "query_hash":  str,
                "source":      "cache" | "owner" | "subscriber",
                "rows":        list[dict],
                "elapsed_ms":  float,
            }
    """
    t0 = time.perf_counter()
    canonical_sql, _ = normalize_and_hash(sql)
    # Incorporate domain into the query hash to isolate identical SQL across domains
    query_hash = hashlib.sha256(f"{domain}:{canonical_sql}".encode("utf-8")).hexdigest()

    logger.info(
        "[%s] Starting task=%r  hash=%s  sql=%.60r",
        agent_id, task_description, query_hash[:12], canonical_sql,
    )

    # ── Step 1: Check result cache (already completed by a prior run) ─────────
    cached = await cache_get(query_hash)
    if cached is not None:
        # cache_get now returns a result dict (with 'rows') or legacy list
        rows = cached.get("rows") if isinstance(cached, dict) and "rows" in cached else cached
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] CACHE HIT  hash=%s  rows=%d  %.0fms", agent_id, query_hash[:12], len(rows), elapsed)
        return _build_result(agent_id, task_description, canonical_sql, query_hash, "cache", rows, elapsed)

    # ── Step 2: Atomic claim via SETNX ────────────────────────────────────────
    is_owner = await try_claim_query(query_hash, agent_id)

    if is_owner:
        # ── Owner path: execute the query and notify subscribers ──────────────
        rows = await _execute_as_owner(agent_id, query_hash, canonical_sql, domain)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] OWNER DONE  hash=%s  rows=%d  %.0fms", agent_id, query_hash[:12], len(rows), elapsed)
        return _build_result(agent_id, task_description, canonical_sql, query_hash, "owner", rows, elapsed)

    else:
        # ── Subscriber path: wait for the owner to publish the result ─────────
        logger.info("[%s] SUBSCRIBER  hash=%s — waiting for owner...", agent_id, query_hash[:12])
        cached_result = await wait_for_result(query_hash)

        if cached_result is None:
            # Owner timed out — fall back to executing ourselves
            logger.warning("[%s] Owner timed out for hash=%s — falling back to direct execution", agent_id, query_hash[:12])
            rows = await execute_query(canonical_sql, domain)
            await complete_query(query_hash, rows)
        else:
            # wait_for_result returns the cached payload (dict with 'rows')
            rows = cached_result.get("rows") if isinstance(cached_result, dict) and "rows" in cached_result else cached_result

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] SUBSCRIBER GOT RESULT  hash=%s  rows=%d  %.0fms", agent_id, query_hash[:12], len(rows), elapsed)
        return _build_result(agent_id, task_description, canonical_sql, query_hash, "subscriber", rows, elapsed)


async def _execute_as_owner(
    agent_id: str,
    query_hash: str,
    sql: str,
    domain: str = "default",
) -> list[dict[str, Any]]:
    """
    Execute the query as the owning agent, renewing TTL heartbeat during execution.
    Publishes the result to all waiting subscribers upon completion.
    """
    # Run the DB query and heartbeat renewal concurrently
    heartbeat_task = asyncio.create_task(_heartbeat_loop(query_hash))
    try:
        rows = await execute_query(sql, domain)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    # Persist result in cache and notify all subscribers via Pub/Sub
    await complete_query(query_hash, rows)
    return rows


async def _heartbeat_loop(query_hash: str) -> None:
    """Periodically renew the TTL of a running registry entry."""
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await renew_heartbeat(query_hash)
    except asyncio.CancelledError:
        pass


def _build_result(
    agent_id: str,
    task: str,
    sql: str,
    query_hash: str,
    source: str,
    rows: list[dict[str, Any]],
    elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "task": task,
        "sql": sql,
        "query_hash": query_hash,
        "source": source,       # "cache" | "owner" | "subscriber"
        "rows": rows,
        "row_count": len(rows),
        "elapsed_ms": round(elapsed_ms, 1),
    }
