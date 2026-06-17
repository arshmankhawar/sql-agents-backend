"""
db/pool.py — PostgreSQL Database Access Layer (Table Gateway).

Every domain shares a single Postgres database. Domain separation is enforced
inside that database by per-domain VIEWS (see db/setup_postgres.py): an
"airport" agent queries the `airport_employees` view, which only ever exposes
airport rows. The `domain` argument is therefore not used to pick a database —
it is retained for query-hash isolation and logging — and every query runs
against the one unified database via a pooled asyncpg connection.

execute_query() runs real SQL and returns results as a list of dicts
(column_name -> value), matching the interface the rest of the pipeline
expects. This is the single chokepoint the rest of the codebase depends on —
its signature has not changed across the SQLite -> Postgres migration, so no
caller needed to change.
"""

import asyncio
import logging
import os
from typing import Any

import asyncpg

from config import DATABASE_URL

# Optional simulated latency (milliseconds) for each DB call. Was originally
# added to approximate Postgres round-trip cost while still on SQLite; now
# that real Postgres is in place this is mostly vestigial but harmless to keep
# for benchmarking against an even slower (e.g. remote) database.
_DB_LATENCY_MS: float = float(os.getenv("DB_LATENCY_MS", "0"))

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    """Lazily create (once) and return the shared asyncpg connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=DATABASE_URL,
                min_size=2,
                max_size=10,
            )
            logger.info("[DB][Postgres] Connection pool created (min=2, max=10)")
    return _pool


async def execute_query(sql: str, domain: str = "default") -> list[dict[str, Any]]:
    """
    Execute a SQL query against the unified Postgres database.

    Acquires a connection from the shared pool so concurrent DAG tasks reuse a
    small set of real connections instead of opening one per query.

    Set env var DB_LATENCY_MS to simulate additional database latency (e.g. for
    a remote DB benchmark). Default is 0.

    Args:
        sql:    The SQL SELECT statement to execute.
        domain: One of "airport", "tech_startup", "restaurant", "uploads".
                Kept for query-hash isolation (Blackboard) and logging only —
                domain scoping itself happens via SQL views.

    Returns:
        List of row dicts (column_name -> value).
    """
    if _DB_LATENCY_MS > 0:
        await asyncio.sleep(_DB_LATENCY_MS / 1000.0)

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch(sql)
            rows = [dict(r) for r in records]
            logger.info("[DB][Postgres][%s] sql=%.80r  rows=%d", domain, sql, len(rows))
            return rows
        except asyncpg.PostgresError as exc:
            logger.error("[DB][Postgres][%s] Error executing sql=%.80r: %s", domain, sql, exc)
            raise


async def close_pool() -> None:
    """Close the shared connection pool. Call once on application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("[DB][Postgres] Connection pool closed")
