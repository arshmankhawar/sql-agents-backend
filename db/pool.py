"""
db/pool.py — SQLite Database Access Layer (Table Gateway).

All domains now share a single database file, db/analytics.db. Domain
separation is enforced inside that file by per-domain VIEWS (see
db/setup_sqlite.py): an "airport" agent queries the `airport_employees` view,
which only ever exposes airport rows. The `domain` argument is therefore no
longer used to pick a file — it is retained for query-hash isolation and
logging — and every query runs against the one unified database.

execute_query() runs real SQL against analytics.db and returns results as a
list of dicts (column_name → value), matching the interface the rest of the
pipeline expects.

The function is async (uses asyncio.to_thread) so callers on the event loop
are never blocked by disk I/O.
"""

import asyncio
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from config import SQLITE_DB_PATH

# Optional simulated latency (milliseconds) for each DB call.
# Set DB_LATENCY_MS=200 in environment to represent production PostgreSQL cost.
# Default is 0 (true SQLite speed, no simulation).
_DB_LATENCY_MS: float = float(os.getenv("DB_LATENCY_MS", "0"))

logger = logging.getLogger(__name__)

_DB_DIR = Path(SQLITE_DB_PATH)

# Single unified database. Every domain resolves to this file; domain isolation
# is provided by per-domain views inside it (see db/setup_sqlite.py).
_DB_FILE = _DB_DIR / "analytics.db"


def _domain_db(domain: str) -> Path:
    """Return the unified SQLite file path (same for every domain)."""
    return _DB_FILE


def _sync_execute(sql: str, domain: str) -> list[dict[str, Any]]:
    """
    Execute a SQL query synchronously against the domain's SQLite database.

    This is the real DB call — no mocking, no fake latency.
    Results are returned as a list of dicts (column_name → value).
    """
    db_file = _domain_db(domain)
    if not db_file.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {db_file}\n"
            f"Run `python db/setup_sqlite.py` to create it."
        )

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        logger.info("[DB][SQLite][%s] sql=%.80r  rows=%d", domain, sql, len(rows))
        return rows
    except sqlite3.Error as exc:
        logger.error("[DB][SQLite][%s] Error executing sql=%.80r: %s", domain, sql, exc)
        raise
    finally:
        conn.close()


async def execute_query(sql: str, domain: str = "default") -> list[dict[str, Any]]:
    """
    Execute a SQL query against the domain's SQLite database.

    Runs the blocking sqlite3 call in a thread pool so the asyncio event loop
    is never blocked, preserving parallelism in the DAG executor.

    Set env var DB_LATENCY_MS to simulate production database latency (e.g., 200
    for a realistic PostgreSQL round-trip on a real dataset). Default is 0.

    Args:
        sql:    The SQL SELECT statement to execute.
        domain: One of "airport", "tech_startup", "restaurant".

    Returns:
        List of row dicts (column_name → value).
    """
    if _DB_LATENCY_MS > 0:
        await asyncio.sleep(_DB_LATENCY_MS / 1000.0)
    return await asyncio.to_thread(_sync_execute, sql, domain)


# ── Legacy close helpers (no-ops for SQLite, kept for interface compatibility) ──

async def close_pool() -> None:
    """No-op: SQLite connections are per-query, no persistent pool to close."""
    pass
