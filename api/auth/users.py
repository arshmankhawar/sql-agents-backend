"""
api/auth/users.py — Data access for the `users` table.

Auth lookups use parameterised sqlite3 queries directly (NOT the pipeline's
db.pool.execute_query, which executes raw LLM-generated SQL and routes through
domain views). Keeping auth on its own parameterised path prevents any chance
of SQL injection via the username field.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from config import SQLITE_DB_PATH

_DB_FILE = Path(SQLITE_DB_PATH) / "analytics.db"


def _sync_get_user(username: str) -> dict[str, Any] | None:
    if not _DB_FILE.exists():
        return None
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        # users table may not exist yet (pre-migration) — treat as no user.
        return None
    finally:
        conn.close()


async def get_user(username: str) -> dict[str, Any] | None:
    """Return the user row (id, username, password_hash) or None if not found."""
    return await asyncio.to_thread(_sync_get_user, username)
