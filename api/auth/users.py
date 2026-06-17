"""
api/auth/users.py — Data access for the `users` table.

Auth lookups use parameterised psycopg2 queries directly (NOT the pipeline's
db.pool.execute_query, which executes raw LLM-generated SQL and routes through
domain views). Keeping auth on its own parameterised path prevents any chance
of SQL injection via the username field.
"""

import asyncio
from typing import Any

import psycopg2

from config import DATABASE_URL


def _sync_get_user(username: str) -> dict[str, Any] | None:
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))
    except psycopg2.Error:
        # users table may not exist yet (pre-migration) — treat as no user.
        return None
    finally:
        conn.close()


async def get_user(username: str) -> dict[str, Any] | None:
    """Return the user row (id, username, password_hash) or None if not found."""
    return await asyncio.to_thread(_sync_get_user, username)
