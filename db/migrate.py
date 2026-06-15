"""
db/migrate.py — Idempotent, data-preserving schema migrations for analytics.db.

Unlike setup_sqlite.py (which drops and recreates everything), this script only
adds what is missing and never destroys existing data. It is safe to run on
every deploy. Currently it ensures the `users` table exists and contains the
seed admin account (so an already-deployed analytics.db gains JWT auth without
being rebuilt).

Run:  python db/migrate.py
"""

import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.passwords import hash_password  # noqa: E402

load_dotenv()

DB_FILE = Path(os.getenv("SQLITE_DB_PATH", "./db")) / "analytics.db"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    )
    return cur.fetchone() is not None


def ensure_users_table(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "users"):
        conn.execute(
            """
            CREATE TABLE users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        print("  created table: users")

    # Seed the admin account only if no users exist (preserves any added later).
    (count,) = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    if count == 0:
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("ADMIN_PASSWORD", "change-me")
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (admin_user, hash_password(admin_pass)),
        )
        print(f"  seeded admin user: {admin_user!r}")
    else:
        print(f"  users table already populated ({count} user(s)) — no seed needed")


def migrate() -> None:
    if not DB_FILE.exists():
        print(f"  {DB_FILE} does not exist — run db/setup_sqlite.py first.")
        return
    print(f"Migrating {DB_FILE} ...")
    conn = sqlite3.connect(str(DB_FILE))
    try:
        ensure_users_table(conn)
        conn.commit()
    finally:
        conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
