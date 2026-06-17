"""
db/migrate.py — Idempotent, data-preserving schema migrations for the unified
PostgreSQL database.

Unlike setup_postgres.py (which drops and recreates everything), this script
only adds what is missing and never destroys existing data. It is safe to run
on every deploy. Ensures the `users`, `uploaded_datasets`, `documents`, and
`document_chunks` tables exist and that the seed admin account is present.

Run:  python db/migrate.py
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_URL  # noqa: E402
from utils.passwords import hash_password  # noqa: E402

load_dotenv()


def _table_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
        return cur.fetchone()[0] is not None


def ensure_users_table(conn) -> None:
    if not _table_exists(conn, "users"):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE users (
                    id            SERIAL PRIMARY KEY,
                    username      TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        print("  created table: users")

    # Seed the admin account only if no users exist (preserves any added later).
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        (count,) = cur.fetchone()
    if count == 0:
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("ADMIN_PASSWORD", "change-me")
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (admin_user, hash_password(admin_pass)),
            )
        print(f"  seeded admin user: {admin_user!r}")
    else:
        print(f"  users table already populated ({count} user(s)) — no seed needed")


def ensure_upload_tables(conn) -> None:
    """Add the file/CSV upload tables if missing (idempotent, data-preserving)."""
    if not _table_exists(conn, "uploaded_datasets"):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE uploaded_datasets (
                    id          SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    table_name  TEXT NOT NULL,
                    domain      TEXT NOT NULL,
                    columns     TEXT NOT NULL,
                    row_count   INTEGER NOT NULL DEFAULT 0,
                    filename    TEXT,
                    description TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        print("  created table: uploaded_datasets")

    if not _table_exists(conn, "documents"):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE documents (
                    id          SERIAL PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    description TEXT,
                    file_type   TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        print("  created table: documents")

    if not _table_exists(conn, "document_chunks"):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE document_chunks (
                    id          SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id),
                    chunk_index INTEGER NOT NULL,
                    text        TEXT NOT NULL,
                    heading     TEXT,
                    category    TEXT NOT NULL DEFAULT 'general'
                )
                """
            )
        print("  created table: document_chunks")
    else:
        # Added for category/rule-based chunking — backfill on pre-existing tables.
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS heading TEXT")
            cur.execute(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'general'"
            )
        print("  ensured columns: document_chunks.heading, document_chunks.category")


def migrate() -> None:
    print(f"Migrating Postgres database (DSN={DATABASE_URL!r}) ...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as exc:
        print(f"  could not connect — is Postgres running? ({exc})")
        return
    try:
        ensure_users_table(conn)
        ensure_upload_tables(conn)
        conn.commit()
    finally:
        conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
