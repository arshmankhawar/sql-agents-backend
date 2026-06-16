"""
config.py — Centralised configuration for the Multi-Agent SQL System.

All values are loaded from environment variables (via .env file).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Small, cheap, fast model used for the input guard / triage step. Classifying a
# message ("is this a real data question, chit-chat, or an off-topic/jailbreak
# attempt?") does not need the 70B model — the 8B instant model is ~10x cheaper
# and gates the expensive pipeline so trivial/abusive queries cost almost nothing.
GUARD_MODEL: str = os.getenv("GUARD_MODEL", "llama-3.1-8b-instant")

# ── Redis / Blackboard ────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Seconds a "running" entry can live before it is auto-expired
# (guards against a crashed agent holding a query slot indefinitely)
QUERY_TTL_SECONDS: int = int(os.getenv("QUERY_TTL_SECONDS", "120"))

# Seconds to keep completed results in the shared result cache
RESULT_CACHE_TTL_SECONDS: int = int(os.getenv("RESULT_CACHE_TTL_SECONDS", "3600"))

# ── SQLite databases ──────────────────────────────────────────────────────────
# Directory containing the unified database file (analytics.db)
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./db")

# ── Schema Retrieval ──────────────────────────────────────────────────────────
SCHEMA_INDEX_PATH: str = os.getenv("SCHEMA_INDEX_PATH", "./schema_index")
SCHEMA_TOP_K: int = int(os.getenv("SCHEMA_TOP_K", "5"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── File & Document Uploads ───────────────────────────────────────────────────
# Directory where raw uploaded files (documents) are persisted.
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

# FAISS index directory for uploaded-document chunk embeddings (separate from the
# schema index so document search and schema retrieval never collide).
DOCUMENT_INDEX_PATH: str = os.getenv("DOCUMENT_INDEX_PATH", "./document_index")

# Chunking parameters for document ingestion.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# Two-step file retrieval: first narrow to the N most relevant documents, then
# pull the K best chunks from within only those documents. This keeps an
# abundance of files from flooding a single flat search.
FILE_SEARCH_TOP_DOCS: int = int(os.getenv("FILE_SEARCH_TOP_DOCS", "3"))
FILE_SEARCH_TOP_CHUNKS: int = int(os.getenv("FILE_SEARCH_TOP_CHUNKS", "5"))

# Domain name under which uploaded CSV/Excel tables live in analytics.db.
UPLOADS_DOMAIN: str = os.getenv("UPLOADS_DOMAIN", "uploads")

# ── JWT Authentication ──────────────────────────────────────────────────────
# Secret used to sign access tokens. MUST be overridden in production via the
# JWT_SECRET_KEY env var; the insecure default only exists so local dev runs.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-only-insecure-change-me")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Seed admin user, created by db/setup_sqlite.py if the users table is empty.
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "change-me")
