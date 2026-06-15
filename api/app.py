"""
api/app.py — FastAPI application factory.

Responsibilities:
  - Lifespan: preload FAISS retrievers at startup, close Redis at shutdown.
  - CORS: allow Vite dev server (localhost:5173) and Vite preview (localhost:4173).
  - Static files: serve the compiled React SPA from ./static when it exists
    (production single-container mode).
  - Routers: mount health, query, and index routes under /api/v1.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    from schema.indexer import MOCK_SCHEMAS
    from schema.retriever import preload_retrievers

    all_domains = list(MOCK_SCHEMAS.keys())
    logger.info("[Startup] Preloading FAISS retrievers for domains: %s", all_domains)
    await preload_retrievers(all_domains)
    logger.info("[Startup] FAISS preload complete. Ready to serve requests.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    from blackboard.client import close_redis
    await close_redis()
    logger.info("[Shutdown] Redis connection closed.")


app = FastAPI(
    title="Multi-Agent SQL Analytics API",
    description="Stream real-time SQL analytics pipeline results via SSE.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed CORS origins. The deployed frontend origin is supplied via the
# FRONTEND_ORIGIN env var so no code change is needed across environments.
# In production (ENVIRONMENT=production) only that origin is allowed; the
# localhost dev/preview origins are added only outside production.
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "")
_is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

_dev_origins = [] if _is_production else [
    "http://localhost:5173",  # Vite dev
    "http://localhost:4173",  # Vite preview
]
_allowed_origins = [o for o in [*_dev_origins, _frontend_origin] if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────
from api.routes import compare, health, index, query  # noqa: E402

app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(index.router, prefix="/api/v1")

# ── Static SPA (production single-container mode) ────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

if os.path.isdir(_STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(_STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
