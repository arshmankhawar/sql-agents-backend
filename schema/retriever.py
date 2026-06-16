"""
schema/retriever.py — Semantic Schema Retrieval via FAISS.

Given a task description like "revenue by region", returns only the
subset of tables relevant to that task — not the entire schema.

This directly addresses:
  - Problem 3: Excessive Schema Context (tokens / hallucination)
  - Reduced prompt size → faster reasoning, fewer hallucinations

The retriever is a stateful object that loads the index once and
is shared across all agents for the lifetime of the application.
"""

import asyncio
import logging
import threading
from typing import Any

import numpy as np

from config import EMBEDDING_MODEL, SCHEMA_TOP_K
from schema.indexer import load_index

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding model (singleton)
#
# The sentence-transformer model is large (~80MB) and its load dominates
# pipeline latency. Loading a separate copy per domain wastes seconds and RAM,
# so we load exactly ONE instance and share it across every SchemaRetriever.
#
# The model is domain-agnostic, so it can be warmed concurrently with planning
# (see warm_model_async). A lock makes the load safe under concurrent callers.
# ─────────────────────────────────────────────────────────────────────────────

_shared_model = None
_model_lock = threading.Lock()


def _get_shared_model():
    """Load (once, thread-safely) and return the shared SentenceTransformer."""
    global _shared_model
    if _shared_model is None:
        with _model_lock:
            if _shared_model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer
                _shared_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("[Retriever] Loaded shared embedding model %r", EMBEDDING_MODEL)
    return _shared_model


async def warm_model_async() -> None:
    """
    Load the (domain-agnostic) embedding model off the event loop.

    Intended to be launched as a background task at pipeline start so the slow
    model load overlaps with planning LLM calls instead of running after them.
    """
    await asyncio.to_thread(_get_shared_model)


class SchemaRetriever:
    """
    Semantic schema retrieval using FAISS + sentence-transformers.

    Usage:
        retriever = SchemaRetriever("airport")
        schema_context = retriever.retrieve("average salary")
    """

    def __init__(self, domain: str, top_k: int = SCHEMA_TOP_K):
        self.domain = domain
        self.top_k = top_k
        self._index = None
        self._schema_defs: list[dict[str, Any]] = []
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy load index and (shared) embedding model on first use."""
        if self._index is not None:
            return
        from schema.indexer import ensure_index_exists

        ensure_index_exists(self.domain)
        self._index, self._schema_defs = load_index(self.domain)
        # Reuse the single shared embedding model instead of loading a per-domain copy.
        self._model = _get_shared_model()
        logger.info("[Retriever][%s] Loaded index with %d tables", self.domain, len(self._schema_defs))

    def reload(self) -> None:
        """
        Force the index + metadata to be re-read from disk on next use. Called
        after a CSV upload rebuilds this domain's index so newly added tables
        become retrievable without restarting the server.
        """
        self._index = None
        self._schema_defs = []

    def _compile_table_keywords(self) -> dict[str, set[str]]:
        """Build a keyword map from schema metadata for general relevance matching."""
        keyword_map: dict[str, set[str]] = {}
        for table_def in self._schema_defs:
            table_name = table_def["table"].lower()
            keywords = {table_name, table_name.rstrip("s")}
            keywords.update(c["name"].lower() for c in table_def.get("columns", []))
            keywords.update(word.lower().strip(".,") for word in table_def.get("description", "").split())
            for example in table_def.get("examples", []):
                keywords.update(word.lower().strip(".,") for word in example.split())
            # Keep only non-empty tokens.
            keyword_map[table_name] = {token for token in keywords if token}
        return keyword_map

    def retrieve(self, task_description: str) -> list[dict[str, Any]]:
        """
        Return the top-k most relevant table definitions for a task.

        Args:
            task_description: Natural language description of the agent's task.

        Returns:
            List of table definition dicts (subset of full schema).
        """
        self._ensure_loaded()

        lower = task_description.lower()
        table_keywords = self._compile_table_keywords()
        table_scores: dict[str, int] = {}
        for table_name, keywords in table_keywords.items():
            score = sum(1 for token in keywords if token in lower)
            if score > 0:
                table_scores[table_name] = score

        if table_scores:
            prioritized_tables = sorted(table_scores, key=lambda name: table_scores[name], reverse=True)
            prioritized = [t for t in self._schema_defs if t["table"].lower() in prioritized_tables]
            logger.info(
                "[Retriever][%s] task=%r → keyword-matched tables %s",
                self.domain, task_description[:40], [t["table"] for t in prioritized],
            )
            return prioritized[: self.top_k]

        query_vec = self._model.encode(
            [task_description], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        k = min(self.top_k, len(self._schema_defs))
        if k == 0:
            return []

        distances, indices = self._index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            table_def = self._schema_defs[idx]
            logger.debug(
                "[Retriever][%s] task=%r → table=%s (score=%.3f)",
                self.domain, task_description[:40], table_def["table"], score,
            )
            results.append(table_def)

        logger.info(
            "[Retriever][%s] task=%r → %d tables: %s",
            self.domain,
            task_description[:40],
            len(results),
            [t["table"] for t in results],
        )
        return results

    def format_schema_context(self, task_description: str) -> str:
        """
        Return a formatted string schema context suitable for injection into an LLM prompt.

        Args:
            task_description: Natural language description of the agent's task.

        Returns:
            Multi-line string with table schemas, ready for prompt injection.
        """
        tables = self.retrieve(task_description)
        lines = ["## Relevant Schema\n"]
        for t in tables:
            cols = ", ".join(
                f"{c['name']} {c['type'].upper()}"
                + (" PRIMARY KEY" if c.get("pk") else "")
                + (f" REFERENCES {c['fk']}" if c.get("fk") else "")
                for c in t["columns"]
            )
            lines.append(f"### {t['table']}")
            lines.append(f"-- {t['description']}")
            lines.append(f"CREATE TABLE {t['table']} ({cols});\n")
        return "\n".join(lines)


# Module-level dictionary — shared across all agents per domain
_retrievers: dict[str, SchemaRetriever] = {}


def get_retriever(domain: str) -> SchemaRetriever:
    """Return the module-level shared SchemaRetriever instance for a domain."""
    if domain not in _retrievers:
        _retrievers[domain] = SchemaRetriever(domain)
    return _retrievers[domain]


def refresh_retriever(domain: str) -> None:
    """Invalidate a domain's cached index so it reloads from disk on next use."""
    if domain in _retrievers:
        _retrievers[domain].reload()


async def preload_retrievers(domains: list[str]) -> None:
    """
    Warm the shared embedding model and per-domain FAISS indices BEFORE the
    DAG executes, off the event loop.

    The model load is CPU/IO-bound and synchronous, so running it inside a
    gathered SQL agent would block the loop and serialize everything. By doing
    it once up front in a worker thread, the later async SQL-generation calls
    can actually overlap.
    """
    def _load_all() -> None:
        _get_shared_model()  # one shared model for every domain
        for domain in domains:
            get_retriever(domain)._ensure_loaded()

    await asyncio.to_thread(_load_all)
    logger.info("[Retriever] Preloaded model + indices for domains: %s", domains)
