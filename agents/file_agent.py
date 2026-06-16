"""
agents/file_agent.py — File Search Agent (vector retrieval over uploaded docs).

The File Agent is the document-side counterpart to the SQL Agent. Where the SQL
Agent fetches rows from the database, the File Agent fetches the most relevant
text chunks from uploaded documents via two-step semantic search (document-level
narrowing, then chunk-level refinement — see storage/document_store.py).

Its result dict mirrors the SQL agent's shape closely enough that the DAG
executor, blackboard logging, and synthesis agent treat it uniformly: it carries
``source="file_search"`` and a ``chunks`` list instead of ``rows``.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from config import FILE_SEARCH_TOP_CHUNKS, FILE_SEARCH_TOP_DOCS
from storage import document_store

logger = logging.getLogger(__name__)


class FileAgent:
    """Retrieves relevant document chunks for a natural-language task."""

    def __init__(self, agent_id: str | None = None):
        self.agent_id = agent_id or f"file_agent_{uuid.uuid4().hex[:6]}"

    async def run(
        self,
        task: str,
        top_docs: int = FILE_SEARCH_TOP_DOCS,
        top_chunks: int = FILE_SEARCH_TOP_CHUNKS,
    ) -> dict[str, Any]:
        """
        Search uploaded documents for the chunks most relevant to ``task``.

        The search (embedding + FAISS) is CPU-bound, so it runs in a worker
        thread to avoid blocking the event loop and to stay parallel with the
        SQL tasks running alongside it in the DAG.
        """
        t0 = time.perf_counter()
        logger.info("[%s] File search: %r", self.agent_id, task)

        chunks = await asyncio.to_thread(
            document_store.search, task, top_docs, top_chunks
        )
        elapsed = (time.perf_counter() - t0) * 1000

        sources = sorted({c["filename"] for c in chunks})
        logger.info(
            "[%s] File search done — %d chunks from %d docs (%.0fms)",
            self.agent_id, len(chunks), len(sources), elapsed,
        )
        return {
            "agent_id": self.agent_id,
            "task": task,
            "source": "file_search",
            "chunks": chunks,
            "documents": sources,
            "row_count": len(chunks),
            "elapsed_ms": round(elapsed, 1),
        }
