"""
storage/document_store.py — Vector store + ingestion for uploaded documents.

Two FAISS indices live on disk under DOCUMENT_INDEX_PATH:

  docs.faiss    — ONE vector per document (the mean of its chunk vectors).
  chunks.faiss  — one vector per chunk.

Retrieval is two-step ("narrow then refine"), which is what keeps an abundance
of files from drowning the result set:

  1. Embed the query and search docs.faiss → the FILE_SEARCH_TOP_DOCS most
     relevant *documents*.
  2. Search chunks.faiss broadly, then keep only chunks belonging to those
     candidate documents, and return the FILE_SEARCH_TOP_CHUNKS best.

Chunk *text* is stored in the SQLite document_chunks table (source of truth);
the FAISS metadata only holds ids, so the index stays small.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    DOCUMENT_INDEX_PATH,
    FILE_SEARCH_TOP_CHUNKS,
    FILE_SEARCH_TOP_DOCS,
    SQLITE_DB_PATH,
)
from storage.chunker import chunk_text
from storage.text_extractor import extract_text

logger = logging.getLogger(__name__)

_DB_FILE = Path(SQLITE_DB_PATH) / "analytics.db"
_INDEX_DIR = Path(DOCUMENT_INDEX_PATH)
_DOCS_INDEX = _INDEX_DIR / "docs.faiss"
_DOCS_META = _INDEX_DIR / "docs_meta.json"
_CHUNKS_INDEX = _INDEX_DIR / "chunks.faiss"
_CHUNKS_META = _INDEX_DIR / "chunks_meta.json"

# Serialises index mutation (add/persist). Searches are read-only on a flat index.
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding (reuse the one shared all-MiniLM model)
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> np.ndarray:
    from schema.retriever import _get_shared_model

    model = _get_shared_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Index persistence
# ─────────────────────────────────────────────────────────────────────────────

def _load_index(index_path: Path, meta_path: Path):
    """Load a (faiss index, metadata list) pair, or (None, []) if absent."""
    import faiss

    if not index_path.exists() or not meta_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def _save_index(index, meta: list[dict], index_path: Path, meta_path: Path) -> None:
    import faiss

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _new_flat_index(dim: int):
    import faiss

    return faiss.IndexFlatIP(dim)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_document(
    path: str | Path,
    filename: str,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Extract → chunk → embed → persist an uploaded document.

    Writes the document + its chunks to SQLite, appends chunk vectors to
    chunks.faiss and a single document vector to docs.faiss, then saves both.

    Returns metadata: document_id, filename, chunk_count.
    """
    path = Path(path)
    text = extract_text(path)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found in the document.")

    file_type = path.suffix.lower().lstrip(".")

    # 1. Persist document + chunk rows (SQLite is the source of truth for text).
    conn = sqlite3.connect(str(_DB_FILE))
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, description, file_type, chunk_count) "
            "VALUES (?, ?, ?, ?)",
            (filename, description, file_type, len(chunks)),
        )
        document_id = cur.lastrowid
        chunk_ids: list[int] = []
        for i, ch in enumerate(chunks):
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, text) "
                "VALUES (?, ?, ?)",
                (document_id, i, ch),
            )
            chunk_ids.append(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()

    # 2. Embed chunks and append to the vector indices.
    chunk_vecs = _embed(chunks)
    doc_vec = chunk_vecs.mean(axis=0, keepdims=True)
    # Re-normalise the averaged document vector so cosine similarity stays valid.
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm
    doc_vec = doc_vec.astype(np.float32)

    with _lock:
        dim = chunk_vecs.shape[1]

        chunks_index, chunks_meta = _load_index(_CHUNKS_INDEX, _CHUNKS_META)
        if chunks_index is None:
            chunks_index = _new_flat_index(dim)
        chunks_index.add(chunk_vecs)
        for cid, ci in zip(chunk_ids, range(len(chunks))):
            chunks_meta.append({
                "document_id": document_id,
                "chunk_id": cid,
                "chunk_index": ci,
                "filename": filename,
            })
        _save_index(chunks_index, chunks_meta, _CHUNKS_INDEX, _CHUNKS_META)

        docs_index, docs_meta = _load_index(_DOCS_INDEX, _DOCS_META)
        if docs_index is None:
            docs_index = _new_flat_index(dim)
        docs_index.add(doc_vec)
        docs_meta.append({"document_id": document_id, "filename": filename})
        _save_index(docs_index, docs_meta, _DOCS_INDEX, _DOCS_META)

    logger.info(
        "[DocStore] Ingested %r → document_id=%d, %d chunks",
        filename, document_id, len(chunks),
    )
    return {"document_id": document_id, "filename": filename, "chunk_count": len(chunks)}


# ─────────────────────────────────────────────────────────────────────────────
# Two-step retrieval
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_docs: int = FILE_SEARCH_TOP_DOCS,
    top_chunks: int = FILE_SEARCH_TOP_CHUNKS,
) -> list[dict[str, Any]]:
    """
    Two-step semantic search over uploaded documents.

    Returns up to ``top_chunks`` chunk dicts:
        {"text", "document_id", "filename", "chunk_index", "score"}
    sorted by descending relevance. Empty list if nothing is indexed.
    """
    docs_index, docs_meta = _load_index(_DOCS_INDEX, _DOCS_META)
    chunks_index, chunks_meta = _load_index(_CHUNKS_INDEX, _CHUNKS_META)
    if not docs_meta or not chunks_meta or chunks_index is None:
        return []

    qvec = _embed([query])

    # Step 1 — narrow to the most relevant documents.
    kd = min(top_docs, len(docs_meta))
    _, doc_idx = docs_index.search(qvec, kd)
    candidate_doc_ids = {
        docs_meta[i]["document_id"] for i in doc_idx[0] if i != -1
    }
    logger.info("[DocStore] Step 1 — candidate documents: %s", candidate_doc_ids)

    # Step 2 — refine within those documents only. Search broadly, then filter.
    kc = min(max(top_chunks * 5, top_chunks), len(chunks_meta))
    scores, chunk_idx = chunks_index.search(qvec, kc)

    hits: list[dict[str, Any]] = []
    for pos, score in zip(chunk_idx[0], scores[0]):
        if pos == -1:
            continue
        meta = chunks_meta[pos]
        if meta["document_id"] not in candidate_doc_ids:
            continue
        hits.append({**meta, "score": float(score)})
        if len(hits) >= top_chunks:
            break

    if not hits:
        return []

    # Hydrate chunk text from the DB (source of truth) in one query.
    chunk_ids = [h["chunk_id"] for h in hits]
    placeholders = ", ".join("?" for _ in chunk_ids)
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT id, text FROM document_chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        text_by_id = {r["id"]: r["text"] for r in rows}
    finally:
        conn.close()

    results = []
    for h in hits:
        results.append({
            "text": text_by_id.get(h["chunk_id"], ""),
            "document_id": h["document_id"],
            "filename": h["filename"],
            "chunk_index": h["chunk_index"],
            "score": round(h["score"], 4),
        })
    logger.info("[DocStore] Step 2 — returning %d chunks", len(results))
    return results


def has_documents() -> bool:
    """True if at least one document has been indexed."""
    _, docs_meta = _load_index(_DOCS_INDEX, _DOCS_META)
    return bool(docs_meta)


def list_documents() -> list[dict[str, Any]]:
    """Return all uploaded documents (most recent first)."""
    if not _DB_FILE.exists():
        return []
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute(
            "SELECT id, filename, description, file_type, chunk_count, created_at "
            "FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
