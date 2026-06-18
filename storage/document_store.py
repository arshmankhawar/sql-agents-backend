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

Chunk *text* is stored in the Postgres document_chunks table (source of
truth); the FAISS metadata only holds ids, so the index stays small.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2

from config import (
    DATABASE_URL,
    DOCUMENT_INDEX_PATH,
    FILE_SEARCH_TOP_CHUNKS,
    FILE_SEARCH_TOP_DOCS,
)
from storage.categorizer import classify_category
from storage.chunker import chunk_sections
from storage.text_extractor import extract_sections

logger = logging.getLogger(__name__)

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
    Extract → split into heading-delimited sections → chunk (respecting
    section boundaries) → embed → persist an uploaded document.

    Writes the document + its chunks to Postgres (each chunk tagged with its
    section heading and a rule-based category — see storage/categorizer.py),
    appends chunk vectors to chunks.faiss and a single document vector to
    docs.faiss, then saves both.

    Returns metadata: document_id, filename, chunk_count.
    """
    path = Path(path)
    sections = extract_sections(path)
    chunks = chunk_sections(sections)
    if not chunks:
        raise ValueError("No extractable text found in the document.")

    chunk_texts = [c["text"] for c in chunks]
    file_type = path.suffix.lower().lstrip(".")

    # 1. Persist document + chunk rows (Postgres is the source of truth for text).
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, description, file_type, chunk_count) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (filename, description, file_type, len(chunks)),
            )
            document_id = cur.fetchone()[0]
            chunk_ids: list[int] = []
            for i, ch in enumerate(chunks):
                cur.execute(
                    "INSERT INTO document_chunks (document_id, chunk_index, text, heading, category) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (document_id, i, ch["text"], ch["heading"], ch["category"]),
                )
                chunk_ids.append(cur.fetchone()[0])
        conn.commit()
    finally:
        conn.close()

    # 2. Embed chunks and append to the vector indices.
    chunk_vecs = _embed(chunk_texts)
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
                "heading": chunks[ci]["heading"],
                "category": chunks[ci]["category"],
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
    Two-step semantic search over uploaded documents, narrowed by a third,
    free rule-based filter: category.

    The query is classified into the same small category taxonomy used to
    tag chunks at ingest time (storage/categorizer.py — no LLM call, just
    keyword rules). Chunks whose category matches the query's are preferred
    over equally-FAISS-ranked chunks that don't, so an "average salary
    policy" question is less likely to surface an unrelated narrative-section
    chunk just because it scored a fraction higher on raw cosine similarity.
    When the query has no inferable category ("general"), no preference is
    applied — behaviourally identical to plain two-step search.

    Returns up to ``top_chunks`` chunk dicts:
        {"text", "document_id", "filename", "chunk_index", "score", "category"}
    sorted by descending relevance (category match first, then score).
    Empty list if nothing is indexed.
    """
    docs_index, docs_meta = _load_index(_DOCS_INDEX, _DOCS_META)
    chunks_index, chunks_meta = _load_index(_CHUNKS_INDEX, _CHUNKS_META)
    if not docs_meta or not chunks_meta or chunks_index is None:
        return []

    qvec = _embed([query])
    query_category = classify_category(None, query)

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

    matching: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for pos, score in zip(chunk_idx[0], scores[0]):
        if pos == -1:
            continue
        meta = chunks_meta[pos]
        if meta["document_id"] not in candidate_doc_ids:
            continue
        hit = {**meta, "score": float(score)}
        if query_category != "general" and meta.get("category") == query_category:
            matching.append(hit)
        else:
            other.append(hit)

    # Step 3 — category preference: category-matching chunks first, then
    # backfill with the remaining best-scoring chunks if there aren't enough.
    hits = (matching + other)[:top_chunks]
    if query_category != "general":
        logger.info(
            "[DocStore] Step 3 — query category=%r, %d category-matching chunks preferred",
            query_category, min(len(matching), top_chunks),
        )

    if not hits:
        return []

    # Hydrate chunk text from the DB (source of truth) in one query.
    chunk_ids = [h["chunk_id"] for h in hits]
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM document_chunks WHERE id = ANY(%s)",
                (chunk_ids,),
            )
            text_by_id = {row[0]: row[1] for row in cur.fetchall()}
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
            "heading": h.get("heading"),
            "category": h.get("category", "general"),
        })
    logger.info("[DocStore] Step 2 — returning %d chunks", len(results))
    return results


def has_documents() -> bool:
    """True if at least one document has been indexed."""
    _, docs_meta = _load_index(_DOCS_INDEX, _DOCS_META)
    return bool(docs_meta)


def list_documents() -> list[dict[str, Any]]:
    """Return all uploaded documents (most recent first)."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.documents')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                "SELECT id, filename, description, file_type, chunk_count, created_at "
                "FROM documents ORDER BY created_at DESC"
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
