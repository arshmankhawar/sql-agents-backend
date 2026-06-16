"""
schema/dynamic_schema.py — Runtime schema for uploaded CSV/Excel datasets.

The built-in domains (airport, tech_startup, restaurant) have static schemas
hard-coded in schema/indexer.py:MOCK_SCHEMAS. Uploaded datasets are not known
until runtime, so their schema is read from the ``uploaded_datasets`` table and
returned in the SAME shape MOCK_SCHEMAS uses. This lets the existing FAISS
retriever and SQL agent treat uploaded tables identically to built-in ones.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

_DB_FILE = Path(SQLITE_DB_PATH) / "analytics.db"


def get_uploads_schema_defs() -> list[dict[str, Any]]:
    """
    Build schema definitions (MOCK_SCHEMAS format) for every uploaded dataset.

    Each returned dict has: table, description, columns [{name, type}], examples.
    Returns an empty list if there are no uploads (or the table is missing).
    """
    if not _DB_FILE.exists():
        return []

    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='uploaded_datasets'"
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute(
            "SELECT name, table_name, columns, description, row_count FROM uploaded_datasets"
        ).fetchall()
    finally:
        conn.close()

    defs: list[dict[str, Any]] = []
    for r in rows:
        try:
            columns = json.loads(r["columns"])
        except (TypeError, json.JSONDecodeError):
            columns = []
        # Normalise the stored SQLite types into the retriever's lowercase form.
        norm_cols = [
            {"name": c["name"], "type": str(c.get("type", "text")).lower()}
            for c in columns
        ]
        col_names = [c["name"] for c in norm_cols]
        description = r["description"] or (
            f"User-uploaded dataset '{r['name']}' ({r['row_count']} rows). "
            f"Columns: {', '.join(col_names)}."
        )
        defs.append({
            "table": r["table_name"],
            "description": description,
            "columns": norm_cols,
            "examples": [
                f"data from {r['name']}",
                f"rows of {r['name']}",
                f"{r['name']} {' '.join(col_names[:4])}",
            ],
        })
    return defs


def list_uploaded_table_summaries() -> list[str]:
    """Short one-line summaries of uploaded tables for the parent planner prompt."""
    summaries = []
    for d in get_uploads_schema_defs():
        cols = ", ".join(c["name"] for c in d["columns"])
        summaries.append(f"{d['table']} (columns: {cols})")
    return summaries
