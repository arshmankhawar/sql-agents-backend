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
from typing import Any

import psycopg2

from config import DATABASE_URL

logger = logging.getLogger(__name__)


def get_uploads_schema_defs() -> list[dict[str, Any]]:
    """
    Build schema definitions (MOCK_SCHEMAS format) for every uploaded dataset.

    Each returned dict has: table, description, columns [{name, type}], examples.
    Returns an empty list if there are no uploads (or the table is missing).
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.uploaded_datasets')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                "SELECT name, table_name, columns, description, row_count FROM uploaded_datasets"
            )
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    defs: list[dict[str, Any]] = []
    for r in rows:
        try:
            columns = json.loads(r["columns"])
        except (TypeError, json.JSONDecodeError):
            columns = []
        # Normalise the stored Postgres types into the retriever's lowercase form.
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
