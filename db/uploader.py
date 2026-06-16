"""
db/uploader.py — CSV / Excel ingestion into the unified analytics database.

An uploaded tabular file becomes a physical SQLite table named ``user_<name>``
inside db/analytics.db, plus a row in ``uploaded_datasets`` describing it. Once
ingested, the table is just another surface the existing SQL pipeline can query
— the dynamic schema registry (schema/dynamic_schema.py) exposes its columns to
the FAISS retriever under the uploads domain, so SQLAgent needs no changes.

Column types are inferred from the parsed pandas dtypes and mapped to SQLite
storage classes (INTEGER / REAL / TEXT). Names are sanitised to safe SQL
identifiers to avoid injection and reserved-word issues.
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import SQLITE_DB_PATH, UPLOADS_DOMAIN

logger = logging.getLogger(__name__)

_DB_FILE = Path(SQLITE_DB_PATH) / "analytics.db"


# ─────────────────────────────────────────────────────────────────────────────
# Identifier sanitisation
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_identifier(raw: str) -> str:
    """
    Convert an arbitrary string into a safe SQL identifier: lowercase, only
    [a-z0-9_], never starting with a digit, never empty.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", raw.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "dataset"
    if cleaned[0].isdigit():
        cleaned = f"d_{cleaned}"
    return cleaned


def _infer_sqlite_type(dtype: Any) -> str:
    """Map a pandas dtype to a SQLite storage class."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    return "TEXT"


def _read_file(path: Path, file_ext: str) -> pd.DataFrame:
    """Parse a CSV or Excel file into a DataFrame."""
    if file_ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if file_ext in (".xlsx", ".xls"):
        return pd.read_excel(path)  # requires openpyxl
    raise ValueError(f"Unsupported tabular file type: {file_ext!r} (use .csv or .xlsx)")


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(
    path: str | Path,
    dataset_name: str,
    filename: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Ingest a CSV/Excel file into analytics.db as ``user_<dataset_name>``.

    Re-uploading the same dataset name replaces the previous table and its
    registry row (idempotent overwrite).

    Returns a metadata dict: name, table_name, columns, row_count.
    """
    path = Path(path)
    file_ext = path.suffix.lower()
    df = _read_file(path, file_ext)

    if df.empty or len(df.columns) == 0:
        raise ValueError("Uploaded file has no rows or no columns.")

    safe_name = sanitize_identifier(dataset_name)
    table_name = f"user_{safe_name}"

    # Sanitise column names, keeping a mapping so we still insert the right data.
    col_specs: list[dict[str, str]] = []
    rename: dict[str, str] = {}
    seen: set[str] = set()
    for original in df.columns:
        safe_col = sanitize_identifier(str(original))
        # Disambiguate collisions after sanitisation.
        base = safe_col
        i = 1
        while safe_col in seen:
            i += 1
            safe_col = f"{base}_{i}"
        seen.add(safe_col)
        rename[original] = safe_col
        col_specs.append({"name": safe_col, "type": _infer_sqlite_type(df[original].dtype)})

    df = df.rename(columns=rename)

    cols_ddl = ", ".join(f'"{c["name"]}" {c["type"]}' for c in col_specs)

    conn = sqlite3.connect(str(_DB_FILE))
    try:
        cur = conn.cursor()
        # Replace any prior version of this dataset.
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cur.execute(f'CREATE TABLE "{table_name}" ({cols_ddl})')

        placeholders = ", ".join("?" for _ in col_specs)
        col_names = ", ".join(f'"{c["name"]}"' for c in col_specs)
        # NaN → None so SQLite stores NULL rather than the string 'nan'.
        records = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in df.itertuples(index=False, name=None)
        ]
        cur.executemany(
            f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})',
            records,
        )

        row_count = len(records)
        cur.execute(
            """
            INSERT INTO uploaded_datasets
                (name, table_name, domain, columns, row_count, filename, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                table_name=excluded.table_name,
                domain=excluded.domain,
                columns=excluded.columns,
                row_count=excluded.row_count,
                filename=excluded.filename,
                description=excluded.description,
                created_at=datetime('now')
            """,
            (
                safe_name,
                table_name,
                UPLOADS_DOMAIN,
                json.dumps(col_specs),
                row_count,
                filename or path.name,
                description,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "[Uploader] Ingested %r → table %s (%d rows, %d cols)",
        dataset_name, table_name, row_count, len(col_specs),
    )
    return {
        "name": safe_name,
        "table_name": table_name,
        "columns": col_specs,
        "row_count": row_count,
    }


def list_datasets() -> list[dict[str, Any]]:
    """Return all registered uploaded datasets (most recent first)."""
    if not _DB_FILE.exists():
        return []
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        # Guard against the table not existing yet (pre-migration DB).
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='uploaded_datasets'"
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute(
            "SELECT name, table_name, row_count, filename, description, created_at "
            "FROM uploaded_datasets ORDER BY created_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            out.append(d)
        return out
    finally:
        conn.close()
