"""
schema/indexer.py — Schema Knowledge Index Builder.

Builds a FAISS vector index from the database schema so agents can retrieve
only the tables relevant to their specific task (not the entire schema).

In mock mode: uses a hard-coded e-commerce schema.
In real mode: introspects the live PostgreSQL information_schema.

Each indexed document is a rich text representation of one table:
    "<table_name>: <description>. Columns: col1 (type), col2 (type), ..."

At startup (or on demand), call build_index() to create / refresh the index.
The index is saved to disk so subsequent startups just load it.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import EMBEDDING_MODEL, SCHEMA_INDEX_PATH

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mock schema definition (e-commerce domain)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "airport": [
        {
            "table": "employees",
            "description": "Airport staff including clearance levels.",
            "columns": [
                {"name": "employee_id", "type": "integer", "pk": True},
                {"name": "name", "type": "varchar"},
                {"name": "department", "type": "varchar"},
                {"name": "salary", "type": "numeric"},
                {"name": "clearance_level", "type": "integer"},
            ],
            "examples": ["average salary by clearance", "list of security staff"],
        },
        {
            "table": "flights",
            "description": "Flight schedules and statuses.",
            "columns": [
                {"name": "flight_id", "type": "integer", "pk": True},
                {"name": "airline", "type": "varchar"},
                {"name": "destination", "type": "varchar"},
                {"name": "status", "type": "varchar"},
                {"name": "passengers", "type": "integer"},
            ],
            "examples": ["delayed flights", "passengers per airline"],
        },
    ],
    "tech_startup": [
        {
            "table": "employees",
            "description": "Tech startup employees including programming skills.",
            "columns": [
                {"name": "employee_id", "type": "integer", "pk": True},
                {"name": "name", "type": "varchar"},
                {"name": "department", "type": "varchar"},
                {"name": "salary", "type": "numeric"},
                {"name": "primary_language", "type": "varchar"},
            ],
            "examples": ["average salary for python developers", "engineering head count"],
        },
        {
            "table": "projects",
            "description": "Active tech projects and their budgets.",
            "columns": [
                {"name": "project_id", "type": "integer", "pk": True},
                {"name": "name", "type": "varchar"},
                {"name": "status", "type": "varchar"},
                {"name": "budget", "type": "numeric"},
            ],
            "examples": ["total budget of active projects", "completed projects"],
        },
    ],
    "restaurant": [
        {
            "table": "employees",
            "description": "Restaurant staff including shift schedules.",
            "columns": [
                {"name": "employee_id", "type": "integer", "pk": True},
                {"name": "name", "type": "varchar"},
                {"name": "department", "type": "varchar"},
                {"name": "salary", "type": "numeric"},
                {"name": "shift", "type": "varchar"},
            ],
            "examples": ["salary for night shift", "number of chefs"],
        },
        {
            "table": "menus",
            "description": "Menu items and pricing.",
            "columns": [
                {"name": "item_id", "type": "integer", "pk": True},
                {"name": "name", "type": "varchar"},
                {"name": "category", "type": "varchar"},
                {"name": "price", "type": "numeric"},
            ],
            "examples": ["average price of desserts", "menu items under 10 dollars"],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Schema text serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _table_to_text(table_def: dict[str, Any]) -> str:
    """
    Convert a table definition dict into a rich text string for embedding.
    Including table name, description, column names/types, and usage examples.
    """
    cols = ", ".join(
        f"{c['name']} ({c['type']})" + (" [PK]" if c.get("pk") else "") + (f" [FK→{c['fk']}]" if c.get("fk") else "")
        for c in table_def["columns"]
    )
    examples = "; ".join(table_def.get("examples", []))
    return (
        f"Table: {table_def['table']}. "
        f"Description: {table_def['description']} "
        f"Columns: {cols}. "
        f"Use cases: {examples}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Index build and load
# ─────────────────────────────────────────────────────────────────────────────

def _get_schema_defs(domain: str) -> list[dict[str, Any]]:
    """Return the schema definitions for the given domain."""
    return MOCK_SCHEMAS.get(domain, [])


def build_index() -> None:
    """
    Build (or rebuild) the FAISS schema indexes for all domains and save to disk.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    logger.info("[Schema] Building indexes using model=%s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    base_out_dir = Path(SCHEMA_INDEX_PATH)

    for domain in MOCK_SCHEMAS.keys():
        schema_defs = _get_schema_defs(domain)
        if not schema_defs:
            continue

        texts = [_table_to_text(t) for t in schema_defs]

        logger.info("[Schema][%s] Embedding %d tables...", domain, len(texts))
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save to disk per domain
        out_dir = base_out_dir / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(out_dir / "index.faiss"))
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(schema_defs, f, indent=2)

        logger.info("[Schema][%s] Index saved -> %s  (%d tables)", domain, out_dir, len(schema_defs))


def load_index(domain: str):
    """
    Load the FAISS index and metadata from disk for a specific domain.
    """
    import faiss

    out_dir = Path(SCHEMA_INDEX_PATH) / domain
    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Schema index not found at {index_path}. Run build_index() first."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        schema_defs = json.load(f)

    logger.info("[Schema][%s] Index loaded from %s  (%d tables)", domain, out_dir, len(schema_defs))
    return index, schema_defs


def ensure_index_exists(domain: str) -> None:
    """Build the index if it doesn't already exist on disk."""
    index_path = Path(SCHEMA_INDEX_PATH) / domain / "index.faiss"
    if not index_path.exists():
        logger.info("[Schema][%s] Index not found — building for the first time...", domain)
        build_index()
    else:
        logger.info("[Schema][%s] Index already exists at %s", domain, index_path)
