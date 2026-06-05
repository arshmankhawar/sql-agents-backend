"""
utils/sql_normalizer.py — SQL Normalisation and Query Hashing.

Problem addressed:
    Two SQL queries may be semantically identical but differ in formatting,
    whitespace, aliasing, or keyword casing. Without normalisation, they'd
    receive different hashes and bypass the deduplication system.

Solution:
    1. Parse the SQL with sqlglot into an AST.
    2. Re-generate the SQL from the AST in a canonical dialect.
    3. SHA-256 hash the canonical form.

This guarantees that all of the following produce the same hash:

    SELECT * FROM orders WHERE year = 2025
    select * from orders where year=2025
    SELECT   *   FROM  orders  WHERE  year  =  2025

Note: sqlglot is dialect-aware. For PostgreSQL, use dialect="postgres".
"""

import hashlib
import logging
import re

import sqlglot

logger = logging.getLogger(__name__)

# Target normalisation dialect — use "postgres" to preserve PG-specific syntax
_DIALECT = "postgres"


def normalize_sql(sql: str) -> str:
    """
    Parse and re-serialise SQL in a canonical form.

    Strips extra whitespace, normalises keyword casing, and removes trivial
    alias differences via AST round-trip through sqlglot.

    Falls back to a simple whitespace-collapse if sqlglot cannot parse the input
    (e.g., SQL fragments from tool calls that aren't yet complete).

    Args:
        sql: Raw SQL string from an agent.

    Returns:
        Canonical SQL string.
    """
    try:
        statements = sqlglot.parse(sql, dialect=_DIALECT, error_level=sqlglot.ErrorLevel.WARN)
        if statements and statements[0] is not None:
            canonical = statements[0].sql(dialect=_DIALECT, pretty=False)
            return canonical.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sqlglot parse failed (%s), falling back to whitespace normalisation", exc)

    # Fallback: collapse whitespace and uppercase keywords
    return re.sub(r"\s+", " ", sql).strip().upper()


def hash_query(sql: str) -> str:
    """
    Return a stable SHA-256 hex digest for a SQL query.

    The input is first normalised so semantically equivalent queries share
    the same hash regardless of formatting.

    Args:
        sql: Raw or already-normalised SQL string.

    Returns:
        64-character lowercase hex string.
    """
    canonical = normalize_sql(sql)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    logger.debug("hash_query  canonical=%r  hash=%s", canonical[:80], digest[:12])
    return digest


def normalize_and_hash(sql: str) -> tuple[str, str]:
    """
    Convenience wrapper returning both the canonical SQL and its hash.

    Returns:
        (canonical_sql, sha256_hex)
    """
    canonical = normalize_sql(sql)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical, digest
