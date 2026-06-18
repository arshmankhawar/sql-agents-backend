"""
storage/categorizer.py — Rule-based category classification for document chunks.

Deliberately NOT an LLM call: classifying a chunk (or a query) by keyword
overlap against a small fixed taxonomy is nearly free, so it can run on every
chunk at ingest time and on every query at retrieval time without adding
latency or token cost to the pipeline. This is what lets retrieval narrow by
metadata *before* the (also free, but broader) FAISS vector search — fewer
candidate chunks reach the LLM synthesis prompt, which is the actual "faster
and less tokens" goal.

The taxonomy is intentionally small and tuned to the kinds of documents this
system expects (policies, financial reports, narrative/background material,
and tabular data embedded in prose documents). Add new categories by adding a
keyword list — no retraining, no embeddings.
"""

import re

CATEGORIES = ("policy", "financial", "narrative", "table", "general")

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "policy": [
        "policy", "procedure", "compliance", "regulation", "guideline",
        "rule", "clearance", "security", "governance", "protocol",
        "requirement", "standard", "code of conduct", "must not", "shall",
    ],
    "financial": [
        "revenue", "budget", "salary", "cost", "price", "expense",
        "financial", "invoice", "payment", "profit", "margin", "earnings",
        "balance sheet", "quarter", "fiscal", "tax", "investment",
    ],
    "narrative": [
        "overview", "summary", "background", "introduction", "history",
        "story", "mission", "vision", "about us", "founded", "journey",
    ],
}

# A chunk that's mostly numeric rows / delimited columns reads like an
# embedded table even inside a PDF/DOCX paragraph flow.
_TABLE_ROW_RE = re.compile(r"(\d+[.,]?\d*\s*[|\t]\s*){2,}|(\s{2,}\d+[.,]?\d*){3,}")


def _keyword_score(haystack: str, keywords: list[str]) -> int:
    return sum(haystack.count(kw) for kw in keywords)


def classify_category(heading: str | None, text: str) -> str:
    """
    Classify a chunk (or a query) into one of CATEGORIES using keyword
    overlap. The heading (if present) is weighted higher than body text
    since it's a much stronger, cheaper signal of what a section is about.

    Returns "general" when no rule scores above zero.
    """
    heading_l = (heading or "").lower()
    body_l = (text or "")[:800].lower()  # cap: classification doesn't need the whole chunk

    if _TABLE_ROW_RE.search(text or ""):
        return "table"

    best_category = "general"
    best_score = 0
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = _keyword_score(heading_l, keywords) * 3 + _keyword_score(body_l, keywords)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score > 0 else "general"
