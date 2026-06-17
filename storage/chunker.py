"""
storage/chunker.py — Split extracted document text into overlapping chunks.

Chunks are the unit of embedding and retrieval. Overlap preserves context that
would otherwise be split across a boundary (a sentence cut in half still appears
whole in one of the two neighbouring chunks).

The splitter is paragraph-aware: it accumulates whole paragraphs up to the size
limit before cutting, and only hard-splits a single oversized paragraph. This
keeps chunks semantically coherent rather than slicing mid-word.

chunk_sections() builds on top of this with one more rule: a chunk never spans
two different headings. A 1200-character section under "Security Policy" still
gets split into multiple ~512-char chunks (chunk_text does that), but none of
those chunks bleed into the next section's content — keeping each chunk's
heading/category metadata accurate for the rule-based retrieval filter in
storage/document_store.py.
"""

from typing import TypedDict

from config import CHUNK_OVERLAP, CHUNK_SIZE
from storage.categorizer import classify_category
from storage.text_extractor import Section


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split ``text`` into chunks of roughly ``chunk_size`` characters with
    ``overlap`` characters of trailing context carried into the next chunk.

    Returns a list of non-empty chunk strings.
    """
    text = (text or "").strip()
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = chunk_size // 4

    # Normalise paragraph boundaries.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        # A single paragraph larger than the limit: hard-split it with overlap.
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start = end - overlap
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            # Carry the tail of the current chunk as overlap into the next.
            tail = current[-overlap:] if overlap else ""
            flush()
            current = f"{tail}\n{para}".strip() if tail else para

    flush()
    return [c for c in chunks if c]


class Chunk(TypedDict):
    text: str
    heading: str | None
    category: str


def chunk_sections(
    sections: list[Section],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Chunk a heading-delimited document (see storage/text_extractor.py) into
    size-bounded pieces that respect section boundaries and carry metadata.

    Each section is chunked independently via chunk_text() — preserving the
    existing paragraph-aware overlap behaviour within a section — but chunks
    never cross a heading boundary, and each chunk is tagged with its
    section's heading plus a rule-based category (storage/categorizer.py) so
    retrieval can filter by metadata before running the vector search.
    """
    out: list[Chunk] = []
    for section in sections:
        heading = section.get("heading")
        for piece in chunk_text(section["text"], chunk_size=chunk_size, overlap=overlap):
            out.append({
                "text": piece,
                "heading": heading,
                "category": classify_category(heading, piece),
            })
    return out
