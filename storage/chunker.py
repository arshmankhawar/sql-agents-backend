"""
storage/chunker.py — Split extracted document text into overlapping chunks.

Chunks are the unit of embedding and retrieval. Overlap preserves context that
would otherwise be split across a boundary (a sentence cut in half still appears
whole in one of the two neighbouring chunks).

The splitter is paragraph-aware: it accumulates whole paragraphs up to the size
limit before cutting, and only hard-splits a single oversized paragraph. This
keeps chunks semantically coherent rather than slicing mid-word.
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE


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
