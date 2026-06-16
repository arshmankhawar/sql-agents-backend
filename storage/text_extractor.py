"""
storage/text_extractor.py — Extract plain text from uploaded documents.

Supported types: PDF (.pdf), Word (.docx), Markdown (.md), and plain text (.txt).
PDF parsing uses pdfplumber; DOCX uses python-docx. Each is imported lazily so a
missing optional dependency only fails when that file type is actually uploaded.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def extract_text(path: str | Path) -> str:
    """
    Return the plain-text content of a document.

    Raises ValueError for unsupported file types.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")

    raise ValueError(
        f"Unsupported document type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
    text = "\n".join(parts)
    logger.info("[Extractor] PDF %s → %d chars", path.name, len(text))
    return text


def _extract_docx(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    logger.info("[Extractor] DOCX %s → %d chars", path.name, len(text))
    return text
