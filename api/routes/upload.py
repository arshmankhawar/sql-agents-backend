"""
api/routes/upload.py — File & dataset upload endpoints (all auth-protected).

  POST /api/v1/upload/csv   — CSV/Excel → a queryable table in analytics.db
  POST /api/v1/upload/file  — document (pdf/docx/md/txt) → vector store
  GET  /api/v1/datasets     — list uploaded tabular datasets
  GET  /api/v1/files        — list uploaded documents

CSV ingestion also rebuilds the uploads FAISS schema index and refreshes the
cached retriever so the new table is immediately queryable by the SQL pipeline.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.auth.models import UserInfo
from api.auth.security import get_current_user
from config import UPLOAD_DIR, UPLOADS_DOMAIN
from utils.error_handler import friendly_error

logger = logging.getLogger("api.upload")

router = APIRouter()

_CSV_EXTS = {".csv", ".xlsx", ".xls", ".txt"}
_DOC_EXTS = {".pdf", ".docx", ".md", ".txt"}
_MAX_BYTES = 25 * 1024 * 1024  # 25 MB upload cap


async def _read_capped(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB).")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


@router.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    description: str | None = Form(None),
    _: UserInfo = Depends(get_current_user),
):
    """Ingest a CSV/Excel file into analytics.db as a queryable table."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _CSV_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported tabular type {ext!r}. Use CSV or Excel.")

    data = await _read_capped(file)
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            from db.uploader import ingest_file
            meta = await asyncio.to_thread(
                ingest_file, tmp_path, dataset_name, filename=file.filename, description=description
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Make the new table immediately retrievable: rebuild the uploads schema
        # index and invalidate the cached retriever.
        from schema.indexer import build_domain_index
        from schema.retriever import refresh_retriever
        build_domain_index(UPLOADS_DOMAIN)
        refresh_retriever(UPLOADS_DOMAIN)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("CSV upload failed")
        raise HTTPException(status_code=400, detail=friendly_error(exc))

    return {
        "status": "ok",
        "dataset": meta["name"],
        "table": meta["table_name"],
        "row_count": meta["row_count"],
        "columns": [c["name"] for c in meta["columns"]],
    }


@router.post("/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    _: UserInfo = Depends(get_current_user),
):
    """Ingest a document into the vector store for file search."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _DOC_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported document type {ext!r}. Use PDF, DOCX, MD, or TXT.")

    data = await _read_capped(file)
    try:
        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / (file.filename or f"document{ext}")
        dest.write_bytes(data)

        from storage.document_store import ingest_document
        meta = await asyncio.to_thread(
            ingest_document, dest, filename=file.filename or dest.name, description=description
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document upload failed")
        raise HTTPException(status_code=400, detail=friendly_error(exc))

    return {
        "status": "ok",
        "document_id": meta["document_id"],
        "filename": meta["filename"],
        "chunk_count": meta["chunk_count"],
    }


@router.get("/datasets")
async def list_uploaded_datasets(_: UserInfo = Depends(get_current_user)):
    """List uploaded tabular datasets."""
    from db.uploader import list_datasets
    return {"datasets": await asyncio.to_thread(list_datasets)}


@router.get("/files")
async def list_uploaded_files(_: UserInfo = Depends(get_current_user)):
    """List uploaded documents."""
    from storage.document_store import list_documents
    return {"documents": await asyncio.to_thread(list_documents)}
