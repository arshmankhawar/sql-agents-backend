"""
api/routes/index.py — POST /api/build-index

Triggers a FAISS schema index rebuild in a background thread.
build_index() is synchronous and CPU-bound, so it runs via asyncio.to_thread
to avoid blocking the event loop or any in-flight requests.
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends

from api.auth.models import UserInfo
from api.auth.security import get_current_user

router = APIRouter()


@router.post("/build-index")
async def build_index_endpoint(
    background_tasks: BackgroundTasks,
    _: UserInfo = Depends(get_current_user),
):
    async def _rebuild():
        from schema.indexer import build_index
        await asyncio.to_thread(build_index)

    background_tasks.add_task(_rebuild)
    return {"status": "index rebuild started"}
