"""
api/routes/compare.py — POST /api/v1/compare

Runs the same query through Baseline and Improved architectures, streaming
progress events via SSE. Three comparison modes:
  mode=1: DB_LATENCY=0ms   (SQLite native speed — overhead visible)
  mode=2: DB_LATENCY=250ms (Production DB — deduplication wins)
  mode=3: DB_LATENCY=250ms + warm cache (Repeat query — cache dominates)
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth.models import UserInfo
from api.auth.security import get_current_user

logger = logging.getLogger("api.compare")
router = APIRouter()


class CompareRequest(BaseModel):
    query: str
    mode: int = 1


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _run_comparison(query: str, mode: int, q: asyncio.Queue) -> None:
    import db.pool as _pool
    from comparison_demo import BaselineResult, run_baseline, run_improved
    from planner.parent_planner import ParentOrchestrator
    from planner.task_planner import validate_dag
    from utils.logging_config import new_request_id

    new_request_id()
    logger.info(
        "[Compare] compare_received",
        extra={"phase": "received", "query": query, "mode": mode},
    )

    step = 0

    def status(message: str, total: int) -> None:
        nonlocal step
        step += 1
        q.put_nowait({"event": "compare_status", "message": message, "step": step, "total": total})

    try:
        latency_ms = 0 if mode == 1 else 250
        _pool._DB_LATENCY_MS = float(latency_ms)

        total = 4 if mode == 3 else 3

        # Plan once to derive agent_tasks for the baseline
        status("Planning query structure…", total)
        tasks = await ParentOrchestrator().plan_global_dag(query)
        validate_dag(tasks)
        unique_sql_tasks = [(t.domain, t.description) for t in tasks if t.task_type == "sql"]
        if not unique_sql_tasks:
            unique_sql_tasks = [
                ("tech_startup", "fetch employee salary data"),
                ("airport", "fetch employee salary data"),
            ]

        # Baseline: each unique task is duplicated to simulate independent agents
        # making the same DB call without any deduplication layer.
        baseline_tasks = unique_sql_tasks * 2

        baseline: BaselineResult
        if mode == 3:
            status("Running improved pipeline (warming cache)…", total)
            await run_improved(query, flush_cache=True)

            status("Running baseline pipeline…", total)
            baseline = await run_baseline(query, baseline_tasks)

            status("Running improved pipeline with warm cache…", total)
            improved = await run_improved(query, flush_cache=False)
        else:
            status("Running baseline pipeline…", total)
            baseline = await run_baseline(query, baseline_tasks)

            status("Running improved pipeline…", total)
            improved = await run_improved(query, flush_cache=True)

        db_saved = baseline.db_calls - improved.db_calls
        tok_saved = baseline.schema_tokens - improved.schema_tokens
        time_diff = baseline.elapsed_ms - improved.elapsed_ms

        logger.info(
            "[Compare] comparison_complete",
            extra={
                "phase": "comparison_complete",
                "mode": mode,
                "db_calls_saved": db_saved,
                "schema_tokens_saved": tok_saved,
                "time_diff_ms": round(time_diff),
            },
        )

        q.put_nowait({
            "event": "comparison_complete",
            "mode": mode,
            "latency_ms": latency_ms,
            "baseline": {
                "db_calls": baseline.db_calls,
                "schema_tokens": baseline.schema_tokens,
                "elapsed_ms": round(baseline.elapsed_ms),
                "answer": baseline.answer,
            },
            "improved": {
                "db_calls": improved.db_calls,
                "cache_hits": improved.cache_hits,
                "schema_tokens": improved.schema_tokens,
                "elapsed_ms": round(improved.elapsed_ms),
                "answer": improved.answer,
            },
            "delta": {
                "db_calls_saved": db_saved,
                "db_calls_saved_pct": round(db_saved / baseline.db_calls * 100) if baseline.db_calls else 0,
                "schema_tokens_saved": tok_saved,
                "schema_tokens_saved_pct": round(tok_saved / baseline.schema_tokens * 100) if baseline.schema_tokens else 0,
                "time_diff_ms": round(time_diff),
                "winner": "improved" if time_diff > 0 else "baseline",
            },
        })

    except Exception as exc:
        logger.exception("[Compare] Error for query %r mode=%d", query, mode)
        from utils.error_handler import friendly_error
        q.put_nowait({"event": "error", "message": friendly_error(exc)})
    finally:
        q.put_nowait(None)


async def _compare_generator(query: str, mode: int) -> AsyncGenerator[str, None]:
    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_run_comparison(query, mode, q))
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            yield _sse(item)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@router.post("/compare")
async def stream_compare(
    body: CompareRequest,
    _: UserInfo = Depends(get_current_user),
):
    if body.mode not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="mode must be 1, 2, or 3")
    return StreamingResponse(
        _compare_generator(body.query, body.mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
