"""
api/routes/query.py — POST /api/v1/query

Streams the full pipeline execution as Server-Sent Events (SSE).

Architecture:
  - Each request creates an asyncio.Queue (fully isolated between concurrent requests).
  - _run_pipeline_into_queue() runs as a Task, pushing events into the queue as side effects.
  - _sse_generator() drains the queue and yields formatted SSE lines until the None sentinel.
  - StreamingResponse wraps the generator with media_type="text/event-stream".
  - If the client disconnects mid-stream, the generator's finally block cancels the pipeline Task.
"""

import asyncio
import json
import logging
import time as _time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth.models import UserInfo
from api.auth.security import get_current_user

logger = logging.getLogger("api.query")

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _run_pipeline_into_queue(user_request: str, q: asyncio.Queue) -> None:
    """
    Full pipeline driver. Pushes SSE event dicts into q as the pipeline progresses.
    Always pushes a None sentinel when done (success or failure) so the generator stops.
    """
    from agents.synthesis_agent import SynthesisAgent
    from dag.executor import DAGExecutor
    from planner.parent_planner import ParentOrchestrator

    t_plan_start = _time.perf_counter()
    try:
        q.put_nowait({"event": "planning_started", "ts": _time.time()})

        planner = ParentOrchestrator()
        tasks = await planner.plan_global_dag(user_request)

        domains = sorted({t.domain for t in tasks if t.domain not in ("global", "default")})
        requires_cross = any(t.id == "global_plot_1" for t in tasks)
        q.put_nowait({
            "event": "domains_identified",
            "ts": _time.time(),
            "domains": domains,
            "requires_cross_domain_plot": requires_cross,
        })

        plan_ms = (_time.perf_counter() - t_plan_start) * 1000
        q.put_nowait({
            "event": "planning_complete",
            "ts": _time.time(),
            "task_count": len(tasks),
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "task_type": t.task_type,
                    "domain": t.domain,
                    "depends_on": t.depends_on,
                }
                for t in tasks
            ],
        })

        t_exec_start = _time.perf_counter()
        executor = DAGExecutor()
        results = await executor.execute(tasks, event_queue=q)
        exec_ms = (_time.perf_counter() - t_exec_start) * 1000

        q.put_nowait({"event": "synthesis_started", "ts": _time.time()})
        t_synth_start = _time.perf_counter()
        synthesis = SynthesisAgent()
        answer = await synthesis.synthesize(user_request, tasks, results)
        synth_ms = (_time.perf_counter() - t_synth_start) * 1000

        # Collect charts from plot task results
        charts = []
        for task in tasks:
            if task.task_type == "plot" and task.id in results:
                r = results[task.id]
                if "chart" in r:
                    charts.append({"task_id": task.id, **r["chart"]})

        # Compute DB stats from SQL task results
        sql_results = [
            results[t.id] for t in tasks if t.task_type == "sql" and t.id in results
        ]
        db_calls = sum(1 for r in sql_results if r.get("source") == "owner")
        cache_hits = sum(1 for r in sql_results if r.get("source") == "cache")

        q.put_nowait({
            "event": "synthesis_complete",
            "ts": _time.time(),
            "answer": answer,
            "charts": charts,
            "stats": {
                "plan_ms": round(plan_ms),
                "exec_ms": round(exec_ms),
                "synth_ms": round(synth_ms),
                "total_ms": round(plan_ms + exec_ms + synth_ms),
                "db_calls": db_calls,
                "cache_hits": cache_hits,
            },
        })

    except Exception as exc:
        logger.exception("[Pipeline] Unhandled error for request %r", user_request)
        q.put_nowait({
            "event": "error",
            "ts": _time.time(),
            "message": str(exc),
            "phase": "pipeline",
        })
    finally:
        q.put_nowait(None)  # sentinel — generator will stop


async def _sse_generator(user_request: str) -> AsyncGenerator[str, None]:
    """
    Async generator consumed by StreamingResponse.
    Spawns the pipeline as a Task, drains the queue, yields SSE-formatted strings.
    Cancels the pipeline Task if the client disconnects.
    """
    q: asyncio.Queue = asyncio.Queue()
    pipeline_task = asyncio.create_task(_run_pipeline_into_queue(user_request, q))
    try:
        while True:
            item = await q.get()
            if item is None:  # sentinel — pipeline finished
                break
            yield _sse_line(item)
    finally:
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


@router.post("/query")
async def stream_query(
    body: QueryRequest,
    _: UserInfo = Depends(get_current_user),
):
    return StreamingResponse(
        _sse_generator(body.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )
