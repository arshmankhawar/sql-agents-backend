"""
api/routes/query.py — POST /api/v1/query

Streams the full pipeline execution as Server-Sent Events (SSE).

Architecture:
  - Each request creates an asyncio.Queue (fully isolated between concurrent requests).
  - _run_pipeline_into_queue() runs as a Task, pushing events into the queue as side effects.
  - _sse_generator() drains the queue and yields formatted SSE lines until the None sentinel.
  - StreamingResponse wraps the generator with media_type="text/event-stream".
  - If the client disconnects mid-stream, the generator's finally block cancels the pipeline Task.

Top-level routing (guard -> direct reply | full pipeline) is a tiny LangGraph
StateGraph with a genuine conditional edge: the guard's data_query vs.
chitchat/refuse verdict is a static two-way branch decided once per request,
which is exactly what add_conditional_edges is for (unlike the per-task DAG
in dag/executor.py, whose dependency structure is dynamic and needs
finer-grained gating than a superstep-batched edge can give it).
"""

import asyncio
import json
import logging
import time as _time
from typing import Any, AsyncGenerator, TypedDict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from api.auth.models import UserInfo
from api.auth.security import get_current_user

logger = logging.getLogger("api.query")

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class _RouterState(TypedDict):
    user_request: str
    queue: asyncio.Queue
    request_id: str
    t_plan_start: float
    verdict: Any


async def _guard_node(state: _RouterState) -> dict[str, Any]:
    from agents.guard import triage
    from utils.logging_config import new_request_id

    request_id = new_request_id()
    q: asyncio.Queue = state["queue"]
    logger.info("[Pipeline] query_received", extra={"phase": "received", "query": state["user_request"]})
    t_plan_start = _time.perf_counter()
    q.put_nowait({"event": "planning_started", "ts": _time.time(), "request_id": request_id})

    verdict = await triage(state["user_request"])
    return {"request_id": request_id, "t_plan_start": t_plan_start, "verdict": verdict}


def _route_after_guard(state: _RouterState) -> str:
    """The conditional edge: data_query -> run the real pipeline, else -> direct reply."""
    return "run_pipeline" if state["verdict"].is_data_query else "direct_reply"


async def _direct_reply_node(state: _RouterState) -> dict[str, Any]:
    verdict = state["verdict"]
    logger.info("[Pipeline] short-circuited by guard (intent=%s)", verdict.intent)
    elapsed = round((_time.perf_counter() - state["t_plan_start"]) * 1000)
    state["queue"].put_nowait({
        "event": "synthesis_complete",
        "ts": _time.time(),
        "answer": verdict.reply,
        "charts": [],
        "sources": [],
        "stats": {
            "plan_ms": elapsed, "exec_ms": 0, "synth_ms": 0, "total_ms": elapsed,
            "db_calls": 0, "cache_hits": 0,
        },
    })
    return {}


async def _run_pipeline_into_queue(user_request: str, q: asyncio.Queue) -> None:
    """
    Full pipeline driver. Pushes SSE event dicts into q as the pipeline progresses.
    Always pushes a None sentinel when done (success or failure) so the generator stops.

    Routing into either the direct-reply short-circuit or the real
    plan/execute/synthesize pipeline happens via a tiny compiled LangGraph
    graph (guard -> conditional edge -> {direct_reply, run_pipeline} -> END).
    """
    try:
        graph = StateGraph(_RouterState)
        graph.add_node("guard", _guard_node)
        graph.add_node("direct_reply", _direct_reply_node)
        graph.add_node("run_pipeline", _run_full_pipeline_node)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", _route_after_guard, {
            "direct_reply": "direct_reply",
            "run_pipeline": "run_pipeline",
        })
        graph.add_edge("direct_reply", END)
        graph.add_edge("run_pipeline", END)
        compiled = graph.compile()

        await compiled.ainvoke({"user_request": user_request, "queue": q})
    except Exception as exc:
        logger.exception("[Pipeline] Unhandled error for request %r", user_request)
        from utils.error_handler import friendly_error
        q.put_nowait({
            "event": "error",
            "ts": _time.time(),
            "message": friendly_error(exc),
            "phase": "pipeline",
        })
    finally:
        q.put_nowait(None)  # sentinel — generator will stop


async def _run_full_pipeline_node(state: _RouterState) -> dict[str, Any]:
    """Plan -> execute DAG -> synthesize. Runs only when the guard's
    conditional edge routes here (a genuine data_query)."""
    from agents.synthesis_agent import SynthesisAgent
    from dag.executor import DAGExecutor
    from planner.parent_planner import ParentOrchestrator

    user_request = state["user_request"]
    q: asyncio.Queue = state["queue"]
    request_id = state["request_id"]
    t_plan_start = state["t_plan_start"]

    try:
        planner = ParentOrchestrator()
        tasks = await planner.plan_global_dag(user_request)

        domains = sorted({t.domain for t in tasks if t.domain not in ("global", "default", "files")})
        requires_cross = any(t.id == "global_plot_1" for t in tasks)
        q.put_nowait({
            "event": "domains_identified",
            "ts": _time.time(),
            "domains": domains,
            "requires_cross_domain_plot": requires_cross,
        })

        plan_ms = (_time.perf_counter() - t_plan_start) * 1000
        logger.info(
            "[Pipeline] planning_complete",
            extra={
                "phase": "planning_complete",
                "domains": domains,
                "task_count": len(tasks),
                "plan_ms": round(plan_ms),
            },
        )
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
        logger.info(
            "[Pipeline] execution_complete",
            extra={"phase": "execution_complete", "exec_ms": round(exec_ms)},
        )

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

        # Collect document citations from any file_search task results so the
        # client can render source cards beneath the answer.
        sources = []
        for task in tasks:
            if task.task_type == "file_search" and task.id in results:
                for chunk in results[task.id].get("chunks", []):
                    sources.append({
                        "filename": chunk.get("filename"),
                        "text": chunk.get("text", "")[:500],
                        "score": chunk.get("score"),
                    })

        # Compute DB stats from SQL task results
        sql_results = [
            results[t.id] for t in tasks if t.task_type == "sql" and t.id in results
        ]
        db_calls = sum(1 for r in sql_results if r.get("source") == "owner")
        cache_hits = sum(1 for r in sql_results if r.get("source") == "cache")

        logger.info(
            "[Pipeline] synthesis_complete",
            extra={
                "phase": "synthesis_complete",
                "synth_ms": round(synth_ms),
                "total_ms": round(plan_ms + exec_ms + synth_ms),
                "db_calls": db_calls,
                "cache_hits": cache_hits,
            },
        )
        q.put_nowait({
            "event": "synthesis_complete",
            "ts": _time.time(),
            "answer": answer,
            "charts": charts,
            "sources": sources,
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
        from utils.error_handler import friendly_error
        q.put_nowait({
            "event": "error",
            "ts": _time.time(),
            "message": friendly_error(exc),
            "phase": "pipeline",
            "request_id": request_id,
        })
    return {}


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
