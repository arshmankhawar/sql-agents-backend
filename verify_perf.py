"""
verify_perf.py — Objective measurement harness for architecture fixes.

It instruments every Groq LLM call (sync .invoke and async .ainvoke) and every
FAISS retriever load, recording the wall-clock [start, end] interval of each.

From those intervals it computes:
  - total LLM busy time  = sum of individual call durations
  - LLM wall span        = (max end - min start) across all calls
  - overlap factor       = busy / span   (1.0 = fully serial, N = N-way parallel)

A higher overlap factor proves calls actually ran concurrently. It also reports
end-to-end planning/execution wall time.

Run:  python verify_perf.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

# ── Interval recorder ─────────────────────────────────────────────────────────


@dataclass
class Recorder:
    llm_intervals: list[tuple[float, float, str]] = field(default_factory=list)
    faiss_intervals: list[tuple[float, float, str]] = field(default_factory=list)

    def summarize(self, intervals: list[tuple[float, float, str]], label: str) -> None:
        if not intervals:
            print(f"  {label}: (no calls recorded)")
            return
        busy = sum(e - s for s, e, _ in intervals)
        span = max(e for _, e, _ in intervals) - min(s for s, _, _ in intervals)
        overlap = busy / span if span > 0 else 0.0
        print(f"  {label}: count={len(intervals)}  "
              f"busy={busy*1000:.0f}ms  span={span*1000:.0f}ms  "
              f"overlap_factor={overlap:.2f}x")


REC = Recorder()


def _install_instrumentation() -> None:
    """Monkeypatch ChatGoogleGenerativeAI.invoke/ainvoke and SchemaRetriever._ensure_loaded."""
    from langchain_groq import ChatGroq
    from schema.retriever import SchemaRetriever

    orig_invoke = ChatGroq.invoke
    orig_ainvoke = ChatGroq.ainvoke
    orig_ensure = SchemaRetriever._ensure_loaded

    def timed_invoke(self, *a, **k):
        s = time.perf_counter()
        try:
            return orig_invoke(self, *a, **k)
        finally:
            REC.llm_intervals.append((s, time.perf_counter(), "sync"))

    async def timed_ainvoke(self, *a, **k):
        s = time.perf_counter()
        try:
            return await orig_ainvoke(self, *a, **k)
        finally:
            REC.llm_intervals.append((s, time.perf_counter(), "async"))

    def timed_ensure(self):
        already = self._index is not None
        s = time.perf_counter()
        try:
            return orig_ensure(self)
        finally:
            if not already:
                REC.faiss_intervals.append((s, time.perf_counter(), self.domain))

    ChatGroq.invoke = timed_invoke
    ChatGroq.ainvoke = timed_ainvoke
    SchemaRetriever._ensure_loaded = timed_ensure


async def run(request: str) -> None:
    from blackboard.client import close_redis
    from planner.parent_planner import ParentOrchestrator
    from planner.task_planner import validate_dag
    from dag.executor import DAGExecutor

    print(f"\nRequest: {request!r}")
    print("-" * 70)

    planner = ParentOrchestrator()
    t0 = time.perf_counter()
    tasks = await planner.plan_global_dag(request)
    validate_dag(tasks)
    plan_ms = (time.perf_counter() - t0) * 1000

    sql_n = sum(1 for t in tasks if t.task_type == "sql")
    print(f"Planning: {plan_ms:.0f}ms  ->  {len(tasks)} tasks ({sql_n} SQL)")
    phase_boundary = time.perf_counter()

    executor = DAGExecutor()
    t1 = time.perf_counter()
    await executor.execute(tasks)
    exec_ms = (time.perf_counter() - t1) * 1000
    print(f"Execution: {exec_ms:.0f}ms")
    print(f"TOTAL: {plan_ms + exec_ms:.0f}ms")

    # Split LLM calls into planning-phase vs execution-phase for clean overlap proof
    plan_llm = [iv for iv in REC.llm_intervals if iv[1] <= phase_boundary]
    exec_llm = [iv for iv in REC.llm_intervals if iv[1] > phase_boundary]

    print("\nInstrumentation:")
    REC.summarize(plan_llm, "LLM (planning) ")
    REC.summarize(exec_llm, "LLM (execution)")
    REC.summarize(REC.llm_intervals, "LLM (all)      ")
    REC.summarize(REC.faiss_intervals, "FAISS loads    ")

    await close_redis()


if __name__ == "__main__":
    _install_instrumentation()
    req = "Compare the average employee salary between the tech startup and the airport."
    asyncio.run(run(req))
