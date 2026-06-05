"""
verify_streaming.py — Proves the DAG executor streams instead of batching.

No LLM/Groq needed. We build a DAG with uneven task durations and a subclassed
executor whose _execute_task just sleeps for a per-task duration while recording
each task's start/end time relative to t0.

DAG:
    slow_sql   (no deps, 0.60s)
    fast_sql   (no deps, 0.05s)
    fast_deriv (depends on fast_sql)
    slow_deriv (depends on slow_sql)

Batch scheduler property: level-1 = [slow_sql, fast_sql]; level-2 cannot start
until BOTH finish (>=0.60s). So fast_deriv would start at ~0.60s.

Streaming scheduler property: fast_deriv starts the instant fast_sql finishes
(~0.05s), long before slow_sql (~0.60s) completes.

PASS condition: fast_deriv.start < slow_sql.end  (dependent of fast task started
before the slow sibling finished -> NOT batched).
"""

import asyncio
import time

from dag.executor import DAGExecutor
from planner.task_planner import TaskNode

DURATIONS = {
    "slow_sql": 0.60,
    "fast_sql": 0.05,
    "fast_deriv": 0.02,
    "slow_deriv": 0.02,
}


class TimedExecutor(DAGExecutor):
    def __init__(self):
        super().__init__()
        self.starts: dict[str, float] = {}
        self.ends: dict[str, float] = {}
        self._t0 = time.perf_counter()

    async def _execute_task(self, task, prior_results):
        self.starts[task.id] = time.perf_counter() - self._t0
        await asyncio.sleep(DURATIONS[task.id])
        self.ends[task.id] = time.perf_counter() - self._t0
        return {"source": "test", "row_count": 0, "elapsed_ms": DURATIONS[task.id] * 1000}


async def main():
    tasks = [
        TaskNode(id="slow_sql", description="slow", task_type="sql", depends_on=[]),
        TaskNode(id="fast_sql", description="fast", task_type="sql", depends_on=[]),
        TaskNode(id="fast_deriv", description="d", task_type="derived", depends_on=["fast_sql"]),
        TaskNode(id="slow_deriv", description="d", task_type="derived", depends_on=["slow_sql"]),
    ]

    ex = TimedExecutor()
    # Skip the FAISS preload path: these are "sql" type but TimedExecutor overrides
    # _execute_task, and preload only touches real domains. Patch it out to keep
    # this test pure-scheduler.
    import schema.retriever as r
    orig = r.preload_retrievers
    async def noop(_domains): return None
    r.preload_retrievers = noop
    try:
        await ex.execute(tasks)
    finally:
        r.preload_retrievers = orig

    print("Task timeline (seconds from start):")
    for tid in DURATIONS:
        print(f"  {tid:<11} start={ex.starts[tid]:.3f}  end={ex.ends[tid]:.3f}")

    fast_deriv_start = ex.starts["fast_deriv"]
    slow_sql_end = ex.ends["slow_sql"]
    total = max(ex.ends.values())

    print(f"\nfast_deriv started at {fast_deriv_start:.3f}s")
    print(f"slow_sql finished at  {slow_sql_end:.3f}s")
    print(f"total wall time:      {total:.3f}s")

    streaming = fast_deriv_start < slow_sql_end
    # A batch scheduler would finish in ~0.60 + 0.02 = 0.62s minimum.
    # Streaming finishes in ~0.60 + 0.02 = 0.62s too for the critical path, but
    # the discriminating signal is WHEN fast_deriv starts.
    print(f"\nSTREAMING CONFIRMED: {streaming}  "
          f"(fast dependent started {slow_sql_end - fast_deriv_start:.3f}s "
          f"before slow sibling finished)")
    assert streaming, "Executor is still batching!"


if __name__ == "__main__":
    asyncio.run(main())
