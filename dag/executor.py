"""
dag/executor.py — LangGraph DAG Execution Engine.

Converts the TaskNode list from the Global Task Planner into a LangGraph
StateGraph and executes it with proper parallelism:

  - Independent nodes (no depends_on) run in parallel via asyncio.gather.
  - Dependent nodes wait only for their direct prerequisites.
  - Results flow through graph state and are collected at the end.

The executor spawns one SQLAgent per SQL task and feeds results to
PlotAgent tasks that depend on them.
"""

import asyncio
import logging
import time
from typing import Any

from agents.plot_agent import PlotAgent
from agents.sql_agent import SQLAgent
from planner.task_planner import TaskNode, validate_dag

logger = logging.getLogger(__name__)


class DAGExecutor:
    """
    Executes a DAG of TaskNodes, running independent nodes in parallel
    and sequencing dependent nodes correctly.
    """

    def __init__(self):
        self._plot_agent = PlotAgent()

    async def execute(
        self,
        tasks: list[TaskNode],
        event_queue: asyncio.Queue | None = None,
    ) -> dict[str, Any]:
        """
        Execute all tasks in dependency order with maximum parallelism.

        Args:
            tasks: Validated list of TaskNodes from the Global Task Planner.
            event_queue: Optional queue for streaming SSE events to the API layer.
                         When provided, task_started / task_completed events are
                         pushed as each task runs. Existing callers that omit this
                         argument are unaffected.

        Returns:
            Dict mapping task_id → result dict.
        """
        # Validate DAG structure first
        validate_dag(tasks)

        results: dict[str, Any] = {}

        # Warm the shared embedding model + FAISS indices for every domain we are
        # about to query, off the event loop. This prevents the first SQL agent's
        # synchronous model load from blocking (and serializing) all the others.
        sql_domains = sorted({t.domain for t in tasks if t.task_type == "sql"})
        if sql_domains:
            from schema.retriever import preload_retrievers
            await preload_retrievers(sql_domains)

        logger.info("[DAG] Starting streaming execution of %d tasks", len(tasks))

        # ── Streaming scheduler ──────────────────────────────────────────────
        # Instead of running tasks in synchronized "batches" (where every task in
        # a level waits for the SLOWEST task in that level before any dependent
        # can start), give each task its own coroutine that:
        #   1. awaits ONLY its own direct dependencies' completion events, then
        #   2. runs immediately, then
        #   3. signals its own completion event.
        # A dependent task therefore starts the instant its specific parents are
        # done — not when an unrelated sibling in the same level finishes.
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}

        async def run_node(task: TaskNode) -> None:
            # Wait for this task's direct prerequisites only.
            for dep in task.depends_on:
                await done_events[dep].wait()

            if event_queue is not None:
                event_queue.put_nowait({
                    "event": "task_started",
                    "ts": time.time(),
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "domain": task.domain,
                    "description": task.description,
                })

            result = await self._execute_task(task, results)
            results[task.id] = result
            done_events[task.id].set()

            if event_queue is not None:
                event_queue.put_nowait({
                    "event": "task_completed",
                    "ts": time.time(),
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "domain": task.domain,
                    "description": task.description,
                    "source": result.get("source", "unknown"),
                    "row_count": result.get("row_count", result.get("task_count", 0)),
                    "elapsed_ms": result.get("elapsed_ms", 0),
                })

            logger.info(
                "[DAG] Task %s (%s) done — source=%s  rows=%s  elapsed=%.0fms",
                task.id,
                task.task_type,
                result.get("source", "?"),
                result.get("row_count", result.get("task_count", "?")),
                result.get("elapsed_ms", 0),
            )

        # Launch every node at once; the await-on-deps gating enforces ordering.
        await asyncio.gather(*(run_node(t) for t in tasks), return_exceptions=False)

        logger.info("[DAG] All %d tasks completed.", len(tasks))
        return results

    async def _execute_task(
        self,
        task: TaskNode,
        prior_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch a single task to the appropriate agent."""
        if task.task_type == "sql":
            return await self._run_sql_task(task)
        elif task.task_type == "file_search":
            return await self._run_file_task(task)
        elif task.task_type == "derived":
            from agents.derived_agent import compute_derived
            from blackboard.result_cache import cache_set

            result = compute_derived(task, prior_results)
            # Store the result in Blackboard so PlotAgent can find it via query_hash
            await cache_set(result["query_hash"], result)
            return result
        elif task.task_type == "plot":
            return await self._run_plot_task(task, prior_results)
        else:
            raise ValueError(f"Unknown task_type: {task.task_type!r}")

    async def _run_sql_task(self, task: TaskNode) -> dict[str, Any]:
        """Spawn a SQL agent and run the task through the Blackboard."""
        agent = SQLAgent(agent_id=f"sql_{task.id}", domain=task.domain)
        result = await agent.run(task.description)
        return result

    async def _run_file_task(self, task: TaskNode) -> dict[str, Any]:
        """Spawn a File agent and run two-step document retrieval."""
        from agents.file_agent import FileAgent

        agent = FileAgent(agent_id=f"file_{task.id}")
        return await agent.run(task.description)

    async def _run_plot_task(
        self,
        task: TaskNode,
        prior_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a plot task using the Plot Agent.

        The Plot Agent reads results from the Blackboard (via query_hash of
        upstream SQL task results) — never from PostgreSQL.
        """
        import time
        t0 = time.perf_counter()

        # Gather all SQL/derived results this plot task depends on
        upstream_results = [
            prior_results[dep_id]
            for dep_id in task.depends_on
            if dep_id in prior_results
        ]

        if not upstream_results:
            return {
                "agent_id": "plot_agent",
                "task": task.description,
                "error": "No upstream results available",
                "source": "blackboard",
                "elapsed_ms": 0,
            }

        # Generate summary from all upstream results
        summary = await self._plot_agent.generate_summary(upstream_results)

        # Generate a chart from the first SQL/derived result (not a plot result)
        chart = None
        for res in upstream_results:
            # Only use SQL and derived task results for chart generation
            # Skip plot task results which don't have query_hash
            if res.get("source") in ("owner", "subscriber", "cache", "derived"):
                query_hash = res.get("query_hash")
                if query_hash:
                    chart = await self._plot_agent.generate_chart(
                        query_hash=query_hash,
                        chart_type="bar_chart",
                        title=task.description,
                    )
                    break

        elapsed = (time.perf_counter() - t0) * 1000
        result = {
            "agent_id": "plot_agent",
            "task": task.description,
            "source": "blackboard",
            "summary": summary,
            "task_count": len(upstream_results),
            "elapsed_ms": round(elapsed, 1),
        }
        if chart:
            result["chart"] = chart
        return result
