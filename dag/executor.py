"""
dag/executor.py — LangGraph DAG Execution Engine.

Converts the TaskNode list from the Global Task Planner into a LangGraph
``StateGraph`` and executes it through LangGraph's own compiled-graph runtime
(``graph.compile().ainvoke(...)``) instead of hand-rolled ``asyncio.gather`` +
bare ``asyncio.Event`` bookkeeping.

Why every node is wired straight from START (and the gating still happens
inside each node, not via the edges):

  The task set and its dependency edges are different on every request — they
  come straight out of the LLM planner — so this graph is built fresh per
  call rather than declared once at import time. LangGraph schedules nodes in
  "supersteps": every node ready in superstep N runs concurrently, and
  superstep N+1 only starts once ALL of superstep N has finished. If we wired
  dependency edges directly (task -> its dependent), a fast child with one
  slow, unrelated sibling in the same superstep would still wait for that
  sibling — exactly the coarse, level-batched scheduling the original
  hand-rolled executor was deliberately built to avoid (see its previous
  docstring: "a node should start the instant its specific parents finish,
  not wait for an entire level of the graph").

  So instead: every TaskNode becomes its own graph node, wired directly from
  START (all scheduled in the same superstep), and each node's *first* line
  of work is to await an ``asyncio.Event`` for each of its own direct
  dependencies before doing anything else. That preserves the exact
  fine-grained gating of the original design while genuinely running through
  LangGraph's StateGraph/node/edge machinery rather than a bespoke scheduler.

Streaming: SSE ``task_started``/``task_completed`` events are pushed into the
caller-supplied ``event_queue`` from inside each node, exactly as before — the
frontend's PipelineProgress/useSSEQuery contract is unchanged.

Task type dispatch (sql/derived/plot/file_search) is still a single
dispatcher function (``_execute_task``) rather than per-type conditional
edges: each task instance needs its own closure (id, description, domain),
and the dependency graph itself is inherently per-task — there's no shared
"next task" condition to branch on at the graph level. Conditional routing
*is* used one layer up, in api/routes/query.py, where the input guard's
data_query vs. chitchat/refuse verdict is a genuine static two-way branch
decided once per request — a much better fit for ``add_conditional_edges``
than this dynamic per-task DAG.
"""

import asyncio
import logging
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.plot_agent import PlotAgent
from agents.sql_agent import SQLAgent
from planner.task_planner import TaskNode, validate_dag

logger = logging.getLogger(__name__)


def _merge_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """State reducer: union two partial {task_id: result} dicts.

    Each node only ever writes its own task_id key, so a plain union is safe
    even though many nodes update this field concurrently within the same
    superstep.
    """
    return {**left, **right}


class PipelineState(TypedDict):
    results: Annotated[dict[str, Any], _merge_results]


class DAGExecutor:
    """
    Executes a DAG of TaskNodes via a compiled LangGraph StateGraph, running
    independent nodes in parallel and gating dependent nodes on their direct
    prerequisites only (see module docstring for why gating lives inside the
    node functions rather than in the graph's edges).
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

        # Warm the shared embedding model + FAISS indices for every domain we are
        # about to query, off the event loop. This prevents the first SQL agent's
        # synchronous model load from blocking (and serializing) all the others.
        sql_domains = sorted({t.domain for t in tasks if t.task_type == "sql"})
        if sql_domains:
            from schema.retriever import preload_retrievers
            await preload_retrievers(sql_domains)

        logger.info("[DAG] Building LangGraph StateGraph for %d tasks", len(tasks))

        # Fine-grained gating primitives (see module docstring) and the shared
        # results dict that downstream derived/plot nodes read from — kept as a
        # plain dict in closure scope (not solely the LangGraph state value)
        # because derived/plot nodes need to read sibling results the instant
        # they're available, not after a superstep-wide state merge.
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        results: dict[str, Any] = {}

        graph: StateGraph = StateGraph(PipelineState)
        for task in tasks:
            graph.add_node(task.id, self._make_node(task, done_events, results, event_queue))
        for task in tasks:
            graph.add_edge(START, task.id)
            graph.add_edge(task.id, END)

        compiled = graph.compile()
        await compiled.ainvoke({"results": {}})

        logger.info("[DAG] All %d tasks completed.", len(tasks))
        return results

    def _make_node(
        self,
        task: TaskNode,
        done_events: dict[str, asyncio.Event],
        results: dict[str, Any],
        event_queue: asyncio.Queue | None,
    ):
        """Build the LangGraph node function for a single TaskNode."""

        async def node(_state: PipelineState) -> dict[str, Any]:
            # Wait for this task's direct prerequisites only — not its
            # siblings, not the rest of the superstep.
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
            return {"results": {task.id: result}}

        return node

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
