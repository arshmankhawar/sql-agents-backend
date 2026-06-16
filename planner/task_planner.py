"""
planner/task_planner.py — LLM-Based Global Task Planner.

The Global Task Planner is the first component in the pipeline after the
Agent Orchestrator receives a user request.

Responsibilities:
  1. Parse the user's natural language request.
  2. Decompose it into a list of independent or dependent subtasks.
  3. Identify shared data dependencies (so the DAG executor can run them correctly).
  4. Return a structured task graph (list of TaskNode objects).

The planner uses the Groq LLM with a structured JSON output format.
It explicitly identifies `depends_on` relationships so the DAG executor
can determine which tasks run in parallel vs. sequentially.

Example:
    User: "Show me revenue trends and top customers for 2025"

    Planner output:
    [
        TaskNode(id="t1", description="revenue trends for 2025", depends_on=[]),
        TaskNode(id="t2", description="top customers by revenue for 2025", depends_on=[]),
        TaskNode(id="t3", description="plot revenue vs customers", depends_on=["t1", "t2"]),
    ]

t1 and t2 run in parallel. t3 waits for both.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, GROQ_MODEL, UPLOADS_DOMAIN

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """A single node in the task execution DAG."""
    id: str                              # Unique task ID (e.g. "airport_t1")
    description: str                     # Natural language task description
    task_type: str = "sql"               # "sql", "derived", or "plot"
    domain: str = "default"              # Domain for this task (e.g. "airport")
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite tasks
    operation: dict[str, Any] | None = None # Operation details for derived tasks


_PLANNER_SYSTEM_PROMPT_TEMPLATE = """\
You are a task decomposition planner for the {domain} database ONLY.

CRITICAL RULE: You are operating on the {domain} database exclusively.
- Do NOT reference any other domain (e.g., if the user mentions "airport" and "tech_startup", \
you only plan tasks for {domain}).
- Do NOT add WHERE clauses or filters based on domain names — the {domain} views already \
expose only {domain} data.
- The {domain} tables are named with a {domain}_ prefix (e.g. {domain}_employees). Always \
reference tables by their full prefixed name.
- SQL task descriptions must tell the SQL agent what columns and table to use within the \
{domain} schema.

Task Types:
1. "sql": Fetches base data from the {domain} tables. Descriptions must be SPECIFIC: \
name the table and columns (e.g., "fetch employee_id, name, department, salary from {domain}_employees").
2. "derived": Computes analytics in Python from an upstream SQL result (zero DB calls).
3. "plot": Visualises data from an upstream derived or sql result.

Output a valid JSON array of task objects. Each object must have:
  - "id": unique string like "t1", "t2", etc.
  - "description": specific, actionable description of what to fetch or compute
  - "task_type": "sql", "derived", or "plot"
  - "depends_on": array of task IDs this task must wait for (empty if independent)
  - "operation": (ONLY for "derived" tasks) one of:
      {{"type": "group_avg", "group_keys": ["department"], "value_key": "salary"}}   <- USE for average/mean
      {{"type": "group_sum", "group_keys": ["department"], "value_key": "salary"}}   <- USE for totals
      {{"type": "top_n_per_group", "partition_key": "department", "rank_key": "salary", "n": 5}}
      {{"type": "contribution_pct", "group_key": "department", "value_key": "salary"}}
      {{"type": "mom_growth", "date_key": "date_col", "value_key": "amount"}}
      {{"type": "rank_within_group", "group_key": "department", "rank_key": "salary"}}

  IMPORTANT: Use "group_avg" (not "group_sum") whenever the user asks for average or mean values.
  IMPORTANT: When the user asks to compare an overall metric between domains (e.g., "compare average
  salary between tech_startup and airport") use group_keys: [] (empty list) — this produces a single
  overall average for this domain. Only use group_keys: ["department"] when the user explicitly asks
  for a breakdown by department or category.

Rules:
  - Each SQL task fetches RAW data (no aggregation SQL needed — derived tasks handle that).
  - Use "derived" tasks for grouping, averaging, ranking — not SQL aggregation.
  - Plot tasks always depend on derived or sql tasks.
  - Keep the plan minimal (1-3 SQL tasks maximum).

Example for "average salary by department in the {domain} domain" (user asks for department breakdown):
[
  {{"id": "t1", "description": "fetch employee_id, name, department, salary from {domain}_employees", \
"task_type": "sql", "depends_on": []}},
  {{"id": "t2", "description": "compute average salary grouped by department", \
"task_type": "derived", "depends_on": ["t1"], \
"operation": {{"type": "group_avg", "group_keys": ["department"], "value_key": "salary"}}}},
  {{"id": "t3", "description": "bar chart of average salary by department", \
"task_type": "plot", "depends_on": ["t2"]}}
]

Example for "compare average salary between domains" (user wants ONE number per domain, no breakdown):
[
  {{"id": "t1", "description": "fetch employee_id, name, salary from {domain}_employees", \
"task_type": "sql", "depends_on": []}},
  {{"id": "t2", "description": "compute overall average salary", \
"task_type": "derived", "depends_on": ["t1"], \
"operation": {{"type": "group_avg", "group_keys": [], "value_key": "salary"}}}}
]
"""


_UPLOADS_PLANNER_PROMPT_TEMPLATE = """\
You are a task decomposition planner for USER-UPLOADED datasets.

The user has uploaded the following tables. Use ONLY these exact table and column names:
{tables_block}

Rules:
- Reference tables by their exact name shown above (they start with `user_`).
- SQL tasks fetch RAW rows only — name the table and the columns to select.
- Do NOT aggregate in SQL. Use a "derived" task for averages/sums/ranking.
- Do NOT invent columns that are not listed above.

Task Types:
1. "sql": Fetch raw rows from one uploaded table.
2. "derived": Aggregate an upstream SQL result in Python (zero DB calls).
3. "plot": Visualise an upstream result.

Output a valid JSON array of task objects, each with:
  - "id": unique string like "t1", "t2"
  - "description": specific action (e.g. "fetch region, amount from user_sales")
  - "task_type": "sql", "derived", or "plot"
  - "depends_on": array of prerequisite task IDs (empty if independent)
  - "operation": (derived only) one of:
      {{"type": "group_avg", "group_keys": ["col"], "value_key": "num_col"}}
      {{"type": "group_sum", "group_keys": ["col"], "value_key": "num_col"}}
      {{"type": "top_n_per_group", "partition_key": "col", "rank_key": "num_col", "n": 5}}
      {{"type": "contribution_pct", "group_key": "col", "value_key": "num_col"}}
      {{"type": "rank_within_group", "group_key": "col", "rank_key": "num_col"}}

Keep the plan minimal (1-3 SQL tasks). Use group_avg for averages, group_sum for totals.
"""


class ChildTaskPlanner:
    """
    LLM-powered task decomposition planner for a specific domain.

    Converts a natural language user request into a structured DAG of TaskNodes.
    """

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self._llm: ChatGroq | None = None

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=4096,
            )
        return self._llm

    async def plan(self, user_request: str) -> list[TaskNode]:
        """
        Decompose a user request into a list of TaskNodes.

        Uses the async LLM API (`ainvoke`) so that multiple child planners can
        run concurrently via asyncio.gather without blocking the event loop.

        Args:
            user_request: Free-form natural language user request.

        Returns:
            List of TaskNode objects forming a DAG.
        """
        logger.info("[ChildPlanner][%s] Decomposing request: %r", self.domain, user_request)

        system_prompt = self._system_prompt()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User request (plan ONLY for the {self.domain} domain): {user_request}"),
        ]

        response = await self.llm.ainvoke(messages)
        raw = response.content.strip()
        logger.debug("[ChildPlanner][%s] LLM raw output:\n%s", self.domain, raw)

        tasks = self._parse_tasks(raw)
        logger.info("[ChildPlanner][%s] Decomposed into %d tasks: %s", self.domain, len(tasks), [t.id for t in tasks])
        return tasks

    def _system_prompt(self) -> str:
        """
        Build the planner system prompt for this domain.

        Built-in domains use the static employee/flights template. The uploads
        domain is schema-driven: the prompt is generated from the actual columns
        of whatever the user uploaded, so the LLM references real table names.
        """
        if self.domain == UPLOADS_DOMAIN:
            from schema.dynamic_schema import get_uploads_schema_defs

            defs = get_uploads_schema_defs()
            if not defs:
                tables_block = "(no uploaded tables)"
            else:
                lines = []
                for d in defs:
                    cols = ", ".join(f"{c['name']} ({c['type']})" for c in d["columns"])
                    lines.append(f"- {d['table']}: {cols}")
                tables_block = "\n".join(lines)
            return _UPLOADS_PLANNER_PROMPT_TEMPLATE.format(tables_block=tables_block)

        return _PLANNER_SYSTEM_PROMPT_TEMPLATE.format(domain=self.domain)

    def _parse_tasks(self, raw: str) -> list[TaskNode]:
        """Parse LLM JSON output into TaskNode list with robust error handling."""
        try:
            # Find the first '[' and last ']' to extract the JSON array
            start_idx = raw.index("[")
            end_idx = raw.rindex("]") + 1
            cleaned = raw[start_idx:end_idx]
            data: list[dict[str, Any]] = json.loads(cleaned)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("[ChildPlanner][%s] Failed to parse LLM output as JSON: %s\nRaw: %s", self.domain, exc, raw[:500])
            # Attempt to fix common truncation issues: if the JSON ends with an incomplete field, try to close it
            try:
                start_idx = raw.index("[")
                # Find last complete object or array closing
                for end_idx in range(len(raw) - 1, start_idx, -1):
                    test_str = raw[start_idx:end_idx] + "]"
                    try:
                        data = json.loads(test_str)
                        logger.info("[ChildPlanner][%s] Recovered incomplete JSON (truncated response)", self.domain)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    raise ValueError("Could not recover JSON from truncated response")
            except Exception:
                logger.error("[ChildPlanner][%s] JSON recovery failed; using fallback task", self.domain)
                # Final fallback: create a single catch-all SQL task with better description context
                return [TaskNode(
                    id=f"{self.domain}_t1",
                    description=f"retrieve and analyze comprehensive employee data from the {self.domain} domain",
                    task_type="sql",
                    domain=self.domain,
                    depends_on=[]
                )]

        nodes = []
        for item in data:
            raw_id = str(item.get("id", f"t{len(nodes)+1}"))
            # Prefix the ID with the domain to avoid collisions when merging DAGs
            domain_id = f"{self.domain}_{raw_id}"

            # Also update depends_on to use the prefixed IDs
            depends_on = [f"{self.domain}_{str(d)}" for d in item.get("depends_on", [])]

            nodes.append(TaskNode(
                id=domain_id,
                description=str(item.get("description", "unknown task")),
                task_type=str(item.get("task_type", "sql")),
                domain=self.domain,
                depends_on=depends_on,
                operation=item.get("operation")
            ))
        return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_dag(tasks: list[TaskNode]) -> bool:
    """
    Validate that the task graph has no cycles and all depends_on IDs exist.

    Returns True if valid, raises ValueError if invalid.
    """
    task_ids = {t.id for t in tasks}
    for task in tasks:
        for dep in task.depends_on:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {task.id!r} depends on unknown task {dep!r}. "
                    f"Available IDs: {task_ids}"
                )
    # Topological sort to detect cycles
    _topological_sort(tasks)  # Raises if cycle detected
    return True


def sanitize_dag(tasks: list[TaskNode]) -> list[TaskNode]:
    """
    Deterministically repair common LLM planning mistakes so the DAG is safe to
    execute, WITHOUT an extra LLM round-trip.

    Repairs applied (each logged as a warning):
      1. Drop ``depends_on`` IDs that reference a non-existent task (dangling edges).
      2. Drop ``derived``/``plot`` tasks that have no valid upstream dependency —
         they have nothing to compute or visualise from and would only error.
      3. Iterates to a fixpoint: removing a task can orphan its dependents, so we
         repeat until no further repairs are needed.

    SQL tasks are never dropped (they are the data sources). Returns the cleaned
    list, preserving original order.
    """
    cleaned = list(tasks)

    while True:
        valid_ids = {t.id for t in cleaned}
        changed = False
        next_tasks: list[TaskNode] = []

        for task in cleaned:
            # 1. Prune dangling dependency edges.
            pruned_deps = [d for d in task.depends_on if d in valid_ids]
            if len(pruned_deps) != len(task.depends_on):
                dropped = set(task.depends_on) - set(pruned_deps)
                logger.warning(
                    "[sanitize_dag] Task %s references unknown deps %s — dropping those edges",
                    task.id, dropped,
                )
                task.depends_on = pruned_deps
                changed = True

            # 2. A derived/plot task with no surviving upstream cannot run.
            if task.task_type in ("derived", "plot") and not task.depends_on:
                logger.warning(
                    "[sanitize_dag] Dropping %s task %s — it has no valid upstream dependency",
                    task.task_type, task.id,
                )
                changed = True
                continue  # drop it

            next_tasks.append(task)

        cleaned = next_tasks
        if not changed:
            break

    return cleaned


def _topological_sort(tasks: list[TaskNode]) -> list[str]:
    """Kahn's algorithm — returns execution order or raises on cycle."""
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in tasks}

    for task in tasks:
        for dep in task.depends_on:
            adj[dep].append(task.id)
            in_degree[task.id] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(tasks):
        raise ValueError("Cycle detected in task dependency graph!")

    return order
