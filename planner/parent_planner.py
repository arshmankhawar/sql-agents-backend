"""
planner/parent_planner.py — LLM-Based Global Parent Orchestrator.

The Parent Orchestrator receives the initial user request, determines which
database domains are required, and delegates to the ChildTaskPlanner for each.
It then merges their task DAGs into a single global execution DAG.
If cross-domain comparison is needed, it adds a final global task.
"""

import asyncio
import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, GROQ_MODEL
from planner.task_planner import ChildTaskPlanner, TaskNode, sanitize_dag

logger = logging.getLogger(__name__)


_PARENT_SYSTEM_PROMPT = """\
You are the Parent Orchestrator for a multi-domain data analytics system.

Available domains are:
- "airport" (tables: airport_employees, airport_flights)
- "tech_startup" (tables: tech_startup_employees, tech_startup_projects)
- "restaurant" (tables: restaurant_employees, restaurant_menus)

Given a user request, you must:
1. Identify which domains are needed to fulfill the request.
2. Determine if a global cross-domain comparison or plot task is required at the end.

If the request does not explicitly name a domain, infer the most plausible domain from the query content.
If the request refers to multiple domains or explicitly compares domains, include both.
If the request is generic and does not clearly imply a specific domain, include all domains only when the user asks for cross-domain comparison or overall cross-domain analysis.

Output a valid JSON object with:
  - "domains": array of strings (e.g. ["airport", "tech_startup"])
  - "requires_cross_domain_plot": boolean (true if the user wants to compare them)
  - "cross_domain_task_description": string (description of the plot/comparison, or null)

Example: "Compare the average salary of employees in the tech startup with the airport"
{
  "domains": ["tech_startup", "airport"],
  "requires_cross_domain_plot": true,
  "cross_domain_task_description": "plot comparing average employee salary between tech startup and airport"
}

Example: "What is the average salary of employees?"
{
  "domains": ["airport", "tech_startup", "restaurant"],
  "requires_cross_domain_plot": false,
  "cross_domain_task_description": null
}
"""


class ParentOrchestrator:
    """
    Determines required domains and merges child DAGs.
    """

    def __init__(self):
        self._llm: ChatGroq | None = None

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=2048,
            )
        return self._llm

    async def plan_global_dag(self, user_request: str) -> list[TaskNode]:
        """
        Create a unified DAG spanning multiple domains.

        The parent routing call uses the async LLM API, and all child planners
        are run concurrently via asyncio.gather so that an N-domain request pays
        roughly one planning round-trip instead of N serial ones.
        """
        logger.info("[ParentOrchestrator] Analysing global request: %r", user_request)

        # Kick off the (domain-agnostic) embedding-model load in the background so
        # its multi-second cost overlaps with the planning LLM round-trips instead
        # of running serially before execution. Fire-and-forget: the executor's
        # preload will await/reuse the same shared model.
        from schema.retriever import warm_model_async
        # Keep a strong reference on the instance so the loop doesn't GC the task.
        self._warm_task = asyncio.create_task(warm_model_async())

        messages = [
            SystemMessage(content=_PARENT_SYSTEM_PROMPT),
            HumanMessage(content=f"User request: {user_request}"),
        ]

        response = await self.llm.ainvoke(messages)
        raw = response.content.strip()

        try:
            start_idx = raw.index("{")
            end_idx = raw.rindex("}") + 1
            data = json.loads(raw[start_idx:end_idx])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("[ParentOrchestrator] Failed to parse LLM output: %s", exc)
            # Fallback to all domains
            data = {"domains": ["airport", "tech_startup", "restaurant"], "requires_cross_domain_plot": False}

        domains = data.get("domains", [])
        if not domains:
             domains = ["airport", "tech_startup", "restaurant"]

        logger.info("[ParentOrchestrator] Identified domains: %s", domains)

        global_dag: list[TaskNode] = []
        leaf_nodes: list[str] = []

        # Delegate to children CONCURRENTLY — each child planner is an independent
        # LLM round-trip, so we fan them out with asyncio.gather instead of looping
        # serially. Results come back in the same order as `domains`.
        children = [ChildTaskPlanner(domain=domain) for domain in domains]
        child_dags = await asyncio.gather(
            *(child.plan(user_request) for child in children)
        )

        for child_dag in child_dags:
            global_dag.extend(child_dag)

            # Find leaf nodes (nodes that nothing else depends on within this child's DAG)
            all_ids = {t.id for t in child_dag}
            has_dependents = {dep for t in child_dag for dep in t.depends_on}
            leaves = all_ids - has_dependents
            leaf_nodes.extend(list(leaves))

        # Add global cross-domain task if needed
        if data.get("requires_cross_domain_plot") and leaf_nodes:
            desc = data.get("cross_domain_task_description", "Cross-domain comparison plot")
            global_task = TaskNode(
                id="global_plot_1",
                description=desc,
                task_type="plot",
                domain="global",
                depends_on=leaf_nodes,
            )
            global_dag.append(global_task)
            logger.info("[ParentOrchestrator] Added global cross-domain task: %s", desc)

        # Deterministic self-repair pass: prune dangling dependency edges and drop
        # derived/plot tasks left with no upstream. This makes the pipeline robust
        # to imperfect LLM plans without paying for an extra reflection round-trip.
        before = len(global_dag)
        global_dag = sanitize_dag(global_dag)
        if len(global_dag) != before:
            logger.info(
                "[ParentOrchestrator] sanitize_dag repaired plan: %d → %d tasks",
                before, len(global_dag),
            )

        # Degenerate-case guard: if repair emptied the plan (or planning produced
        # nothing), fall back to a catch-all SQL task per identified domain so the
        # request still returns data rather than failing.
        if not global_dag:
            logger.warning("[ParentOrchestrator] Empty DAG after sanitize — injecting catch-all SQL tasks")
            for domain in domains:
                global_dag.append(TaskNode(
                    id=f"{domain}_t1",
                    description=f"retrieve employee data from the {domain} domain",
                    task_type="sql",
                    domain=domain,
                    depends_on=[],
                ))

        return global_dag
