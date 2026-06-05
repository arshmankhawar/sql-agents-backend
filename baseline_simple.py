"""
baseline_simple.py — Simple Baseline Architecture (No Blackboard, No Deduplication).

This is the naive "isolated agents" approach where:
  - Each SQL agent independently generates SQL from the full schema.
  - Each concurrent agent hits the DB directly (no deduplication).
  - No result caching.
  - Full schema context sent to every agent.

Used for benchmarking against the improved multi-agent system.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, GROQ_MODEL
from db.pool import execute_query
from schema.indexer import MOCK_SCHEMAS
import re

logger = logging.getLogger(__name__)


class SimpleBaselineAgent:
    """
    A naive SQL agent with NO blackboard, NO caching, NO semantic retrieval.
    Every instance generates SQL independently from the FULL schema.
    """

    def __init__(self, agent_id: str | None = None, domain: str = "default"):
        self.agent_id = agent_id or f"baseline_agent_{uuid.uuid4().hex[:6]}"
        self.domain = domain
        self._llm: ChatGroq | None = None
        self.db_query_count = 0
        self.schema_tokens = 0

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=512,
            )
        return self._llm

    def _full_schema_context(self) -> str:
        """Generate FULL schema context (all tables) for the domain."""
        schema_defs = MOCK_SCHEMAS.get(self.domain, [])
        lines = ["## Full Database Schema\n"]
        for table_def in schema_defs:
            cols = ", ".join(
                f"{c['name']} {c['type'].upper()}"
                for c in table_def.get("columns", [])
            )
            lines.append(f"### {table_def['table']}")
            lines.append(f"-- {table_def.get('description', '')}")
            lines.append(f"CREATE TABLE {table_def['table']} ({cols});\n")
        context = "\n".join(lines)
        # Count tokens (rough estimate: ~4 chars per token)
        self.schema_tokens = len(context) // 4
        return context

    def _generate_sql(self, task: str) -> str:
        """Use LLM to generate SQL from FULL schema."""
        full_schema = self._full_schema_context()

        system_prompt = """\
You are a SQL query writer. Given a task and the full database schema, generate SQL.
Output ONLY the raw SQL statement.
"""
        user_message = f"Task: {task}\n\n{full_schema}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        raw_sql = response.content.strip()
        # Clean markdown if present
        cleaned = re.sub(r"```(?:sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        if statements:
            return statements[0] + ";"
        return cleaned

    async def run(self, task: str) -> dict[str, Any]:
        """Execute task with NO blackboard coordination."""
        logger.info("[%s] Task: %r", self.agent_id, task)

        t0 = time.perf_counter()

        # Step 1: Always generate SQL from full schema
        sql = self._generate_sql(task)
        logger.info("[%s] Generated SQL: %.60r", self.agent_id, sql)

        # Step 2: Always execute directly (no deduplication, no caching)
        self.db_query_count += 1
        rows = await execute_query(sql, domain=self.domain)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "agent_id": self.agent_id,
            "task": task,
            "sql": sql,
            "rows": rows,
            "row_count": len(rows),
            "source": "direct_db",
            "elapsed_ms": elapsed_ms,
            "schema_tokens": self.schema_tokens,
            "db_queries": self.db_query_count,
        }


class SimpleBaselineOrchestrator:
    """Naive orchestrator: spawn N agents, run them all concurrently."""

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self.agents: list[SimpleBaselineAgent] = []

    async def run_tasks(self, tasks: list[tuple[str, str]]) -> dict[str, Any]:
        """
        Run a list of (task_description, sql_query) tuples concurrently.

        Args:
            tasks: List of (description, sql) tuples.

        Returns:
            Aggregated metrics and results.
        """
        self.agents = [SimpleBaselineAgent(domain=self.domain) for _ in range(len(tasks))]

        logger.info("[Baseline] Spawning %d agents to run %d tasks concurrently", len(self.agents), len(tasks))

        t0 = time.perf_counter()

        results = await asyncio.gather(
            *[agent.run(task) for agent, (task, _) in zip(self.agents, tasks)],
            return_exceptions=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Aggregate metrics
        total_db_queries = sum(1 for _ in results)  # Each agent always queries DB once
        total_schema_tokens = sum(r["schema_tokens"] for r in results)

        return {
            "system": "baseline_simple",
            "elapsed_ms": elapsed_ms,
            "total_agents": len(self.agents),
            "total_tasks": len(tasks),
            "total_db_queries": total_db_queries,
            "duplicate_queries": 0,  # No deduplication
            "cache_hits": 0,
            "total_schema_tokens": total_schema_tokens,
            "results": results,
        }
