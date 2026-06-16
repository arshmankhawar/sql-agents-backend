"""
agents/sql_agent.py — Full SQL Agent with LLM + Blackboard Integration.

Each SQL Agent:
  1. Receives a task description from the DAG executor.
  2. Retrieves ONLY the relevant schema tables (via FAISS semantic search).
  3. Uses a Groq-powered LLM to generate the SQL query.
  4. Executes the query through the Blackboard (deduplication + caching).
  5. Returns the result dict for downstream consumption.

The LLM prompt is intentionally minimal because the schema context is
pre-filtered to only the relevant tables — solving Problem 3 (excessive context).
"""

import logging
import re
import sqlite3
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import run_with_blackboard
from config import GROQ_API_KEY, GROQ_MODEL
from schema.retriever import get_retriever

logger = logging.getLogger(__name__)

# Max number of SQL generation attempts. The first is the initial generation;
# each subsequent attempt feeds the previous failure back to the LLM to fix.
_MAX_SQL_ATTEMPTS = 3

# ─────────────────────────────────────────────────────────────────────────────
# LLM initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """Return a Groq LLM instance."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment.\n"
            "Get a free key at: https://console.groq.com/\n"
            "The system requires a valid Groq API key to generate SQL queries."
        )
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=512,
    )


_SYSTEM_PROMPT = """\
You are a SQL query writer for a SQLite database. Given a task description and the relevant \
table schema, write a single SQL SELECT statement.

Rules:
- Output ONLY the raw SQL. No markdown, no explanation, no ```sql blocks.
- Use ONLY the table and column names shown in the schema — do not invent columns.
- Do NOT add WHERE clauses that filter by domain name, company name, or any value not \
  present in the data (e.g., never write WHERE domain = '...' or WHERE company = '...').
- Fetch RAW rows — do not aggregate (no GROUP BY, AVG, SUM). The derived agent will aggregate.
- Use standard SQLite syntax (no YEAR(), no ::cast, no schema-qualified names).
- End the query with a semicolon.

Examples (use the exact prefixed table name shown in the schema):
- Task: "fetch employee_id, name, department, salary from airport_employees"
  → SELECT employee_id, name, department, salary FROM airport_employees;
- Task: "fetch all flight records"
  → SELECT * FROM airport_flights;
- Task: "fetch project name, status and budget from tech_startup_projects"
  → SELECT name, status, budget FROM tech_startup_projects;
"""


def _extract_sql(llm_response: str) -> str:
    """
    Extract a clean SQL string from an LLM response.
    Handles cases where the model wraps the SQL in markdown code blocks.
    """
    # Strip markdown code blocks if present
    cleaned = re.sub(r"```(?:sql)?\s*", "", llm_response, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    # Take only the first statement if multiple are returned
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if statements:
        return statements[0] + ";"
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# SQL Agent class
# ─────────────────────────────────────────────────────────────────────────────

class SQLAgent:
    """
    A single SQL Agent that uses the Blackboard for coordination.

    Can be instantiated multiple times (A, B, C...) — each gets a unique ID
    but they all share the same Blackboard, ensuring deduplication works
    across all concurrent instances.
    """

    def __init__(self, agent_id: str | None = None, domain: str = "default"):
        self.agent_id = agent_id or f"sql_agent_{uuid.uuid4().hex[:6]}"
        self.domain = domain
        self._llm: ChatGroq | None = None
        self._retriever = get_retriever(self.domain)
        logger.info("[SQLAgent] Created agent_id=%s domain=%s", self.agent_id, self.domain)

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = _get_llm()
        return self._llm

    async def _generate_sql(
        self,
        task: str,
        schema_context: str,
        prior_sql: str | None = None,
        prior_error: str | None = None,
    ) -> str:
        """
        Use the Groq LLM to generate a SQL query for the given task.

        When ``prior_sql`` and ``prior_error`` are supplied (a previous attempt
        failed at the database), they are appended to the prompt so the model can
        reflect on its own mistake and correct it — the self-correcting agent loop.
        """
        user_message = f"Domain: {self.domain}\nTask: {task}\n\n{schema_context}"
        if prior_sql and prior_error:
            user_message += (
                f"\n\nYour previous attempt FAILED. Fix it.\n"
                f"Previous SQL: {prior_sql}\n"
                f"Database error: {prior_error}\n"
                f"Re-read the schema above carefully — only use columns/tables that exist. "
                f"Output the corrected SQL only."
            )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = await self.llm.ainvoke(messages)
        raw_sql = response.content.strip()
        sql = _extract_sql(raw_sql)
        logger.info("[%s] Generated SQL: %.80r", self.agent_id, sql)
        return sql

    async def run(self, task: str) -> dict[str, Any]:
        """
        Execute the full agent pipeline for a given task.

        Implements a self-correcting loop: if the generated SQL fails at the
        database (e.g. a hallucinated column name), the agent feeds the error
        back to the LLM and regenerates, up to ``_MAX_SQL_ATTEMPTS`` times.

        Args:
            task: Natural language task description (e.g. "revenue by region for 2025").

        Returns:
            Result dict from the Blackboard coordination layer.
        """
        logger.info("[%s] Task received: %r", self.agent_id, task)

        # Step 1: Retrieve relevant schema (not the full schema!)
        schema_context = self._retriever.format_schema_context(task)
        logger.debug("[%s] Schema context:\n%s", self.agent_id, schema_context[:400])

        # Count schema tokens (rough estimate: ~4 chars per token)
        schema_tokens = len(schema_context) // 4

        prior_sql: str | None = None
        prior_error: str | None = None
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_SQL_ATTEMPTS + 1):
            # Step 2: Generate SQL via LLM (async — does not block the event loop).
            # On retries, prior_sql/prior_error steer the model toward a fix.
            sql = await self._generate_sql(task, schema_context, prior_sql, prior_error)

            # Step 3: Execute via Blackboard (handles dedup, caching, coordination).
            try:
                result = await run_with_blackboard(self.agent_id, task, sql, domain=self.domain)
            except sqlite3.Error as exc:
                # The SQL was syntactically/semantically invalid at the DB. Capture
                # the real error, hand it back to the LLM, and try again. A failed
                # owner claim for this (bad) SQL hash simply expires via its TTL.
                last_exc = exc
                prior_sql = sql
                prior_error = str(exc)
                logger.warning(
                    "[%s] SQL attempt %d/%d failed: %s — retrying with feedback",
                    self.agent_id, attempt, _MAX_SQL_ATTEMPTS, exc,
                )
                continue

            # Success — annotate metrics and return.
            result["schema_tokens"] = schema_tokens
            result["sql_attempts"] = attempt
            return result

        # All attempts exhausted — surface the last DB error to the caller.
        logger.error(
            "[%s] Exhausted %d SQL attempts; last error: %s",
            self.agent_id, _MAX_SQL_ATTEMPTS, last_exc,
        )
        raise last_exc  # type: ignore[misc]
