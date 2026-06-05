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
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import run_with_blackboard
from config import GROQ_API_KEY, GROQ_MODEL
from schema.retriever import get_retriever

logger = logging.getLogger(__name__)

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

Examples:
- Task: "fetch employee_id, name, department, salary from employees"
  → SELECT employee_id, name, department, salary FROM employees;
- Task: "fetch all flight records"
  → SELECT * FROM flights;
- Task: "fetch project name, status and budget from projects"
  → SELECT name, status, budget FROM projects;
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

    async def _generate_sql(self, task: str, schema_context: str) -> str:
        """Use the Groq LLM to generate a SQL query for the given task."""
        user_message = f"Domain: {self.domain}\nTask: {task}\n\n{schema_context}"
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

        # Step 2: Generate SQL via LLM (async — does not block the event loop)
        sql = await self._generate_sql(task, schema_context)

        # Step 3: Execute via Blackboard (handles deduplication, caching, coordination)
        result = await run_with_blackboard(self.agent_id, task, sql, domain=self.domain)
        
        # Add schema tokens to result for metrics tracking
        result["schema_tokens"] = schema_tokens

        return result
