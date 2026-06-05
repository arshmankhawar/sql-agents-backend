"""
Fair benchmark: simple isolated agents vs. Blackboard architecture.

This benchmark intentionally bypasses LLM planning and SQL generation for both
systems. The goal is to compare the part of the architecture that solves the
problem statement: duplicate SQL execution, result sharing, cache reuse, and
domain-safe query coordination.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from agents import base_agent
from agents.base_agent import run_with_blackboard
from blackboard.client import close_redis, get_redis
from db.pool import execute_query as real_execute_query
from utils.sql_normalizer import normalize_and_hash


EMPLOYEE_SQL = "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department;"
PROJECT_SQL = "SELECT status, SUM(budget) AS total_budget FROM projects GROUP BY status;"
FLIGHT_SQL = "SELECT status, COUNT(*) AS flight_count FROM flights GROUP BY status;"
DB_CONCURRENCY_LIMIT = 2


@dataclass(frozen=True)
class QueryTask:
    task: str
    sql: str
    domain: str


@dataclass
class Instrumentation:
    db_calls: int = 0
    calls_by_key: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.calls_by_key is None:
            self.calls_by_key = Counter()
        self._semaphore = asyncio.Semaphore(DB_CONCURRENCY_LIMIT)

    async def execute(self, sql: str, domain: str = "default") -> list[dict[str, Any]]:
        async with self._semaphore:
            self.db_calls += 1
            self.calls_by_key[f"{domain}:{_canonical(sql)}"] += 1
            return await real_execute_query(sql, domain=domain)


def _canonical(sql: str) -> str:
    return " ".join(sql.strip().lower().rstrip(";").split())


def _query_hash(sql: str, domain: str) -> str:
    canonical_sql, _ = normalize_and_hash(sql)
    return hashlib.sha256(f"{domain}:{canonical_sql}".encode("utf-8")).hexdigest()


async def _clear_blackboard_for(tasks: list[QueryTask]) -> None:
    redis = await get_redis()
    for task in tasks:
        query_hash = _query_hash(task.sql, task.domain)
        await redis.delete(f"result:{query_hash}")
        await redis.delete(f"registry:{query_hash}")


def _source_counts(results: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(result.get("source", "unknown")) for result in results)


async def run_baseline(tasks: list[QueryTask], mode: str) -> dict[str, Any]:
    """
    Baseline: every isolated agent executes its SQL directly.

    The baseline has no shared registry, no result cache, and no cross-agent
    communication, so every task is one real DB hit.
    """
    instrumentation = Instrumentation()

    async def one(task: QueryTask) -> dict[str, Any]:
        t0 = time.perf_counter()
        rows = await instrumentation.execute(task.sql, task.domain)
        return {
            "task": task.task,
            "domain": task.domain,
            "sql": task.sql,
            "rows": rows,
            "row_count": len(rows),
            "source": "direct_db",
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    t0 = time.perf_counter()
    if mode == "sequential":
        results = []
        for task in tasks:
            results.append(await one(task))
    else:
        results = await asyncio.gather(*(one(task) for task in tasks))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "system": "baseline",
        "elapsed_ms": elapsed_ms,
        "db_calls": instrumentation.db_calls,
        "calls_by_key": instrumentation.calls_by_key,
        "source_counts": _source_counts(results),
        "results": results,
    }


async def run_improved(tasks: list[QueryTask], mode: str) -> dict[str, Any]:
    """
    Improved: every agent routes through the Blackboard registry/cache.

    Instrumentation patches the DB function imported by agents.base_agent so
    only actual owner or timeout fallback executions are counted.
    """
    await close_redis()
    await _clear_blackboard_for(tasks)
    instrumentation = Instrumentation()
    original_execute_query = base_agent.execute_query
    base_agent.execute_query = instrumentation.execute

    async def one(index: int, task: QueryTask) -> dict[str, Any]:
        return await run_with_blackboard(
            agent_id=f"improved_agent_{index}",
            task_description=task.task,
            sql=task.sql,
            domain=task.domain,
        )

    try:
        t0 = time.perf_counter()
        if mode == "sequential":
            results = []
            for index, task in enumerate(tasks):
                results.append(await one(index, task))
        else:
            results = await asyncio.gather(*(one(index, task) for index, task in enumerate(tasks)))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "system": "improved",
            "elapsed_ms": elapsed_ms,
            "db_calls": instrumentation.db_calls,
            "calls_by_key": instrumentation.calls_by_key,
            "source_counts": _source_counts(results),
            "results": results,
        }
    finally:
        base_agent.execute_query = original_execute_query
        await close_redis()


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "A. Concurrent identical SQL",
        "mode": "concurrent",
        "tasks": [
            QueryTask("average salary by department", EMPLOYEE_SQL, "airport")
            for _ in range(5)
        ],
    },
    {
        "name": "B. Sequential cache reuse",
        "mode": "sequential",
        "tasks": [
            QueryTask("average salary by department", EMPLOYEE_SQL, "airport")
            for _ in range(5)
        ],
    },
    {
        "name": "C. Production overlap mix",
        "mode": "concurrent",
        "tasks": [
            QueryTask("Q1 salary analytics", EMPLOYEE_SQL, "tech_startup"),
            QueryTask("Q2 project budgets", PROJECT_SQL, "tech_startup"),
            QueryTask("Q1 salary analytics", EMPLOYEE_SQL, "tech_startup"),
            QueryTask("Q3 flight status", FLIGHT_SQL, "airport"),
            QueryTask("Q1 salary analytics", EMPLOYEE_SQL, "tech_startup"),
            QueryTask("Q2 project budgets", PROJECT_SQL, "tech_startup"),
            QueryTask("Q3 flight status", FLIGHT_SQL, "airport"),
            QueryTask("Q1 salary analytics", EMPLOYEE_SQL, "tech_startup"),
            QueryTask("Q2 project budgets", PROJECT_SQL, "tech_startup"),
            QueryTask("Q4 airport salary analytics", EMPLOYEE_SQL, "airport"),
        ],
    },
    {
        "name": "D. Domain isolation",
        "mode": "concurrent",
        "tasks": [
            QueryTask("airport salary analytics", EMPLOYEE_SQL, "airport"),
            QueryTask("tech salary analytics", EMPLOYEE_SQL, "tech_startup"),
            QueryTask("airport salary analytics duplicate", EMPLOYEE_SQL, "airport"),
            QueryTask("tech salary analytics duplicate", EMPLOYEE_SQL, "tech_startup"),
        ],
    },
]


def _format_sources(counts: Counter[str]) -> str:
    ordered = ["direct_db", "owner", "subscriber", "cache"]
    parts = [f"{name}={counts[name]}" for name in ordered if counts[name]]
    return ", ".join(parts) if parts else "none"


def _print_result(scenario: dict[str, Any], baseline: dict[str, Any], improved: dict[str, Any]) -> None:
    baseline_calls = baseline["db_calls"]
    improved_calls = improved["db_calls"]
    query_reduction = baseline_calls - improved_calls
    query_reduction_pct = (query_reduction / baseline_calls * 100) if baseline_calls else 0
    baseline_ms = baseline["elapsed_ms"]
    improved_ms = improved["elapsed_ms"]
    time_delta_pct = ((baseline_ms - improved_ms) / baseline_ms * 100) if baseline_ms else 0

    print(f"\n{scenario['name']}")
    print("-" * len(scenario["name"]))
    print(f"Tasks: {len(scenario['tasks'])} ({scenario['mode']})")
    print(f"Baseline: db_calls={baseline_calls}, time={baseline_ms:.1f} ms, sources={_format_sources(baseline['source_counts'])}")
    print(f"Improved: db_calls={improved_calls}, time={improved_ms:.1f} ms, sources={_format_sources(improved['source_counts'])}")
    print(f"DB reduction: {query_reduction}/{baseline_calls} fewer calls ({query_reduction_pct:.1f}%)")
    print(f"Time delta: {time_delta_pct:.1f}%")


async def main() -> None:
    print("Fair Architecture Benchmark")
    print("===========================")
    print("LLM planning and SQL generation are bypassed for both systems.")
    print("Metric focus: actual DB calls, deduplication, cache reuse, and coordination latency.")
    print(f"Both systems use the same simulated DB concurrency limit: {DB_CONCURRENCY_LIMIT}.")

    totals = {
        "baseline_db": 0,
        "improved_db": 0,
        "baseline_ms": 0.0,
        "improved_ms": 0.0,
    }

    for scenario in SCENARIOS:
        baseline = await run_baseline(scenario["tasks"], scenario["mode"])
        improved = await run_improved(scenario["tasks"], scenario["mode"])
        _print_result(scenario, baseline, improved)

        totals["baseline_db"] += baseline["db_calls"]
        totals["improved_db"] += improved["db_calls"]
        totals["baseline_ms"] += baseline["elapsed_ms"]
        totals["improved_ms"] += improved["elapsed_ms"]

    total_reduction = totals["baseline_db"] - totals["improved_db"]
    total_reduction_pct = total_reduction / totals["baseline_db"] * 100
    total_time_delta_pct = (totals["baseline_ms"] - totals["improved_ms"]) / totals["baseline_ms"] * 100

    print("\nOverall")
    print("-------")
    print(f"Baseline DB calls: {totals['baseline_db']}")
    print(f"Improved DB calls: {totals['improved_db']}")
    print(f"Total DB reduction: {total_reduction}/{totals['baseline_db']} fewer calls ({total_reduction_pct:.1f}%)")
    print(f"Baseline total time: {totals['baseline_ms']:.1f} ms")
    print(f"Improved total time: {totals['improved_ms']:.1f} ms")
    print(f"Total time delta: {total_time_delta_pct:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
