"""
main.py — Entry Point for the Multi-Agent SQL System.

Full pipeline:
  Client Request
       |
       v
  Global Task Planner  (LLM decomposes request → DAG of TaskNodes)
       |
       v
  DAG Executor  (runs SQL agents in parallel, sequences dependents)
       |
       v
  Blackboard  (deduplicates queries, caches results, notifies subscribers)
       |
       v
  SQL Agents  (retrieve relevant schema → generate SQL → execute via Blackboard)
       |
       v
  Plot Agent  (reads Blackboard → generates charts, never touches DB)
       |
       v
  Final Output

Run modes:
  python main.py                    → full pipeline demo (requires GROQ_API_KEY in .env)
  python main.py --dedup-demo       → concurrent deduplication demo (no LLM needed)
  python main.py --build-index      → (re)build the FAISS schema index and exit
"""

import argparse
import asyncio
import logging
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup — structured, colourised output
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Suppress noisy third-party loggers
for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Concurrent deduplication (no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

async def run_dedup_demo() -> None:
    """
    Demonstrate the core deduplication guarantee:

      Five concurrent agents all request the same SQL query.
      Expected: exactly ONE database execution, all others get cached/subscribed result.

    This demo requires NO Groq API key — it queries the real SQLite airport DB directly.
    """
    from agents.base_agent import run_with_blackboard
    from blackboard.client import close_redis

    print("\n" + "=" * 70)
    print("  DEDUPLICATION DEMO")
    print("  5 agents request the SAME query concurrently")
    print("  Expected: 1 DB execution, 4 reuses")
    print("=" * 70 + "\n")

    # All 5 agents want the same data — only 1 should hit the DB.
    # agent_delta uses different whitespace/casing but same semantics → same hash.
    SQL = "SELECT * FROM employees;"
    SQL_VARIANT = "select  *  from  employees ;"  # normalises to same hash

    agents = [
        ("agent_alpha",   SQL),
        ("agent_beta",    SQL),         # identical
        ("agent_gamma",   SQL),         # identical
        ("agent_delta",   SQL_VARIANT), # whitespace variant → same hash after normalisation
        ("agent_epsilon", SQL),         # identical
    ]

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[run_with_blackboard(aid, "all employees", sql, domain="airport") for aid, sql in agents]
    )
    total_ms = (time.perf_counter() - t0) * 1000

    print("\n── Results ──────────────────────────────────────────────────────────")
    sources = {}
    for r in results:
        src = r["source"]
        sources[src] = sources.get(src, 0) + 1
        print(f"  {r['agent_id']:<20} source={src:<12} rows={r['row_count']:<5} {r['elapsed_ms']:.0f}ms")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  DB executions:  {sources.get('owner', 0)}  (should be 1)")
    print(f"  Subscribers:    {sources.get('subscriber', 0)}")
    print(f"  Cache hits:     {sources.get('cache', 0)}")
    print(f"  Total wall time: {total_ms:.0f}ms")
    print("=" * 70 + "\n")

    await close_redis()


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline demo (requires GROQ_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_pipeline(user_request: str) -> None:
    """Run the complete multi-agent pipeline for a user request."""
    from blackboard.client import close_redis
    from planner.parent_planner import ParentOrchestrator
    from planner.task_planner import validate_dag
    from dag.executor import DAGExecutor
    from agents.synthesis_agent import SynthesisAgent

    print("\n" + "=" * 70)
    print("  MULTI-AGENT SQL SYSTEM  (Blackboard + FAISS + Streaming DAG)")
    print(f"  Query: {user_request}")
    print("=" * 70)

    # ── Phase 1: Planning ─────────────────────────────────────────────────────
    planner = ParentOrchestrator()
    t0 = time.perf_counter()
    tasks = await planner.plan_global_dag(user_request)
    validate_dag(tasks)
    plan_ms = (time.perf_counter() - t0) * 1000

    sql_tasks   = [t for t in tasks if t.task_type == "sql"]
    deriv_tasks = [t for t in tasks if t.task_type == "derived"]
    print(f"\n  Plan  : {len(tasks)} tasks in {plan_ms:.0f}ms "
          f"({len(sql_tasks)} SQL, {len(deriv_tasks)} derived, "
          f"{len(tasks)-len(sql_tasks)-len(deriv_tasks)} plot)")

    # ── Phase 2: DAG Execution ────────────────────────────────────────────────
    executor = DAGExecutor()
    t1 = time.perf_counter()
    results = await executor.execute(tasks)
    exec_ms = (time.perf_counter() - t1) * 1000

    # Collect DB-call stats from SQL results
    sources = [results[t.id].get("source") for t in sql_tasks if t.id in results]
    db_calls     = sources.count("owner")
    cache_hits   = sources.count("cache")
    subscribers  = sources.count("subscriber")
    schema_tokens = sum(
        results[t.id].get("schema_tokens", 0) for t in sql_tasks if t.id in results
    )

    print(f"  Execute: {exec_ms:.0f}ms | "
          f"DB calls: {db_calls} | cache hits: {cache_hits} | "
          f"subscribers: {subscribers} | schema tokens: {schema_tokens}")

    # ── Phase 3: Natural Language Answer ──────────────────────────────────────
    t2 = time.perf_counter()
    synthesis = SynthesisAgent()
    answer = await synthesis.synthesize(user_request, tasks, results)
    synth_ms = (time.perf_counter() - t2) * 1000

    total_ms = plan_ms + exec_ms + synth_ms

    print("\n" + "-" * 70)
    print("  ANSWER")
    print("-" * 70)
    # Word-wrap to 68 chars for clean terminal output
    import textwrap
    for line in textwrap.wrap(answer, width=68):
        print("  " + line)
    print("-" * 70)
    print(f"  Total: {total_ms:.0f}ms  "
          f"(plan {plan_ms:.0f}ms + exec {exec_ms:.0f}ms + synthesis {synth_ms:.0f}ms)")
    print("=" * 70 + "\n")

    await close_redis()


# ─────────────────────────────────────────────────────────────────────────────
# Build index command
# ─────────────────────────────────────────────────────────────────────────────

def run_build_index() -> None:
    """Build or rebuild the FAISS schema index."""
    from schema.indexer import build_index
    print("\nBuilding FAISS schema index...")
    build_index()
    print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent SQL System with Blackboard Architecture"
    )
    parser.add_argument(
        "--dedup-demo",
        action="store_true",
        help="Run the concurrent deduplication demo (no API key needed)",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build/rebuild the FAISS schema index and exit",
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Show me revenue trends by region and top customers by lifetime value for 2025, then plot both.",
        help="User request to process through the full pipeline",
    )
    args = parser.parse_args()

    if args.build_index:
        run_build_index()
        return

    if args.dedup_demo:
        asyncio.run(run_dedup_demo())
        return

    # Full pipeline (requires GROQ_API_KEY)
    asyncio.run(run_full_pipeline(args.request))


if __name__ == "__main__":
    main()
