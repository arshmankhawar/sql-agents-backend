"""
comparison_benchmark.py — Practical Comparison: Baseline vs Improved Architecture.

This script benchmarks both systems against realistic scenarios where:
  1. Multiple agents request the same data (tests deduplication).
  2. Multiple agents request different data (tests parallelism and schema context).
  3. Multi-domain queries (tests domain isolation and orchestration).

Metrics:
  - Total DB queries executed
  - Duplicate query elimination
  - Total schema tokens sent to LLM
  - Execution time
  - Cache effectiveness
"""

import asyncio
import logging
import sys
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy loggers
for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("comparison")


# ────────────────────────────────────────────────────────────────────────────
# Test Scenarios
# ────────────────────────────────────────────────────────────────────────────

SCENARIO_1 = {
    "name": "Scenario 1: Duplicate Queries (Test Deduplication)",
    "description": "5 agents all request 'average salary by department'",
    "domain": "airport",
    "tasks": [
        "average salary by department",
        "average salary by department",  # Duplicate
        "average salary by department",  # Duplicate
        "average salary by department",  # Duplicate
        "average salary by department",  # Duplicate
    ],
}

SCENARIO_2 = {
    "name": "Scenario 2: Overlapping Queries (Test Partial Deduplication)",
    "description": "Queries with overlapping data requirements",
    "domain": "tech_startup",
    "tasks": [
        "average salary by department",
        "total salary by department",  # Same table (employees), different aggregation
        "count of employees",  # Same table (employees), different aggregation
        "list of projects",  # Different table (projects)
        "total budget of active projects",  # Same table (projects) as task 4
    ],
}

SCENARIO_3 = {
    "name": "Scenario 3: Multi-Domain (Test Domain Isolation)",
    "description": "Multi-domain parent request → child agents per domain",
    "domain": "multi_domain",
    "tasks": [
        ("airport", "average salary by department"),
        ("airport", "delayed flights count"),
        ("tech_startup", "average salary by department"),  # Duplicate SQL but different domain
        ("tech_startup", "list of projects"),
    ],
}


async def run_baseline_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run scenario using simple baseline (no blackboard, no deduplication)."""
    from baseline_simple import SimpleBaselineOrchestrator
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BASELINE: {scenario['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"  {scenario['description']}")
    
    if scenario["domain"] == "multi_domain":
        # For multi-domain, run each domain separately
        all_results = []
        total_db_queries = 0
        total_schema_tokens = 0
        
        t0 = time.perf_counter()
        
        for domain, task in scenario["tasks"]:
            orch = SimpleBaselineOrchestrator(domain=domain)
            result = await orch.run_tasks([(task, None)])
            all_results.append(result)
            total_db_queries += result["total_db_queries"]
            total_schema_tokens += result["total_schema_tokens"]
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "system": "baseline",
            "scenario": scenario["name"],
            "elapsed_ms": elapsed_ms,
            "total_db_queries": total_db_queries,
            "duplicate_queries": 0,
            "cache_hits": 0,
            "total_schema_tokens": total_schema_tokens,
        }
    else:
        orch = SimpleBaselineOrchestrator(domain=scenario["domain"])
        result = await orch.run_tasks([(task, None) for task in scenario["tasks"]])
        return result


async def run_improved_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run scenario using improved multi-agent system."""
    from planner.parent_planner import ParentOrchestrator
    from planner.task_planner import validate_dag
    from dag.executor import DAGExecutor
    
    logger.info(f"\n{'='*70}")
    logger.info(f"IMPROVED: {scenario['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"  {scenario['description']}")
    
    if scenario["domain"] == "multi_domain":
        # For multi-domain, use parent orchestrator
        request = "Compare " + " and ".join([f"{domain}: {task}" for domain, task in scenario["tasks"]])
    else:
        # For single domain, use parent orchestrator to test consistency
        request = "Compare " + " and ".join(scenario["tasks"])
    
    orchestrator = ParentOrchestrator()
    executor = DAGExecutor()
    
    t0 = time.perf_counter()
    
    # Phase 1: Plan the DAG
    tasks = await orchestrator.plan_global_dag(request)
    validate_dag(tasks)
    
    # Phase 2: Execute the DAG
    results = await executor.execute(tasks)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Count metrics from results
    db_queries = 0
    cache_hits = 0
    owner_executions = 0
    total_schema_tokens = 0
    
    for task_id, result in results.items():
        if result.get("source") == "owner":
            db_queries += 1
            owner_executions += 1
        elif result.get("source") == "subscriber":
            db_queries += 1  # Still counted but from cache coordination
        elif result.get("source") == "cache":
            cache_hits += 1
        total_schema_tokens += result.get("schema_tokens", 0)
    
    # Only count actual DB executions (owners)
    actual_db_queries = owner_executions
    
    return {
        "system": "improved",
        "scenario": scenario["name"],
        "elapsed_ms": elapsed_ms,
        "total_db_queries": actual_db_queries,
        "duplicate_queries": len(scenario["tasks"]) - actual_db_queries,
        "cache_hits": cache_hits,
        "total_schema_tokens": total_schema_tokens,
    }


async def main():
    """Run all scenarios and compare results."""
    scenarios = [SCENARIO_1, SCENARIO_2]
    
    for scenario in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# {scenario['name']}")
        print(f"{'#'*70}")
        
        # Run baseline
        try:
            baseline_result = await run_baseline_scenario(scenario)
            logger.info(f"\nBaseline Results:")
            logger.info(f"  Total Execution Time: {baseline_result['elapsed_ms']:.0f} ms")
            logger.info(f"  DB Queries Executed: {baseline_result['total_db_queries']}")
            logger.info(f"  Schema Tokens Sent: {baseline_result['total_schema_tokens']}")
        except Exception as e:
            logger.error(f"Baseline failed: {e}", exc_info=True)
            baseline_result = None
        
        # Run improved
        try:
            improved_result = await run_improved_scenario(scenario)
            logger.info(f"\nImproved Results:")
            logger.info(f"  Total Execution Time: {improved_result['elapsed_ms']:.0f} ms")
            logger.info(f"  DB Queries Executed: {improved_result['total_db_queries']}")
            logger.info(f"  Cache Hits: {improved_result['cache_hits']}")
            logger.info(f"  Schema Tokens Sent: {improved_result['total_schema_tokens']}")
        except Exception as e:
            logger.error(f"Improved failed: {e}", exc_info=True)
            improved_result = None
        
        # Compare
        if baseline_result and improved_result:
            logger.info(f"\n{'='*70}")
            logger.info(f"COMPARISON RESULTS")
            logger.info(f"{'='*70}")
            
            time_improvement = ((baseline_result["elapsed_ms"] - improved_result["elapsed_ms"]) / baseline_result["elapsed_ms"] * 100) if baseline_result["elapsed_ms"] > 0 else 0
            query_reduction = baseline_result["total_db_queries"] - improved_result["total_db_queries"]
            token_reduction = ((baseline_result["total_schema_tokens"] - improved_result["total_schema_tokens"]) / baseline_result["total_schema_tokens"] * 100) if baseline_result["total_schema_tokens"] > 0 else 0
            
            logger.info(f"\n⏱️  EXECUTION TIME:")
            logger.info(f"  Baseline: {baseline_result['elapsed_ms']:.0f} ms")
            logger.info(f"  Improved: {improved_result['elapsed_ms']:.0f} ms")
            logger.info(f"  ✓ Improvement: {time_improvement:.1f}% faster")
            
            logger.info(f"\n🗄️  DB QUERIES:")
            logger.info(f"  Baseline: {baseline_result['total_db_queries']} queries")
            logger.info(f"  Improved: {improved_result['total_db_queries']} queries")
            logger.info(f"  ✓ Reduction: {query_reduction} fewer queries ({(query_reduction/baseline_result['total_db_queries']*100):.1f}% fewer)")
            logger.info(f"  ✓ Deduplication Rate: {(query_reduction/baseline_result['total_db_queries']*100):.1f}%")
            
            logger.info(f"\n💾 SCHEMA TOKENS (LLM Context):")
            logger.info(f"  Baseline: {baseline_result['total_schema_tokens']} tokens")
            logger.info(f"  Improved: {improved_result['total_schema_tokens']} tokens")
            if improved_result['total_schema_tokens'] > 0:
                logger.info(f"  ✓ Reduction: {token_reduction:.1f}% fewer tokens")
            
            logger.info(f"\n📊 KEY METRICS:")
            logger.info(f"  Cache Hits (Improved): {improved_result['cache_hits']}")
            logger.info(f"  Duplicate Queries Eliminated: {query_reduction}")
            
    logger.info(f"\n\n{'='*70}")
    logger.info("Comparison Complete")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
