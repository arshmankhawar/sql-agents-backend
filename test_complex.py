import asyncio
import json
import time

async def run_complex_test():
    from blackboard.client import close_redis
    from planner.parent_planner import ParentOrchestrator
    from planner.task_planner import validate_dag
    from dag.executor import DAGExecutor

    user_request = (
        "For 2025, calculate revenue by region, revenue by customer within each region, "
        "top 10 customers per region, customer contribution percentages, month-over-month growth, "
        "customer rankings, and generate charts for all results."
    )
    
    print("\n" + "=" * 80)
    print("  COMPLEX DEPENDENCY TEST")
    print(f"  Request: {user_request}")
    print("=" * 80 + "\n")

    print("▶ Phase 1: Planning (LLM DAG Generation)")
    planner = ParentOrchestrator()
    t0 = time.perf_counter()
    tasks = await planner.plan_global_dag(user_request)
    validate_dag(tasks)
    plan_ms = (time.perf_counter() - t0) * 1000

    print(f"  Decomposed into {len(tasks)} tasks in {plan_ms:.0f}ms:")
    sql_count = 0
    derived_count = 0
    plot_count = 0
    for task in tasks:
        if task.task_type == "sql": sql_count += 1
        elif task.task_type == "derived": derived_count += 1
        elif task.task_type == "plot": plot_count += 1
        
        deps = f" [depends: {task.depends_on}]" if task.depends_on else ""
        print(f"    [{task.task_type.upper()}] {task.id}: {task.description}{deps}")
        if task.operation:
            print(f"      └─ Op: {task.operation.get('type')}")

    print(f"\n  Summary: SQL={sql_count}, Derived={derived_count}, Plot={plot_count}")

    print("\n▶ Phase 2: DAG Execution")
    executor = DAGExecutor()
    t1 = time.perf_counter()
    results = await executor.execute(tasks)
    exec_ms = (time.perf_counter() - t1) * 1000

    print(f"\n▶ Phase 3: Results")
    db_executions = 0
    cache_hits = 0
    derived_hits = 0
    
    for task_id, result in results.items():
        if result.get("source") == "owner": db_executions += 1
        elif result.get("source") == "cache": cache_hits += 1
        elif result.get("source") == "derived": derived_hits += 1
        elif result.get("source") == "subscriber": cache_hits += 1 # functionally same for this metric

    print(f"  Total Tasks Created:        {len(tasks)}")
    print(f"  SQL Tasks:                  {sql_count}")
    print(f"  Derived Tasks:              {derived_count}")
    print(f"  Plot Tasks:                 {plot_count}")
    print(f"  Database Executions:        {db_executions}")
    print(f"  Cache/Subscriber Hits:      {cache_hits}")
    print(f"  Derived Operations Run:     {derived_hits}")
    print(f"  Execution Time:             {exec_ms:.0f}ms")
    
    await close_redis()

if __name__ == "__main__":
    asyncio.run(run_complex_test())
