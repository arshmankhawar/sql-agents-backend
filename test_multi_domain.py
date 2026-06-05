import asyncio

async def main():
    from planner.parent_planner import ParentOrchestrator
    from dag.executor import DAGExecutor
    from blackboard.client import close_redis

    request = "Compare average employee salary between tech_startup and airport."

    print("\n▶ Planning across domains")
    planner = ParentOrchestrator()
    tasks = await planner.plan_global_dag(request)

    domains = {t.domain for t in tasks}
    print(" Identified domains in DAG:", domains)

    assert "tech_startup" in domains and "airport" in domains, "Expected both domains in the plan"

    print("\n▶ Executing DAG")
    executor = DAGExecutor()
    results = await executor.execute(tasks)

    # Collect SQL task query_hashes
    sql_tasks = [t for t in tasks if t.task_type == "sql"]
    hashes = [results[t.id]["query_hash"] for t in sql_tasks if t.id in results]

    print("\n▶ Results summary")
    for t in sql_tasks:
        if t.id in results:
            r = results[t.id]
            print(f"  {t.id}  domain={t.domain}  rows={r['row_count']}  hash={r['query_hash'][:12]}...")

    # Ensure that queries across different domains do not collide
    assert len(set(hashes)) == len(hashes), "Expected domain-specific query hash isolation"

    await close_redis()

if __name__ == "__main__":
    asyncio.run(main())
