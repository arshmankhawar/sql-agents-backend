"""
Test plot agent never queries the DB:
  - Run a SQL agent to populate the Blackboard.
  - Run the plot agent using only the cached result.
  - Verify plot_agent never calls execute_query.
"""
import asyncio

async def main():
    from agents.base_agent import run_with_blackboard
    from agents.plot_agent import PlotAgent
    from blackboard.client import close_redis

    SQL = "SELECT * FROM customers;"

    # Populate Blackboard with one SQL result
    result = await run_with_blackboard("sql_populate", "customers data", SQL)
    query_hash = result["query_hash"]
    print(f"Populated cache: hash={query_hash[:12]}  rows={result['row_count']}")

    # Plot Agent should read from cache ONLY
    plot = PlotAgent()

    db_called = False
    original_execute = None
    try:
        import db.pool as pool_module
        original_execute = pool_module.execute_query

        async def mock_execute(sql, domain=None):
            nonlocal db_called
            db_called = True
            raise AssertionError("PlotAgent must NOT call execute_query!")

        pool_module.execute_query = mock_execute

        chart = await plot.generate_chart(query_hash, chart_type="bar_chart", title="Customer Chart")
        summary = await plot.generate_summary([result])

    finally:
        if original_execute:
            pool_module.execute_query = original_execute

    assert not db_called, "PlotAgent called the database!"
    assert chart["source"] == "blackboard", f"Expected source=blackboard, got {chart['source']}"
    assert chart["row_count"] == result["row_count"]
    print(f"Chart generated: type={chart['chart_type']}  rows={chart['row_count']}  source={chart['source']}")
    print(f"Summary tasks: {len(summary['tasks'])}")
    print("\nAll assertions passed — PlotAgent never touched the DB.")

    await close_redis()

asyncio.run(main())
