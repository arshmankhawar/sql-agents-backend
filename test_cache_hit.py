"""
Test cache hit behaviour:
  - Run the same query twice.
  - First run = owner (DB execution).
  - Second run = cache hit (zero DB calls).
"""
import asyncio
import sys

async def main():
    from agents.base_agent import run_with_blackboard
    from blackboard.client import close_redis

    SQL = "SELECT * FROM orders WHERE year = 2025;"

    print("Run 1 (cold):")
    r1 = await run_with_blackboard("test_agent_1", "orders cold", SQL)
    print(f"  source={r1['source']}  rows={r1['row_count']}  {r1['elapsed_ms']:.0f}ms")

    print("Run 2 (same query, expect cache hit):")
    r2 = await run_with_blackboard("test_agent_2", "orders warm", SQL)
    print(f"  source={r2['source']}  rows={r2['row_count']}  {r2['elapsed_ms']:.0f}ms")

    assert r1["source"] == "owner",  f"Expected 'owner', got {r1['source']}"
    assert r2["source"] == "cache",  f"Expected 'cache', got {r2['source']}"
    assert r1["rows"] == r2["rows"], "Row mismatch between runs!"
    print("\nAll assertions passed.")

    await close_redis()

asyncio.run(main())
