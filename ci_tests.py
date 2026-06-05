"""
ci_tests.py — Lightweight deterministic unit tests for CI.

Validates the pure-Python aggregation logic in agents/derived_agent.py — the
arithmetic that the entire pipeline relies on for averages, sums, and rankings.
These tests make ZERO database, LLM, or FAISS calls, so they run fast and
deterministically on every push without any secrets or generated artifacts.

Run locally:  python ci_tests.py
Exit code 0 = all passed; non-zero = a failure (fails the CI job).
"""

from agents.derived_agent import _group_avg, _group_sum, _top_n_per_group

_failures: list[str] = []


def check(name: str, got, expected) -> None:
    if got == expected:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}\n        expected: {expected}\n        got:      {got}")
        _failures.append(name)


def test_group_avg_overall() -> None:
    """Empty group_keys → one overall average row."""
    rows = [{"salary": 100}, {"salary": 200}, {"salary": 300}]
    result = _group_avg(rows, [], "salary")
    check("group_avg overall", result, [{"avg_salary": 200.0, "count": 3}])


def test_group_avg_by_key() -> None:
    """Grouping by a key → one average row per group, sorted by avg desc."""
    rows = [
        {"dept": "Eng", "salary": 100},
        {"dept": "Eng", "salary": 200},
        {"dept": "HR", "salary": 60},
    ]
    result = _group_avg(rows, ["dept"], "salary")
    check(
        "group_avg by dept",
        result,
        [
            {"dept": "Eng", "avg_salary": 150.0, "count": 2},
            {"dept": "HR", "avg_salary": 60.0, "count": 1},
        ],
    )


def test_group_sum() -> None:
    """group_sum totals the value per group."""
    rows = [
        {"dept": "Eng", "salary": 100},
        {"dept": "Eng", "salary": 200},
    ]
    result = _group_sum(rows, ["dept"], "salary")
    check("group_sum", result, [{"dept": "Eng", "total_salary": 300.0, "count": 2}])


def test_top_n_per_group() -> None:
    """top_n_per_group keeps the N highest-ranked rows per partition."""
    rows = [
        {"dept": "Eng", "salary": 100},
        {"dept": "Eng", "salary": 300},
        {"dept": "Eng", "salary": 200},
    ]
    result = _top_n_per_group(rows, "dept", "salary", n=2)
    salaries = [r["salary"] for r in result]
    check("top_n_per_group", salaries, [300, 200])


if __name__ == "__main__":
    print("Running CI unit tests (pure Python, no external dependencies)...\n")
    test_group_avg_overall()
    test_group_avg_by_key()
    test_group_sum()
    test_top_n_per_group()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} test(s) failed: {_failures}")
        raise SystemExit(1)
    print("All CI unit tests passed.")
