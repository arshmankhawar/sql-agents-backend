"""
agents/derived_agent.py — Python-Level Derived Computation Engine.

Derived tasks compute analytics from upstream SQL/derived results that are
already in memory — they make ZERO database calls.

This is the key to achieving the ideal outcome:
  - 3 SQL tasks hit the DB (unique base queries)
  - 5-6 derived tasks compute from shared cached results
  - 3 plot tasks render from those derived results

Supported operations:
  - group_sum         : GROUP BY + SUM (revenue by region, by customer)
  - top_n_per_group   : TOP N within each partition (top 10 customers per region)
  - contribution_pct  : % contribution within group (customer share of region revenue)
  - mom_growth        : Month-over-month growth (date_key → month extraction + diff)
  - rank_within_group : RANK() within each group (customer rankings)
  - merge_join        : Join two upstream results on a key
  - filter_top_n      : Global top-N by a sort key
"""

import logging
import time
from collections import defaultdict
from typing import Any

from planner.task_planner import TaskNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation primitives (pure Python, no DB)
# ─────────────────────────────────────────────────────────────────────────────

def _group_sum(rows: list[dict], group_keys: list[str], value_key: str) -> list[dict]:
    """GROUP BY group_keys, SUM(value_key) ORDER BY total DESC."""
    groups: dict[tuple, float] = defaultdict(float)
    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups[key] += float(row.get(value_key, 0) or 0)
        counts[key] += 1
    result = []
    for key, total in sorted(groups.items(), key=lambda x: -x[1]):
        entry = dict(zip(group_keys, key))
        entry[f"total_{value_key}"] = round(total, 2)
        entry["count"] = counts[key]
        result.append(entry)
    return result


def _group_avg(rows: list[dict], group_keys: list[str], value_key: str) -> list[dict]:
    """GROUP BY group_keys, AVG(value_key) ORDER BY avg DESC."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        val = row.get(value_key)
        if val is not None:
            groups[key].append(float(val))
    result = []
    for key, vals in sorted(groups.items(), key=lambda x: -(sum(x[1]) / len(x[1]) if x[1] else 0)):
        entry = dict(zip(group_keys, key))
        entry[f"avg_{value_key}"] = round(sum(vals) / len(vals), 2) if vals else 0.0
        entry["count"] = len(vals)
        result.append(entry)
    return result


def _top_n_per_group(
    rows: list[dict], partition_key: str, rank_key: str, n: int = 10
) -> list[dict]:
    """TOP n rows per partition group, ranked by rank_key DESC."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(partition_key)].append(row)
    result = []
    for group_val, group_rows in sorted(groups.items()):
        top = sorted(
            group_rows,
            key=lambda r: float(r.get(rank_key, 0) or 0),
            reverse=True,
        )[:n]
        for rank, row in enumerate(top, 1):
            result.append({**row, "rank_in_group": rank})
    return result


def _contribution_pct(
    rows: list[dict], group_key: str, value_key: str
) -> list[dict]:
    """Add contribution_pct column = value / group_total * 100."""
    group_totals: dict[Any, float] = defaultdict(float)
    for row in rows:
        group_totals[row.get(group_key)] += float(row.get(value_key, 0) or 0)
    result = []
    for row in rows:
        g = row.get(group_key)
        v = float(row.get(value_key, 0) or 0)
        total = group_totals[g]
        pct = round(v / total * 100, 2) if total > 0 else 0.0
        result.append({**row, "contribution_pct": pct, "group_total": round(total, 2)})
    return sorted(result, key=lambda r: (-float(r.get("group_total", 0)), -float(r.get(value_key, 0))))


def _mom_growth(rows: list[dict], date_key: str, value_key: str) -> list[dict]:
    """
    Extract month from date_key (YYYY-MM-DD) and compute month-over-month growth.
    Works on raw order rows — extracts and groups by month number.
    """
    monthly: dict[int, float] = defaultdict(float)
    for row in rows:
        date_str = str(row.get(date_key, "") or "")
        if len(date_str) >= 7:
            try:
                month = int(date_str[5:7])
            except ValueError:
                month = int(row.get("month", 0) or 0)
        else:
            month = int(row.get("month", 0) or 0)
        monthly[month] += float(row.get(value_key, 0) or 0)

    sorted_months = sorted(monthly.keys())
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    result = []
    for i, m in enumerate(sorted_months):
        curr = monthly[m]
        prev = monthly[sorted_months[i - 1]] if i > 0 else None
        growth = round((curr - prev) / prev * 100, 2) if prev and prev != 0 else None
        result.append({
            "month": m,
            "month_name": month_names.get(m, str(m)),
            f"total_{value_key}": round(curr, 2),
            "mom_growth_pct": growth,
        })
    return result


def _rank_within_group(
    rows: list[dict], group_key: str, rank_key: str
) -> list[dict]:
    """Assign integer rank within each group (1 = highest rank_key value)."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(group_key)].append(row)
    result = []
    for group_val, group_rows in sorted(groups.items()):
        sorted_rows = sorted(
            group_rows,
            key=lambda r: float(r.get(rank_key, 0) or 0),
            reverse=True,
        )
        for rank, row in enumerate(sorted_rows, 1):
            result.append({**row, "region_rank": rank})
    return result


def _merge_join(
    left: list[dict], right: list[dict], on: str
) -> list[dict]:
    """Simple hash join on a shared key column."""
    right_index: dict[Any, dict] = {r.get(on): r for r in right}
    result = []
    for row in left:
        key_val = row.get(on)
        if key_val in right_index:
            merged = {**right_index[key_val], **row}  # left wins on conflict
            result.append(merged)
    return result


def _filter_top_n(rows: list[dict], sort_key: str, n: int = 10) -> list[dict]:
    """Global top-N by sort_key DESC."""
    return sorted(rows, key=lambda r: float(r.get(sort_key, 0) or 0), reverse=True)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Derived task dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def compute_derived(
    task: TaskNode,
    prior_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute a derived computation task using upstream results from the Blackboard.

    Args:
        task:          The TaskNode with task_type="derived" and an operation dict.
        prior_results: Map of task_id → result dict (from already-completed tasks).

    Returns:
        Result dict matching the format used by SQL agents.
    """
    t0 = time.perf_counter()
    op = task.operation or {}
    op_type = op.get("type", "passthrough")

    def _get_rows(task_id: str) -> list[dict]:
        """Safely extract rows from a prior result."""
        r = prior_results.get(task_id, {})
        return r.get("rows", [])

    # ── Dispatch ──────────────────────────────────────────────────────────────
    try:
        if op_type == "group_avg":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _group_avg(_get_rows(source), op["group_keys"], op["value_key"])

        elif op_type == "group_sum":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _group_sum(_get_rows(source), op["group_keys"], op["value_key"])

        elif op_type == "top_n_per_group":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _top_n_per_group(
                _get_rows(source),
                op["partition_key"],
                op["rank_key"],
                op.get("n", 10),
            )

        elif op_type == "contribution_pct":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _contribution_pct(_get_rows(source), op["group_key"], op["value_key"])

        elif op_type == "mom_growth":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _mom_growth(_get_rows(source), op.get("date_key", "order_date"), op["value_key"])

        elif op_type == "rank_within_group":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _rank_within_group(_get_rows(source), op["group_key"], op["rank_key"])

        elif op_type == "merge_join":
            left_rows = _get_rows(op["left_task"])
            right_rows = _get_rows(op["right_task"])
            rows = _merge_join(left_rows, right_rows, op["on"])

        elif op_type == "filter_top_n":
            source = op.get("source_task") or (task.depends_on[0] if task.depends_on else "")
            rows = _filter_top_n(_get_rows(source), op["sort_key"], op.get("n", 10))

        else:
            # Passthrough — return upstream rows unchanged
            source = task.depends_on[0] if task.depends_on else ""
            rows = _get_rows(source)
            logger.warning("[DerivedAgent] Unknown op %r for task %s — passthrough", op_type, task.id)

    except Exception as exc:
        logger.error("[DerivedAgent] Error in task %s (op=%s): %s", task.id, op_type, exc, exc_info=True)
        rows = []

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[DerivedAgent] task=%s  op=%s  rows=%d  %.1fms",
        task.id, op_type, len(rows), elapsed,
    )

    return {
        "agent_id": "derived_agent",
        "task": task.description,
        "task_id": task.id,
        "source": "derived",          # Zero DB calls
        "operation": op_type,
        "derived_from": task.depends_on,
        "rows": rows,
        "row_count": len(rows),
        "query_hash": f"derived:{task.id}",
        "elapsed_ms": round(elapsed, 1),
    }
