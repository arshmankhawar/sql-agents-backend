"""
agents/plot_agent.py — Plot Agent (Blackboard Consumer Only).

The Plot Agent NEVER queries PostgreSQL directly.

It reads results from the Blackboard shared result cache and generates
visualization payloads (chart data structures) ready for a frontend.

This directly addresses Problem 5: Plot Agent re-querying data.

Supported chart types (generated from shared SQL results):
  - bar_chart
  - line_chart
  - pie_chart
  - table (raw tabular data)
"""

import logging
from typing import Any

from blackboard.result_cache import cache_get

logger = logging.getLogger(__name__)


class PlotAgent:
    """
    Visualization agent that consumes shared results from the Blackboard.

    Never touches PostgreSQL — always reads from the result cache.
    """

    def __init__(self, agent_id: str = "plot_agent"):
        self.agent_id = agent_id

    async def generate_chart(
        self,
        query_hash: str,
        chart_type: str = "bar_chart",
        x_key: str | None = None,
        y_key: str | None = None,
        title: str = "Chart",
    ) -> dict[str, Any]:
        """
        Read results from the Blackboard and generate a chart payload.

        Args:
            query_hash: Hash of the SQL query whose results to visualize.
            chart_type: One of "bar_chart", "line_chart", "pie_chart", "table".
            x_key:      Column to use as the X-axis (auto-detected if None).
            y_key:      Column to use as the Y-axis / value (auto-detected if None).
            title:      Chart title.

        Returns:
            Chart payload dict ready for a frontend renderer.
        """
        # Read from Blackboard — NEVER from PostgreSQL
        result = await cache_get(query_hash)

        if result is None:
            logger.warning(
                "[%s] No cached result found for hash=%s", self.agent_id, query_hash[:12]
            )
            return {
                "error": f"No result available for query_hash={query_hash[:12]}",
                "chart_type": chart_type,
                "source": "blackboard",
            }

        rows = result.get("rows", [])

        logger.info(
            "[%s] Generating %s from %d cached rows (hash=%s)",
            self.agent_id, chart_type, len(rows), query_hash[:12],
        )

        # Auto-detect column keys if not provided
        if not rows:
            logger.warning("[plot_agent] No rows found for hash=%s", query_hash)
            return {
                "source": "blackboard",
                "chart_type": "none",
                "row_count": 0,
                "file": None
            }

        if rows and (x_key is None or y_key is None):
            cols = list(rows[0].keys())
            x_key = x_key or cols[0]
            # Try to find a numeric column for y-axis
            numeric_cols = [
                c for c in cols
                if isinstance(rows[0].get(c), (int, float)) and c != x_key
            ]
            y_key = y_key or (numeric_cols[0] if numeric_cols else cols[-1])

        chart_payload = self._build_payload(rows, chart_type, x_key, y_key, title, query_hash)
        return chart_payload

    def _build_payload(
        self,
        rows: list[dict[str, Any]],
        chart_type: str,
        x_key: str,
        y_key: str,
        title: str,
        query_hash: str,
    ) -> dict[str, Any]:
        """Build the chart-specific payload structure."""
        base = {
            "chart_type": chart_type,
            "title": title,
            "source": "blackboard",          # Confirms no DB was queried
            "query_hash": query_hash[:12],
            "row_count": len(rows),
        }

        if chart_type in ("bar_chart", "line_chart"):
            base["labels"] = [str(r.get(x_key, "")) for r in rows]
            base["datasets"] = [{
                "label": y_key,
                "data": [r.get(y_key, 0) for r in rows],
            }]

        elif chart_type == "pie_chart":
            base["labels"] = [str(r.get(x_key, "")) for r in rows]
            base["values"] = [r.get(y_key, 0) for r in rows]

        elif chart_type == "table":
            base["columns"] = list(rows[0].keys()) if rows else []
            base["rows"] = rows

        else:
            base["raw"] = rows

        return base

    async def generate_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Generate a textual summary from a list of agent results.

        All results are read from the Blackboard (already in memory in the result dicts).
        No additional DB queries are made.

        Args:
            results: List of result dicts from SQL agents (each has "rows", "task", etc.)

        Returns:
            Summary dict with aggregated statistics per task.
        """
        summary = {
            "agent_id": self.agent_id,
            "source": "blackboard",
            "tasks": [],
        }

        for res in results:
            rows = res.get("rows", [])
            task_summary = {
                "task": res.get("task", "unknown"),
                "query_hash": res.get("query_hash", "")[:12],
                "source": res.get("source"),    # cache / owner / subscriber
                "row_count": len(rows),
                "elapsed_ms": res.get("elapsed_ms"),
            }

            # Compute numeric column stats
            if rows:
                numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
                for k in numeric_keys:
                    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                    if vals:
                        task_summary[f"{k}_sum"] = round(sum(vals), 2)
                        task_summary[f"{k}_avg"] = round(sum(vals) / len(vals), 2)
                        task_summary[f"{k}_max"] = max(vals)

            summary["tasks"].append(task_summary)

        logger.info("[%s] Generated summary for %d tasks", self.agent_id, len(results))
        return summary
