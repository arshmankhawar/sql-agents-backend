"""
utils/logging_config.py — Structured, correlated logging for the pipeline.

Two goals:
  1. Correlation — every log line emitted while handling one user request carries
     the same `request_id`, so a single pipeline run can be traced end-to-end
     across the planner, DAG executor, SQL agents, Blackboard and synthesis.
     The id is stored in a ContextVar, which asyncio tasks and asyncio.to_thread
     workers inherit automatically — so it propagates without threading an
     argument through every function.
  2. Structure — logs are written as JSON to a rotating file (logs/pipeline.log)
     for easy parsing/aggregation, while the console keeps a human-readable
     format for `pm2 logs`.

Call configure_logging() once at startup. Call new_request_id() at the start of
each request to stamp all of that request's logs.
"""

import contextvars
import datetime as _dt
import json
import logging
import logging.handlers
import os
import uuid

# Holds the current request id; default "-" for logs outside any request.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

# Standard LogRecord attributes — anything else on a record is treated as a
# structured "extra" field and included in the JSON output.
_RESERVED = set(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime", "request_id", "taskName"}


def new_request_id() -> str:
    """Generate a short request id, bind it to the current context, and return it."""
    rid = uuid.uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_var.get()


class _RequestIdFilter(logging.Filter):
    """Inject the current request id onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, _dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        # Merge any structured extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
) -> None:
    """
    Configure root logging with a human-readable console handler and a
    JSON rotating file handler, both stamped with the request id.

    Idempotent: safe to call more than once (handlers are reset each call).
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any handlers a prior basicConfig()/call installed.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    rid_filter = _RequestIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(request_id)s]  %(name)s  %(message)s"
        )
    )
    console.addFilter(rid_filter)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonFormatter())
    file_handler.addFilter(rid_filter)
    root.addHandler(file_handler)
