"""JSONL tool-call tracer. Every agent tool writes one line to logs/agent_trace.jsonl."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from src.config import project_root

TRACE_PATH = project_root() / "logs" / "agent_trace.jsonl"
OUTPUT_LIMIT = 200


def trace_path():
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return TRACE_PATH


def _truncate(value: Any, limit: int = OUTPUT_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def read_trace(limit: int = 40) -> list[dict[str, Any]]:
    path = trace_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _result_was_cached(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(data, dict) and data.get("cached"))


def traced(fn: Callable) -> Callable:
    """Log tool name, inputs, truncated output, duration, and errors."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": fn.__name__.removeprefix("_"),
            "inputs": {**kwargs} if not args else {"args": [str(a) for a in args], **kwargs},
            "ok": True,
        }
        try:
            result = fn(*args, **kwargs)
            record["output"] = _truncate(result)
            record["cached"] = _result_was_cached(result)
            return result
        except Exception as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            path = trace_path()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

    return wrapper
