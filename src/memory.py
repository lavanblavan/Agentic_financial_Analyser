"""Short-term session memory plus a persistent tool cache.

Follow-ups reuse the last DataBrief / FinalReport. Tool results live on disk
so a second run for the same ticker does not re-hit Yahoo or the LLM.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from src.config import project_root
from src.ticker import normalize_symbol

CACHE_DIR = project_root() / "logs" / "cache"
SESSION_PATH = project_root() / "logs" / "session.json"

TOOL_TTL_SECONDS = {
    "_get_price_data": 20 * 60,
    "_calculate_volatility": 20 * 60,
    "_get_news": 15 * 60,
    "_llm_sentiment": 30 * 60,
    "_web_search": 15 * 60,
}
BRIEF_TTL_SECONDS = 45 * 60
MAX_TURNS = 20

_FOLLOWUP = re.compile(
    r"\b(hedge|remind|again|previous|last (run|report|brief)|same (one|ticker|name)|"
    r"why (that|those|the)|those risks|the risks|what about|follow[- ]?up)\b",
    re.I,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)


def cache_key(tool: str, arguments: dict[str, Any]) -> str:
    payload = dict(arguments)
    if "ticker" in payload and payload["ticker"] is not None:
        payload["ticker"] = normalize_symbol(str(payload["ticker"]))
    if "headlines" in payload and isinstance(payload["headlines"], list):
        payload["headlines"] = payload["headlines"][:12]
    raw = json.dumps({"tool": tool, "args": payload}, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{tool.lstrip('_')}-{digest}"


def cache_get(tool: str, arguments: dict[str, Any], ttl: int) -> str | None:
    _ensure_dirs()
    path = CACHE_DIR / f"{cache_key(tool, arguments)}.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    age = time.time() - float(record.get("ts") or 0)
    if age > ttl:
        return None
    payload = record.get("payload")
    return payload if isinstance(payload, str) else None


def cache_put(tool: str, arguments: dict[str, Any], payload: str) -> None:
    _ensure_dirs()
    path = CACHE_DIR / f"{cache_key(tool, arguments)}.json"
    path.write_text(
        json.dumps({"ts": time.time(), "tool": tool, "args": arguments, "payload": payload}, default=str),
        encoding="utf-8",
    )


def mark_cached(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if isinstance(data, dict):
        data["cached"] = True
        return json.dumps(data, default=str)
    return payload


def disk_cached(fn: Callable) -> Callable:
    """Return a prior tool result when the same args are still fresh."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        ttl = TOOL_TTL_SECONDS.get(fn.__name__, 15 * 60)
        hit = cache_get(fn.__name__, arguments, ttl)
        if hit is not None:
            return mark_cached(hit)
        result = fn(*args, **kwargs)
        if isinstance(result, str):
            cache_put(fn.__name__, arguments, result)
        return result

    return wrapper


def _empty_session() -> dict[str, Any]:
    return {
        "session_id": uuid.uuid4().hex[:12],
        "thread_id": f"research-{uuid.uuid4().hex[:8]}",
        "updated_at": _now(),
        "last_ticker": None,
        "last_query": None,
        "tickers": {},
        "turns": [],
    }


def load_session() -> dict[str, Any]:
    _ensure_dirs()
    if not SESSION_PATH.exists():
        return _empty_session()
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_session()
    if not isinstance(data, dict):
        return _empty_session()
    data.setdefault("tickers", {})
    data.setdefault("turns", [])
    data.setdefault("session_id", uuid.uuid4().hex[:12])
    data.setdefault("thread_id", f"research-{uuid.uuid4().hex[:8]}")
    return data


def save_session(session: dict[str, Any]) -> None:
    _ensure_dirs()
    session["updated_at"] = _now()
    SESSION_PATH.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")


def recall(ticker: str | None = None, session: dict[str, Any] | None = None) -> dict[str, Any] | None:
    session = session or load_session()
    symbol = normalize_symbol(ticker) if ticker else session.get("last_ticker")
    if not symbol:
        return None
    stored = (session.get("tickers") or {}).get(symbol)
    return stored if isinstance(stored, dict) else None


def _fresh(stored: dict[str, Any], ttl: int = BRIEF_TTL_SECONDS) -> bool:
    ts = stored.get("ts")
    if not ts:
        return False
    try:
        when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - when).total_seconds()
    return age <= ttl


def remember(result: dict[str, Any], session: dict[str, Any] | None = None) -> dict[str, Any]:
    session = session or load_session()
    ticker = result.get("ticker") or (result.get("brief") or {}).get("ticker")
    if ticker:
        ticker = normalize_symbol(str(ticker))
        session["last_ticker"] = ticker
        session["tickers"][ticker] = {
            "ts": _now(),
            "query": result.get("query"),
            "brief": result.get("brief"),
            "report": result.get("report"),
            "tool_calls": result.get("tool_calls") or (result.get("agent_a") or {}).get("tool_calls") or [],
        }
    session["last_query"] = result.get("query")
    turns = list(session.get("turns") or [])
    turns.append(
        {
            "ts": _now(),
            "query": result.get("query"),
            "ticker": ticker,
            "from_memory": bool(result.get("from_memory")),
            "revisions": result.get("revisions"),
        }
    )
    session["turns"] = turns[-MAX_TURNS:]
    save_session(session)
    return session


def is_followup(query: str, session: dict[str, Any] | None = None) -> bool:
    session = session or load_session()
    if not session.get("last_ticker"):
        return False
    text = (query or "").strip()
    if not text:
        return False
    lower = text.lower()
    if "analyse" in lower or "analyze" in lower:
        return False
    if _FOLLOWUP.search(text):
        return True
    if len(text) < 90 and "?" in text:
        return True
    return False


def describe_memory(session: dict[str, Any] | None = None) -> dict[str, Any]:
    session = session or load_session()
    cache_files = list(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
    last = recall(session.get("last_ticker"), session)
    return {
        "session_id": session.get("session_id"),
        "thread_id": session.get("thread_id"),
        "last_ticker": session.get("last_ticker"),
        "last_query": session.get("last_query"),
        "known_tickers": sorted((session.get("tickers") or {}).keys()),
        "turns": len(session.get("turns") or []),
        "brief_cached": bool(last and last.get("brief")),
        "report_cached": bool(last and last.get("report")),
        "tool_cache_files": len(cache_files),
        "session_path": str(SESSION_PATH),
    }


def _answer_from_store(query: str, stored: dict[str, Any]) -> str:
    brief = stored.get("brief") or {}
    report = stored.get("report") or {}
    ticker = brief.get("ticker") or stored.get("query") or "?"
    lines = [
        f"Follow-up from memory ({ticker}). Tools were not called again.",
        f"Price {brief.get('current_price')} | 30d vol {brief.get('vol_30d_pct')}% | "
        f"90d vol {brief.get('vol_90d_pct')}%",
        "",
        "Hedge:",
        report.get("hedge_or_strategy") or brief.get("hedge_strategy") or "(none stored)",
        "",
        "Risks:",
    ]
    risks = report.get("top_risks") or brief.get("quantitative_risks") or []
    for i, risk in enumerate(risks[:3], start=1):
        if isinstance(risk, dict):
            lines.append(f"  {i}. {risk.get('name')} — {risk.get('evidence')}")
    lines.extend(["", f"Your question: {query.strip()}"])
    return "\n".join(lines)


def answer_followup(query: str, session: dict[str, Any] | None = None) -> dict[str, Any]:
    session = session or load_session()
    stored = recall(session.get("last_ticker"), session)
    if not stored or not stored.get("brief"):
        raise ValueError("No stored brief yet. Run a full research prompt first.")
    text = _answer_from_store(query, stored)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.agent_a import FallbackChat, _extract_json_object
        from src.config import load_settings

        settings = load_settings()
        if settings.llm_ready:
            llm = FallbackChat(settings, tools=False)
            reply = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Answer the follow-up using only the stored DataBrief and FinalReport. "
                            "Do not invent new prices. If they ask for the hedge or risks, quote them. "
                            "Say you used session memory and did not call tools."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\n"
                            f"Brief: {json.dumps(stored.get('brief'), default=str)[:8000]}\n"
                            f"Report: {json.dumps(stored.get('report'), default=str)[:8000]}"
                        )
                    ),
                ]
            )
            content = getattr(reply, "content", "")
            if isinstance(content, str) and content.strip():
                extracted = _extract_json_object(content)
                text = extracted.get("answer") if extracted and extracted.get("answer") else content.strip()
    except Exception:
        pass
    result = {
        "query": query,
        "ticker": session.get("last_ticker"),
        "from_memory": True,
        "reused": ["brief", "report"],
        "brief": stored.get("brief"),
        "report": stored.get("report"),
        "followup_answer": text,
        "tool_calls": [],
        "revisions": 0,
        "critiques": [],
    }
    remember(result, session)
    return result


def ask(query: str, agent_a_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Memory-aware entry: follow-up from disk, or a full A+B run that then persists."""
    session = load_session()
    if is_followup(query, session) and recall(session.get("last_ticker"), session):
        return answer_followup(query, session)

    from src.agent_b import run_two_agents
    from src.ticker import parse_research_query

    parsed = parse_research_query(query)
    reused = agent_a_result
    if reused is None:
        stored = recall(parsed["ticker"], session)
        if stored and stored.get("brief") and _fresh(stored):
            reused = {
                "brief": stored["brief"],
                "ticker": parsed["ticker"],
                "tool_calls": stored.get("tool_calls") or [],
                "missing_after_run": [],
            }
    result = run_two_agents(query, agent_a_result=reused)
    result["from_memory"] = bool(reused and not agent_a_result)
    result["memory"] = describe_memory()
    remember(result)
    return result


def cache_dir() -> Path:
    _ensure_dirs()
    return CACHE_DIR
