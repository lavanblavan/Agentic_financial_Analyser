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
from src.ticker import normalize_symbol, ticker_mentioned

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

_RECALL = re.compile(
    r"\b(hedge|risks?|sentiment|remind|again|previous|you said|last (run|report|brief)|"
    r"same (one|ticker|name)|why (that|those|the)|those|what about|follow[- ]?up)\b",
    re.I,
)
_REFRESH = re.compile(
    r"\b(latest|update|refresh|right now|today'?s price|now what)\b",
    re.I,
)
_EXTEND = re.compile(
    r"\b(search|catalyst|analyst|more news|what happened|tariff|earnings|why is)\b",
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
            "task_mode": result.get("task_mode"),
            "brief": result.get("brief"),
            "report": result.get("report"),
            "followup_answer": result.get("followup_answer"),
            "tool_calls": result.get("tool_calls") or (result.get("agent_a") or {}).get("tool_calls") or [],
        }
    session["last_query"] = result.get("query")
    session["last_answer"] = (
        result.get("followup_answer")
        or (result.get("report") or {}).get("hedge_or_strategy")
        or (result.get("brief") or {}).get("hedge_strategy")
    )
    turns = list(session.get("turns") or [])
    turns.append(
        {
            "ts": _now(),
            "query": result.get("query"),
            "ticker": ticker,
            "from_memory": bool(result.get("from_memory")),
            "related": (result.get("plan") or {}).get("related"),
            "intent": (result.get("plan") or {}).get("intent"),
            "need_tools": result.get("need_tools") or [],
            "reused": result.get("reused") or [],
            "revisions": result.get("revisions"),
        }
    )
    session["turns"] = turns[-MAX_TURNS:]
    save_session(session)
    return session


def classify_intent(query: str) -> str:
    text = (query or "").strip()
    lower = text.lower()
    if "analyse" in lower or "analyze" in lower:
        return "research"
    if _REFRESH.search(text):
        return "refresh"
    if _EXTEND.search(text):
        return "extend"
    if _RECALL.search(text):
        return "recall"
    if "?" in text or len(text) < 90:
        return "recall"
    return "research"


def _reuse_from_store(stored: dict[str, Any] | None) -> list[str]:
    if not stored:
        return []
    brief = stored.get("brief") or {}
    report = stored.get("report") or {}
    reused: list[str] = []
    if brief.get("current_price") is not None:
        reused.append("price")
    if brief.get("vol_30d_pct") is not None:
        reused.append("vol_30")
    if brief.get("vol_90d_pct") is not None:
        reused.append("vol_90")
    if brief.get("headlines"):
        reused.append("news")
    if brief.get("sentiment_label") or brief.get("sentiment_score") is not None:
        reused.append("sentiment")
    if brief.get("quantitative_risks") or report.get("top_risks"):
        reused.append("risks")
    if brief.get("hedge_strategy") or report.get("hedge_or_strategy"):
        reused.append("hedge")
    return reused


def _need_tools(intent: str, stored: dict[str, Any] | None, query: str) -> list[str]:
    brief = (stored or {}).get("brief") or {}
    if intent == "recall":
        return []
    if intent == "refresh":
        tools = ["get_price_data"]
        if re.search(r"\b(vol|volatility)\b", query, re.I):
            tools.append("calculate_volatility")
        return tools
    if intent == "extend":
        tools: list[str] = ["web_search"]
        if not brief.get("headlines"):
            tools.insert(0, "get_news")
        return tools
    if intent == "research" and stored and _fresh(stored):
        return []
    return []


def relate(query: str, session: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare the new question to the last one. Decide what to reuse vs fetch."""
    session = session or load_session()
    last_ticker = session.get("last_ticker")
    last_query = session.get("last_query")
    mentioned = ticker_mentioned(query)
    intent = classify_intent(query)
    if not last_ticker:
        return {
            "related": False,
            "ticker": mentioned,
            "previous_query": last_query,
            "intent": intent if mentioned else "research",
            "reuse": [],
            "need_tools": [],
            "reason": "No prior question in session.",
        }
    ticker = mentioned or last_ticker
    same = normalize_symbol(str(ticker)) == normalize_symbol(str(last_ticker))
    stored = recall(last_ticker, session) if same else None
    if not same:
        return {
            "related": False,
            "ticker": mentioned,
            "previous_query": last_query,
            "intent": "research",
            "reuse": [],
            "need_tools": [],
            "reason": f"New issuer {mentioned} is not the last one ({last_ticker}).",
        }
    reuse = _reuse_from_store(stored)
    need = _need_tools(intent, stored, query)
    reason = {
        "recall": "Same issuer; answer from the stored brief and last report.",
        "refresh": "Same issuer; reuse the brief and refresh the requested market facts.",
        "extend": "Same issuer; reuse the brief and fetch only the extra context.",
        "research": "Same issuer; reuse a fresh brief instead of starting over.",
    }.get(intent, "Same issuer.")
    return {
        "related": True,
        "ticker": last_ticker,
        "previous_query": last_query,
        "intent": intent,
        "reuse": reuse,
        "need_tools": need,
        "reason": reason,
    }


def is_followup(query: str, session: dict[str, Any] | None = None) -> bool:
    plan = relate(query, session)
    return bool(plan["related"] and plan["intent"] in {"recall", "extend", "refresh"})


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


def _run_needed_tools(ticker: str, need_tools: list[str], query: str, stored: dict[str, Any]) -> dict[str, Any]:
    """Call only the tools the relatedness plan asked for."""
    from src.tools import calculate_volatility, get_news, get_price_data, llm_sentiment, web_search

    facts: dict[str, Any] = {}
    brief = stored.get("brief") or {}
    if "get_price_data" in need_tools:
        raw = get_price_data.invoke({"ticker": ticker})
        facts["price"] = json.loads(raw) if isinstance(raw, str) else raw
    if "calculate_volatility" in need_tools:
        raw30 = calculate_volatility.invoke({"ticker": ticker, "window_days": 30})
        raw90 = calculate_volatility.invoke({"ticker": ticker, "window_days": 90})
        facts["vol_30"] = json.loads(raw30) if isinstance(raw30, str) else raw30
        facts["vol_90"] = json.loads(raw90) if isinstance(raw90, str) else raw90
    if "get_news" in need_tools:
        raw = get_news.invoke({"ticker": ticker})
        facts["news"] = json.loads(raw) if isinstance(raw, str) else raw
    if "web_search" in need_tools:
        search_q = query if len(query.split()) >= 4 else f"{ticker} {query} 90 day risks"
        raw = web_search.invoke({"query": search_q, "max_results": 5})
        facts["web_search"] = json.loads(raw) if isinstance(raw, str) else raw
    if "llm_sentiment" in need_tools:
        headlines = brief.get("headlines") or []
        news = facts.get("news") or {}
        if news.get("headlines"):
            headlines = [
                str(item.get("title"))
                for item in news["headlines"]
                if isinstance(item, dict) and item.get("title")
            ]
        raw = llm_sentiment.invoke({"ticker": ticker, "headlines": headlines})
        facts["sentiment"] = json.loads(raw) if isinstance(raw, str) else raw
    return facts


def _merge_facts(brief: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    updated = dict(brief)
    price = facts.get("price") or {}
    if price.get("close") is not None:
        updated["current_price"] = price["close"]
    if price.get("momentum_bias"):
        updated["momentum"] = price["momentum_bias"]
    if price.get("rsi_14") is not None:
        updated["rsi_14"] = price["rsi_14"]
    if (facts.get("vol_30") or {}).get("vol_pct") is not None:
        updated["vol_30d_pct"] = facts["vol_30"]["vol_pct"]
    if (facts.get("vol_90") or {}).get("vol_pct") is not None:
        updated["vol_90d_pct"] = facts["vol_90"]["vol_pct"]
    news = facts.get("news") or {}
    if news.get("headlines"):
        updated["headlines"] = [
            str(item.get("title"))
            for item in news["headlines"]
            if isinstance(item, dict) and item.get("title")
        ][:8]
    sentiment = facts.get("sentiment") or {}
    if sentiment.get("sentiment_score") is not None:
        updated["sentiment_score"] = sentiment["sentiment_score"]
        updated["sentiment_label"] = sentiment.get("label") or updated.get("sentiment_label")
    return updated


def _llm_answer(query: str, stored: dict[str, Any], facts: dict[str, Any], plan: dict[str, Any]) -> str:
    fallback = _answer_from_store(query, stored)
    if facts:
        fallback += "\n\nNew tool facts:\n" + json.dumps(facts, default=str)[:2000]
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.agent_a import FallbackChat, _extract_json_object
        from src.config import load_settings

        settings = load_settings()
        if not settings.llm_ready:
            return fallback
        llm = FallbackChat(settings, tools=False)
        reply = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are answering a follow-up. Previous question and its DataBrief/"
                        "FinalReport are memory. Reuse those numbers. Use new tool facts only "
                        "to fill gaps. Do not invent prices. State what you reused and which "
                        "tools you ran."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Previous question: {plan.get('previous_query')}\n"
                        f"New question: {query}\n"
                        f"Plan: related={plan.get('related')} intent={plan.get('intent')} "
                        f"reuse={plan.get('reuse')} need_tools={plan.get('need_tools')}\n"
                        f"Brief: {json.dumps(stored.get('brief'), default=str)[:8000]}\n"
                        f"Report: {json.dumps(stored.get('report'), default=str)[:8000]}\n"
                        f"New facts: {json.dumps(facts, default=str)[:4000]}"
                    )
                ),
            ]
        )
        content = getattr(reply, "content", "")
        if isinstance(content, str) and content.strip():
            extracted = _extract_json_object(content)
            return extracted.get("answer") if extracted and extracted.get("answer") else content.strip()
    except Exception:
        pass
    return fallback


def answer_followup(query: str, session: dict[str, Any] | None = None) -> dict[str, Any]:
    session = session or load_session()
    plan = relate(query, session)
    stored = recall(plan.get("ticker") or session.get("last_ticker"), session)
    if not stored or not stored.get("brief"):
        raise ValueError("No stored brief yet. Run a full research prompt first.")
    need = list(plan.get("need_tools") or [])
    facts = _run_needed_tools(str(plan.get("ticker") or session.get("last_ticker")), need, query, stored) if need else {}
    brief = _merge_facts(stored.get("brief") or {}, facts)
    stored = {**stored, "brief": brief}
    text = _llm_answer(query, stored, facts, plan)
    result = {
        "query": query,
        "ticker": plan.get("ticker") or session.get("last_ticker"),
        "from_memory": True,
        "plan": plan,
        "reused": plan.get("reuse") or ["brief", "report"],
        "need_tools": need,
        "extra_facts": facts,
        "brief": brief,
        "report": stored.get("report"),
        "followup_answer": text,
        "tool_calls": [{"tool": name, "cached": (facts.get(name) or {}).get("cached")} for name in need],
        "revisions": 0,
        "critiques": [],
    }
    remember(result, session)
    return result


def ask_agent_a(query: str) -> dict[str, Any]:
    """Task 2 entry: Agent A only, with session memory and disk tool cache.

    Follow-ups on the same issuer reuse the stored brief and call only gap tools.
    New questions run Agent A, which picks tools based on the question.
    """
    session = load_session()
    plan = relate(query, session)
    stored = recall(plan.get("ticker") or session.get("last_ticker"), session)

    if plan["related"] and plan["intent"] in {"recall", "extend", "refresh"} and stored and stored.get("brief"):
        result = answer_followup(query, session)
        result["memory"] = describe_memory()
        return result

    from src.agent_a import run_agent_a

    result = run_agent_a(query)
    result["from_memory"] = False
    result["plan"] = plan
    result["reused"] = []
    result["need_tools"] = [step["tool"] for step in (result.get("tool_calls") or []) if step.get("tool")]
    result["memory"] = describe_memory()
    remember(result, session)
    return result


def ask(query: str, agent_a_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Memory-aware entry: relate to the last question, reuse facts, fetch only gaps."""
    session = load_session()
    plan = relate(query, session)
    stored = recall(plan.get("ticker") or session.get("last_ticker"), session)

    if plan["related"] and plan["intent"] in {"recall", "extend", "refresh"} and stored and stored.get("brief"):
        result = answer_followup(query, session)
        result["memory"] = describe_memory()
        return result

    from src.agent_b import run_two_agents
    from src.ticker import parse_research_query

    parsed = parse_research_query(query)
    reused = agent_a_result
    if reused is None and plan["related"] and stored and stored.get("brief") and _fresh(stored):
        reused = {
            "brief": stored["brief"],
            "ticker": parsed["ticker"],
            "tool_calls": stored.get("tool_calls") or [],
            "missing_after_run": [],
        }
    result = run_two_agents(query, agent_a_result=reused)
    result["from_memory"] = bool(reused and not agent_a_result)
    result["plan"] = plan
    result["reused"] = plan.get("reuse") if reused else []
    result["need_tools"] = [] if reused else ["full_research"]
    result["memory"] = describe_memory()
    remember(result)
    return result


def cache_dir() -> Path:
    _ensure_dirs()
    return CACHE_DIR
