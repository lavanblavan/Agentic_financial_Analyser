"""Agent B: critic. Reviews a typed DataBrief, may request one fact, then publishes.

Handoff is Pydantic (DataBrief → CritiqueDecision → DataBrief → FinalReport).
Do not pass raw strings across the A→B boundary.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent_a import (
    FallbackChat,
    _brief_from_facts,
    _extract_json_object,
    run_agent_a,
)
from src.config import load_settings, missing_key_help, project_root
from src.groq_client import LLMNotConfiguredError
from src.schemas import CritiqueDecision, DataBrief, FinalReport, RiskFactor
from src.task_profile import infer_task_profile
from src.ticker import parse_research_query
from src.tools import calculate_volatility, get_news, llm_sentiment, web_search

MAX_REVISIONS = 2

GENERIC_RISK_STEMS = (
    "supply chain",
    "market volatility",
    "consumer demand",
    "economic downturn",
    "geopolitical",
    "interest rate",
    "recession",
    "inflation risk",
)


class TwoAgentState(TypedDict, total=False):
    query: str
    ticker: str
    company_name: str
    task_mode: str
    brief: dict[str, Any]
    critique: dict[str, Any]
    critiques: list[dict[str, Any]]
    extra_facts: dict[str, Any]
    revisions: int
    report: dict[str, Any]
    agent_a: dict[str, Any]


def _system_prompt() -> str:
    return (project_root() / "prompts" / "agent_b.md").read_text(encoding="utf-8")


def _as_brief(payload: dict[str, Any] | DataBrief) -> DataBrief:
    if isinstance(payload, DataBrief):
        return payload
    return DataBrief.model_validate(payload)


def brief_gaps(brief: DataBrief, mode: str = "full_research") -> list[str]:
    """Deterministic holes a critic must not ignore. Not a tool-call order."""
    if mode == "news":
        return [] if brief.headlines else ["news"]
    if mode == "price":
        return [] if brief.current_price else ["price"]
    if mode == "volatility":
        gaps: list[str] = []
        if brief.vol_30d_pct in (None, 0.0) and brief.vol_90d_pct is None:
            gaps.append("vol")
        return gaps
    if mode == "sentiment":
        gaps = []
        if not brief.headlines:
            gaps.append("news")
        if brief.sentiment_label == "neutral" and brief.sentiment_score == 0.0 and not brief.headlines:
            gaps.append("sentiment")
        return gaps
    if mode == "adaptive":
        return []

    gaps: list[str] = []
    if brief.vol_90d_pct is None:
        gaps.append("vol_90")
    if not brief.headlines:
        gaps.append("news")
    if len(brief.quantitative_risks) < 3 or _generic_risks(brief):
        gaps.append("generic_risks")
    if not (brief.hedge_strategy or "").strip():
        gaps.append("hedge")
    elif not _hedge_uses_90d(brief):
        gaps.append("hedge_90d")
    return gaps


def _generic_risks(brief: DataBrief) -> bool:
    if not brief.quantitative_risks:
        return True
    headline_blob = " ".join(brief.headlines).lower()
    generic = 0
    for risk in brief.quantitative_risks:
        name = risk.name.lower()
        evidence = risk.evidence.lower()
        stem_hit = any(stem in name for stem in GENERIC_RISK_STEMS)
        has_number = any(ch.isdigit() for ch in evidence)
        cites_news = bool(headline_blob) and any(
            token in evidence for token in headline_blob.split() if len(token) > 6
        )
        vague = evidence.startswith("potential ") or "could impact" in evidence or "may lead" in evidence
        if stem_hit and not (has_number and cites_news):
            generic += 1
        elif vague and not has_number:
            generic += 1
    return generic >= 2


def _hedge_uses_90d(brief: DataBrief) -> bool:
    text = (brief.hedge_strategy or "").lower()
    if brief.vol_90d_pct is None:
        return False
    vol = str(brief.vol_90d_pct)
    return "90" in text or vol in text


def decision_from_gaps(brief: DataBrief, gaps: list[str]) -> CritiqueDecision:
    ticker = brief.ticker
    if "vol_90" in gaps:
        return CritiqueDecision(
            need_more_data=True,
            request_kind="vol_90",
            request=f"calculate_volatility for {ticker} with window_days=90",
            reason="90-day realized vol is missing; the risk horizon is 90 days.",
            issues=gaps,
        )
    if "news" in gaps:
        return CritiqueDecision(
            need_more_data=True,
            request_kind="news",
            request=f"get_news for {ticker}",
            reason="No headlines; sentiment and 90-day risks are ungrounded.",
            issues=gaps,
        )
    if "generic_risks" in gaps:
        return CritiqueDecision(
            need_more_data=True,
            request_kind="web_search",
            request=f"{brief.company_name or ticker} {ticker} share-price risks catalysts next 90 days",
            reason="Risks are generic or not tied to headlines. Need outside corroboration.",
            issues=gaps,
        )
    if "hedge" in gaps or "hedge_90d" in gaps:
        return CritiqueDecision(
            need_more_data=True,
            request_kind="revise",
            request="Rewrite the hedge using 30d vs 90d realized vol. Keep the same numbers.",
            reason="Hedge does not use the 90-day vol term structure.",
            issues=gaps,
        )
    return CritiqueDecision(need_more_data=False, reason="No structural gaps.", issues=[])


def _llm_review(brief: DataBrief, gaps: list[str]) -> dict[str, Any] | None:
    settings = load_settings()
    if not settings.llm_ready:
        return None
    try:
        llm = FallbackChat(settings, tools=False)
        reply = llm.invoke(
            [
                SystemMessage(content=_system_prompt()),
                HumanMessage(
                    content=(
                        "Review this DataBrief. Known structural gaps: "
                        + ", ".join(gaps or ["none"])
                        + "\nReturn CritiqueDecision JSON only.\n"
                        + brief.model_dump_json()
                    )
                ),
            ]
        )
        content = getattr(reply, "content", "")
        payload = _extract_json_object(content if isinstance(content, str) else str(content))
        return payload
    except Exception:
        return None


def critique_brief(brief: DataBrief, revisions: int = 0, mode: str = "full_research") -> CritiqueDecision:
    gaps = brief_gaps(brief, mode=mode)
    if revisions >= MAX_REVISIONS or not gaps:
        notes = _llm_review(brief, gaps) or {}
        return CritiqueDecision(
            need_more_data=False,
            reason=str(notes.get("reason") or "Brief is complete enough to publish."),
            issues=list(notes.get("issues") or gaps),
        )
    decision = decision_from_gaps(brief, gaps)
    notes = _llm_review(brief, gaps)
    if notes:
        if notes.get("reason"):
            decision.reason = str(notes["reason"])
        extra = notes.get("issues")
        if isinstance(extra, list):
            merged = list(dict.fromkeys([*decision.issues, *[str(item) for item in extra]]))
            decision.issues = merged
    return decision


def fulfill_request(brief: DataBrief, critique: CritiqueDecision) -> dict[str, Any]:
    """Run the one tool Agent B asked for. Kind comes from the critic, not a pipeline."""
    kind = critique.request_kind
    ticker = brief.ticker
    facts: dict[str, Any] = {}
    if kind == "vol_90":
        raw = calculate_volatility.invoke({"ticker": ticker, "window_days": 90})
        facts["vol_90"] = json.loads(raw) if isinstance(raw, str) else raw
    elif kind == "news":
        raw = get_news.invoke({"ticker": ticker})
        facts["news"] = json.loads(raw) if isinstance(raw, str) else raw
    elif kind == "web_search":
        query = critique.request or f"{brief.company_name or ticker} {ticker} 90 day stock risks"
        raw = web_search.invoke({"query": query, "max_results": 5})
        facts["web_search"] = json.loads(raw) if isinstance(raw, str) else raw
    elif kind == "sentiment":
        raw = llm_sentiment.invoke({"ticker": ticker, "headlines": brief.headlines})
        facts["sentiment"] = json.loads(raw) if isinstance(raw, str) else raw
    return facts


def _apply_facts(brief: DataBrief, extra: dict[str, Any]) -> DataBrief:
    payload = brief.model_dump()
    vol90 = extra.get("vol_90") or {}
    if vol90.get("vol_pct") is not None:
        payload["vol_90d_pct"] = float(vol90["vol_pct"])
    news = extra.get("news") or {}
    if news.get("headlines"):
        titles = [
            str(item.get("title"))
            for item in news["headlines"]
            if isinstance(item, dict) and item.get("title")
        ]
        if titles:
            payload["headlines"] = titles[:8]
    sentiment = extra.get("sentiment") or {}
    if sentiment.get("sentiment_score") is not None:
        payload["sentiment_score"] = float(sentiment["sentiment_score"])
        payload["sentiment_label"] = str(sentiment.get("label") or payload.get("sentiment_label"))
    return DataBrief.model_validate(payload)


def revise_brief(brief: DataBrief, critique: CritiqueDecision, extra: dict[str, Any]) -> DataBrief:
    updated = _apply_facts(brief, extra)
    settings = load_settings()
    if not settings.llm_ready:
        return updated
    try:
        llm = FallbackChat(settings, tools=False)
        reply = llm.invoke(
            [
                SystemMessage(content=_system_prompt()),
                HumanMessage(
                    content=(
                        "Revise the DataBrief JSON only. Keep every price and vol number. "
                        "Address this critic request: "
                        f"{critique.request_kind}: {critique.request}\n"
                        f"Reason: {critique.reason}\n"
                        "New tool facts:\n"
                        + json.dumps(extra, default=str)[:8000]
                        + "\nCurrent brief:\n"
                        + updated.model_dump_json()
                    )
                ),
            ]
        )
        payload = _extract_json_object(str(reply.content))
        if payload:
            facts = {
                "price": {"close": updated.current_price, "momentum_bias": updated.momentum, "rsi_14": updated.rsi_14},
                "vol_30": {"vol_pct": updated.vol_30d_pct},
                "vol_90": {"vol_pct": updated.vol_90d_pct},
                "news": {"headlines": [{"title": h} for h in updated.headlines]},
                "sentiment": {"sentiment_score": updated.sentiment_score, "label": updated.sentiment_label},
            }
            return _brief_from_facts(updated.ticker, updated.company_name, facts, payload)
    except Exception:
        pass
    return updated


def report_from_brief(
    brief: DataBrief,
    critiques: list[dict[str, Any]],
    revisions: int,
) -> FinalReport:
    notes = []
    for item in critiques:
        kind = item.get("request_kind") or "approve"
        notes.append(f"{kind}: {item.get('reason')}")
    return FinalReport(
        ticker=brief.ticker,
        financial_health=brief.financial_health or "Technicals only; no fundamental statements.",
        market_sentiment=f"{brief.sentiment_label} ({brief.sentiment_score})",
        top_risks=brief.quantitative_risks[:3],
        hedge_or_strategy=brief.hedge_strategy,
        used_90d_vol=brief.vol_90d_pct is not None,
        critique_rounds=revisions,
        critic_notes=" | ".join(notes),
        brief=brief,
    )


def _llm_report(brief: DataBrief, critiques: list[dict[str, Any]], revisions: int) -> FinalReport:
    fallback = report_from_brief(brief, critiques, revisions)
    settings = load_settings()
    if not settings.llm_ready:
        return fallback
    try:
        llm = FallbackChat(settings, tools=False)
        reply = llm.invoke(
            [
                SystemMessage(content=_system_prompt()),
                HumanMessage(
                    content=(
                        "Publish FinalReport JSON only. Keep Agent A numbers. "
                        "Critique trail:\n"
                        + json.dumps(critiques, default=str)[:4000]
                        + "\nBrief:\n"
                        + brief.model_dump_json()
                    )
                ),
            ]
        )
        payload = _extract_json_object(str(reply.content))
        if not payload:
            return fallback
        payload["brief"] = brief.model_dump()
        payload["used_90d_vol"] = brief.vol_90d_pct is not None
        payload["critique_rounds"] = revisions
        payload.setdefault("critic_notes", fallback.critic_notes)
        payload.setdefault("ticker", brief.ticker)
        if payload.get("top_risks"):
            risks: list[RiskFactor] = []
            for risk in payload["top_risks"][:3]:
                try:
                    risks.append(RiskFactor.model_validate(risk))
                except Exception:
                    continue
            if len(risks) == 3:
                payload["top_risks"] = [r.model_dump() for r in risks]
            else:
                payload["top_risks"] = [r.model_dump() for r in brief.quantitative_risks[:3]]
        else:
            payload["top_risks"] = [r.model_dump() for r in brief.quantitative_risks[:3]]
        return FinalReport.model_validate(payload)
    except Exception:
        return fallback


def build_two_agent_graph():
    settings = load_settings()
    if not settings.llm_ready:
        raise LLMNotConfiguredError(missing_key_help(settings))

    def researcher(state: TwoAgentState) -> dict[str, Any]:
        if state.get("brief"):
            return {}
        result = run_agent_a(state["query"])
        return {
            "ticker": result["ticker"],
            "company_name": result["parsed"]["name"],
            "task_mode": result.get("task_mode") or "full_research",
            "brief": result["brief"],
            "agent_a": {
                "tool_calls": result.get("tool_calls"),
                "missing_after_run": result.get("missing_after_run"),
                "task_mode": result.get("task_mode"),
            },
        }

    def critic(state: TwoAgentState) -> dict[str, Any]:
        brief = _as_brief(state["brief"])
        mode = str(state.get("task_mode") or "full_research")
        decision = critique_brief(brief, revisions=int(state.get("revisions") or 0), mode=mode)
        trail = list(state.get("critiques") or [])
        trail.append(decision.model_dump())
        return {"critique": decision.model_dump(), "critiques": trail}

    def fulfill(state: TwoAgentState) -> dict[str, Any]:
        brief = _as_brief(state["brief"])
        critique = CritiqueDecision.model_validate(state["critique"])
        extra = dict(state.get("extra_facts") or {})
        extra.update(fulfill_request(brief, critique))
        return {"extra_facts": extra}

    def revise(state: TwoAgentState) -> dict[str, Any]:
        brief = _as_brief(state["brief"])
        critique = CritiqueDecision.model_validate(state["critique"])
        extra = state.get("extra_facts") or {}
        updated = revise_brief(brief, critique, extra)
        return {"brief": updated.model_dump(), "revisions": int(state.get("revisions") or 0) + 1}

    def publish(state: TwoAgentState) -> dict[str, Any]:
        brief = _as_brief(state["brief"])
        critiques = list(state.get("critiques") or [])
        report = _llm_report(brief, critiques, int(state.get("revisions") or 0))
        return {"report": report.model_dump()}

    def after_critic(state: TwoAgentState) -> str:
        critique = state.get("critique") or {}
        if critique.get("need_more_data") and int(state.get("revisions") or 0) < MAX_REVISIONS:
            return "fulfill"
        return "publish"

    graph = StateGraph(TwoAgentState)
    graph.add_node("researcher", researcher)
    graph.add_node("critic", critic)
    graph.add_node("fulfill", fulfill)
    graph.add_node("revise", revise)
    graph.add_node("publish", publish)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges("critic", after_critic, {"fulfill": "fulfill", "publish": "publish"})
    graph.add_edge("fulfill", "revise")
    graph.add_edge("revise", "critic")
    graph.add_edge("publish", END)
    return graph.compile()


def run_two_agents(
    query: str,
    agent_a_result: dict[str, Any] | None = None,
    recursion_limit: int = 16,
) -> dict[str, Any]:
    """Agent A research → Agent B critique loop → FinalReport."""
    parsed = parse_research_query(query)
    profile = infer_task_profile(query)
    graph = build_two_agent_graph()
    seed: TwoAgentState = {
        "query": query,
        "ticker": parsed["ticker"],
        "company_name": parsed["name"],
        "task_mode": str(profile["mode"]),
        "revisions": 0,
        "critiques": [],
        "extra_facts": {},
    }
    if agent_a_result and agent_a_result.get("brief"):
        seed["brief"] = agent_a_result["brief"]
        seed["ticker"] = agent_a_result.get("ticker") or parsed["ticker"]
        seed["task_mode"] = str(agent_a_result.get("task_mode") or profile["mode"])
        seed["agent_a"] = {
            "tool_calls": agent_a_result.get("tool_calls"),
            "missing_after_run": agent_a_result.get("missing_after_run"),
            "task_mode": agent_a_result.get("task_mode"),
        }
    result = graph.invoke(seed, config={"recursion_limit": recursion_limit})
    report = result.get("report")
    payload = {
        "query": query,
        "parsed": parsed,
        "ticker": result.get("ticker") or parsed["ticker"],
        "brief": result.get("brief"),
        "critique": result.get("critique"),
        "critiques": result.get("critiques") or [],
        "extra_facts": result.get("extra_facts") or {},
        "revisions": result.get("revisions") or 0,
        "report": report,
        "agent_a": result.get("agent_a") or {},
    }
    try:
        from src.memory import remember

        remember(payload)
    except Exception:
        pass
    return payload


def format_final_report(result: dict[str, Any]) -> str:
    report = result.get("report") or {}
    ticker = report.get("ticker") or result.get("ticker") or "?"
    brief = report.get("brief") or result.get("brief") or {}
    lines = [
        f"{ticker} — final report (Agent A + Agent B)",
        f"Price {brief.get('current_price')}  |  30d vol {brief.get('vol_30d_pct')}%  |  "
        f"90d vol {brief.get('vol_90d_pct')}%  |  used_90d_vol={report.get('used_90d_vol')}",
        f"Critique rounds: {report.get('critique_rounds')}",
        "",
        "Financial health:",
        report.get("financial_health") or "(missing)",
        "",
        f"Market sentiment: {report.get('market_sentiment')}",
        "",
        "Top three 90-day risks:",
    ]
    risks = report.get("top_risks") or []
    if not risks:
        lines.append("  (none)")
    for i, risk in enumerate(risks[:3], start=1):
        if isinstance(risk, dict):
            lines.append(
                f"  {i}. [{risk.get('severity')}] {risk.get('name')} — {risk.get('evidence')}"
            )
        else:
            lines.append(f"  {i}. {risk}")
    lines.extend(["", "Data-driven hedge:", report.get("hedge_or_strategy") or "(missing)"])
    trail = result.get("critiques") or []
    if trail:
        lines.extend(["", "Critique trail:"])
        for i, item in enumerate(trail, start=1):
            kind = item.get("request_kind") or ("publish" if not item.get("need_more_data") else "request")
            lines.append(f"  {i}. {kind}: {item.get('reason')}")
    return "\n".join(lines)
