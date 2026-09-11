"""Agent A: quantitative researcher with a LangGraph ReAct tool loop.

The LLM reads the user question and chooses which tools to call. The graph
only nudges for observations required by that question's task profile.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import (
    AGENT_MAX_TOKENS,
    AGENT_MODEL_FALLBACKS,
    OPENROUTER_MODEL_FALLBACKS,
    load_settings,
    missing_key_help,
    project_root,
)
from src.groq_client import LLMNotConfiguredError
from src.schemas import DataBrief, RiskFactor
from src.task_profile import FULL_RESEARCH, infer_task_profile
from src.ticker import parse_research_query
from src.tools import ALL_TOOLS

MAX_NUDGES = 2


class AgentAState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    ticker: str
    company_name: str
    task_mode: str
    required_observations: list[str]
    nudges: int
    brief: dict[str, Any]


def _system_prompt() -> str:
    return (project_root() / "prompts" / "agent_a.md").read_text(encoding="utf-8")


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def tool_call_sequence(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        for call in message.tool_calls:
            steps.append({"tool": call.get("name"), "args": call.get("args") or {}})
    return steps


def missing_observations(
    messages: list[BaseMessage],
    required: list[str] | None = None,
) -> list[str]:
    """Check which required observations are still missing for this question."""
    required = list(required if required is not None else FULL_RESEARCH)
    if not required:
        return []

    calls = tool_call_sequence(messages)
    names = [step["tool"] for step in calls]
    windows = [
        int((step.get("args") or {}).get("window_days") or 30)
        for step in calls
        if step["tool"] == "calculate_volatility"
    ]
    missing: list[str] = []
    if "get_price_data" in required and "get_price_data" not in names:
        missing.append("get_price_data")
    if "calculate_volatility:30" in required and not any(window <= 30 for window in windows):
        missing.append("calculate_volatility with window_days=30")
    if "calculate_volatility:90" in required and not any(window >= 90 for window in windows):
        missing.append("calculate_volatility with window_days=90")
    if "get_news" in required and "get_news" not in names:
        missing.append("get_news")
    if "llm_sentiment" in required and "llm_sentiment" not in names:
        missing.append("llm_sentiment")
    if "web_search" in required and "web_search" not in names:
        missing.append("web_search")
    return missing


def route_after_agent(state: AgentAState) -> str:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    required = state.get("required_observations") or []
    if (
        required
        and missing_observations(messages, required)
        and int(state.get("nudges") or 0) < MAX_NUDGES
    ):
        return "nudge"
    return "finalize"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = raw.rstrip("`").strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_brief(text: str) -> DataBrief | None:
    payload = _extract_json_object(text)
    if not payload:
        return None
    try:
        return DataBrief.model_validate(payload)
    except Exception:
        return None


def _tool_facts(messages: list[BaseMessage]) -> dict[str, Any]:
    facts: dict[str, Any] = {"price": {}, "vol_30": {}, "vol_90": {}, "news": {}, "sentiment": {}}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content) if isinstance(message.content, str) else message.content
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        name = message.name or ""
        if name == "get_price_data":
            facts["price"] = data
        elif name == "calculate_volatility":
            window = int(data.get("window_days") or 30)
            if window >= 90:
                facts["vol_90"] = data
            else:
                facts["vol_30"] = data
        elif name == "get_news":
            facts["news"] = data
        elif name == "llm_sentiment":
            facts["sentiment"] = data
    return facts


def _brief_from_facts(
    ticker: str,
    company_name: str,
    facts: dict[str, Any],
    llm_payload: dict[str, Any] | None,
) -> DataBrief:
    payload = dict(llm_payload or {})
    price = facts.get("price") or {}
    vol30 = facts.get("vol_30") or {}
    vol90 = facts.get("vol_90") or {}
    news = facts.get("news") or {}
    sentiment = facts.get("sentiment") or {}
    headlines = payload.get("headlines") or [
        str(item.get("title"))
        for item in (news.get("headlines") or [])
        if isinstance(item, dict) and item.get("title")
    ][:6]
    risks = payload.get("quantitative_risks") or []
    parsed_risks: list[RiskFactor] = []
    for risk in risks[:3]:
        try:
            parsed_risks.append(RiskFactor.model_validate(risk))
        except Exception:
            continue
    score = payload.get("sentiment_score", sentiment.get("sentiment_score", 0.0))
    try:
        score = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    close = payload.get("current_price", price.get("close"))
    vol_30 = payload.get("vol_30d_pct", vol30.get("vol_pct"))
    vol_90 = payload.get("vol_90d_pct", vol90.get("vol_pct"))
    return DataBrief(
        ticker=ticker,
        company_name=str(payload.get("company_name") or company_name or ticker),
        current_price=float(close or 0.0),
        vol_30d_pct=float(vol_30 or 0.0),
        vol_90d_pct=None if vol_90 in (None, "") else float(vol_90),
        momentum=str(payload.get("momentum") or price.get("momentum_bias") or "mixed"),
        rsi_14=payload.get("rsi_14") if payload.get("rsi_14") is not None else price.get("rsi_14"),
        financial_health=str(payload.get("financial_health") or ""),
        sentiment_score=score,
        sentiment_label=str(payload.get("sentiment_label") or sentiment.get("label") or "neutral"),
        headlines=[str(h) for h in headlines],
        quantitative_risks=parsed_risks,
        hedge_strategy=str(payload.get("hedge_strategy") or ""),
        notes=str(payload.get("notes") or ""),
    )


def _is_missing_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist" in text or "error code: 404" in text


def _is_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "rate_limit" in text
        or "error code: 429" in text
        or "tokens per minute" in text
        or "otpm" in text
        or "request too large" in text
    )


def _is_router_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        _is_missing_model_error(exc)
        or _is_capacity_error(exc)
        or "402" in text
        or "no endpoints" in text
        or "insufficient" in text
        or "payment required" in text
    )


class FallbackChat:
    """OpenRouter first when a key is set; Groq is the fallback."""

    def __init__(self, settings, tools: bool = True):
        preferred_groq = settings.groq_agent_model
        self.groq_models = list(dict.fromkeys([preferred_groq, *AGENT_MODEL_FALLBACKS]))
        preferred_router = settings.openrouter_model
        self.router_models = list(dict.fromkeys([preferred_router, *OPENROUTER_MODEL_FALLBACKS]))
        self.settings = settings
        self.tools = tools
        self.active = preferred_router if settings.prefer_openrouter else preferred_groq
        self.provider = "openrouter" if settings.prefer_openrouter else "groq"

    def _groq(self, model: str):
        llm = ChatGroq(
            model=model,
            api_key=self.settings.groq_api_key,
            temperature=0.1,
            max_tokens=min(self.settings.max_tokens, AGENT_MAX_TOKENS),
        )
        return llm.bind_tools(ALL_TOOLS) if self.tools else llm

    def _openrouter(self, model: str):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=self.settings.max_tokens,
            default_headers={
                "HTTP-Referer": "https://github.com/lavanblavan/Agentic_financial_Analyser",
                "X-Title": "Agentic Financial Analyser",
            },
        )
        return llm.bind_tools(ALL_TOOLS) if self.tools else llm

    def invoke(self, messages):
        last_error: Exception | None = None
        if self.settings.openrouter_api_key:
            start = self.router_models.index(self.active) if self.active in self.router_models else 0
            for model in self.router_models[start:] + self.router_models[:start]:
                try:
                    result = self._openrouter(model).invoke(messages)
                    self.active = model
                    self.provider = "openrouter"
                    return result
                except Exception as exc:
                    if _is_router_retryable(exc):
                        last_error = exc
                        continue
                    raise
        if self.settings.groq_api_key:
            for model in self.groq_models:
                try:
                    result = self._groq(model).invoke(messages)
                    self.active = model
                    self.provider = "groq"
                    return result
                except Exception as exc:
                    if _is_missing_model_error(exc) or _is_capacity_error(exc):
                        last_error = exc
                        continue
                    raise
        raise last_error or RuntimeError("No chat model available (OpenRouter/Groq)")


def build_agent_a():
    settings = load_settings()
    if not settings.llm_ready:
        raise LLMNotConfiguredError(missing_key_help(settings))

    llm = FallbackChat(settings, tools=True)

    def agent_node(state: AgentAState) -> dict[str, Any]:
        return {"messages": [llm.invoke(state["messages"])]}

    def nudge_node(state: AgentAState) -> dict[str, Any]:
        required = state.get("required_observations") or []
        missing = missing_observations(state.get("messages") or [], required)
        return {
            "nudges": int(state.get("nudges") or 0) + 1,
            "messages": [
                HumanMessage(
                    content=(
                        "Do not finish yet. This question still needs observations from: "
                        + ", ".join(missing)
                        + ". Call those tools now. You choose the order, but do not skip required ones."
                    )
                )
            ],
        }

    def finalize_node(state: AgentAState) -> dict[str, Any]:
        messages = state.get("messages") or []
        facts = _tool_facts(messages)
        last_text = _message_text(messages[-1]) if messages else ""
        payload = _extract_json_object(last_text)
        if payload is None:
            try:
                synthesizer = FallbackChat(settings, tools=False)
                reply = synthesizer.invoke(
                    [
                        SystemMessage(content=_system_prompt()),
                        HumanMessage(
                            content=(
                                "Write the DataBrief JSON only. Use these tool facts; "
                                "do not invent prices.\n"
                                + json.dumps(facts, default=str)[:12000]
                            )
                        ),
                    ]
                )
                payload = _extract_json_object(_message_text(reply))
            except Exception:
                payload = None
        brief = _brief_from_facts(
            ticker=state.get("ticker") or "UNKNOWN",
            company_name=state.get("company_name") or "",
            facts=facts,
            llm_payload=payload,
        )
        return {
            "brief": brief.model_dump(),
            "messages": [AIMessage(content=brief.model_dump_json())],
        }

    graph = StateGraph(AgentAState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("nudge", nudge_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "nudge": "nudge", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent_a(
    query: str = (
        "Analyse the current financial health and market sentiment of nvidia. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    ),
    question: str | None = None,
    recursion_limit: int = 20,
) -> dict[str, Any]:
    """Run Agent A on the user's question; it decides which tools to call."""
    parsed = parse_research_query(query)
    profile = infer_task_profile(query)
    ticker = parsed["ticker"]
    user_question = parsed["task"]
    required = list(profile.get("required") or [])
    question = question or (
        f"User question: {user_question}\n\n"
        f"Resolved issuer: {parsed['name']} ({ticker}).\n"
        f"Task mode: {profile['mode']}.\n"
        f"{profile['tool_guidance']}\n"
        "Pick only the tools needed to answer the user question. You choose call order."
    )
    graph = build_agent_a()
    result = graph.invoke(
        {
            "ticker": ticker,
            "company_name": parsed["name"],
            "task_mode": str(profile["mode"]),
            "required_observations": required,
            "nudges": 0,
            "messages": [
                SystemMessage(content=_system_prompt()),
                HumanMessage(content=question),
            ],
        },
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    last = messages[-1]
    final_text = _message_text(last)
    brief = result.get("brief") or (_parse_brief(final_text).model_dump() if _parse_brief(final_text) else None)
    payload = {
        "query": query,
        "parsed": parsed,
        "resolved": parsed,
        "task_mode": profile["mode"],
        "required_observations": required,
        "ticker": ticker,
        "question": question,
        "tool_calls": tool_call_sequence(messages),
        "missing_after_run": missing_observations(messages, required),
        "final_text": final_text,
        "brief": brief,
        "messages": messages,
    }
    try:
        from src.memory import remember

        remember(payload)
    except Exception:
        pass
    return payload


def format_answer(result: dict[str, Any]) -> str:
    """Readable notebook output: health, sentiment, 3 risks, one hedge."""
    brief = result.get("brief") or {}
    ticker = result.get("ticker") or brief.get("ticker") or "?"
    mode = result.get("task_mode") or "research"
    lines = [
        f"{ticker} — {mode} brief",
        f"Price {brief.get('current_price')}  |  30d vol {brief.get('vol_30d_pct')}%  |  "
        f"90d vol {brief.get('vol_90d_pct')}%  |  momentum {brief.get('momentum')}",
        "",
        "Financial health:",
        brief.get("financial_health") or "(missing)",
        "",
        f"Market sentiment: {brief.get('sentiment_label')} "
        f"({brief.get('sentiment_score')})",
        "",
        "Top three 90-day risks:",
    ]
    risks = brief.get("quantitative_risks") or []
    if not risks:
        lines.append("  (none parsed)")
    for i, risk in enumerate(risks[:3], start=1):
        if isinstance(risk, dict):
            lines.append(
                f"  {i}. [{risk.get('severity')}] {risk.get('name')} — {risk.get('evidence')}"
            )
        else:
            lines.append(f"  {i}. {risk}")
    lines.extend(["", "Data-driven hedge:", brief.get("hedge_strategy") or "(missing)"])
    missing = result.get("missing_after_run") or []
    if missing:
        lines.extend(["", "Still missing tool observations:", ", ".join(missing)])
    return "\n".join(lines)
