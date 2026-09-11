"""Agent A: quantitative researcher with a LangGraph ReAct tool loop.

The LLM chooses tools. should_continue routes to tools if the model emitted
tool calls, otherwise END. No hard-coded tool sequence.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import load_settings, missing_key_help, project_root
from src.groq_client import LLMNotConfiguredError
from src.schemas import DataBrief
from src.ticker import parse_research_query
from src.tools import ALL_TOOLS


class AgentAState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ticker: str


def should_continue(state: AgentAState) -> str:
    """After each agent step: run tools if the LLM asked for them, else stop."""
    return tools_condition(state)


def _system_prompt() -> str:
    return (project_root() / "prompts" / "agent_a.md").read_text(encoding="utf-8")


def build_agent_a():
    settings = load_settings()
    if not settings.llm_ready:
        raise LLMNotConfiguredError(missing_key_help(settings))

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    ).bind_tools(ALL_TOOLS)

    def agent_node(state: AgentAState) -> dict[str, Any]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(AgentAState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


def tool_call_sequence(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Visible evidence that the LLM chose tools at runtime."""
    steps: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        for call in message.tool_calls:
            steps.append(
                {
                    "tool": call.get("name"),
                    "args": call.get("args") or {},
                }
            )
    return steps


def _parse_brief(text: str) -> DataBrief | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        payload = json.loads(raw)
        return DataBrief.model_validate(payload)
    except Exception:
        return None


def run_agent_a(
    query: str = (
        "Analyse the current financial health and market sentiment of nvidia. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    ),
    question: str | None = None,
    recursion_limit: int = 12,
) -> dict[str, Any]:
    """Run Agent A on the assessment prompt, or on a bare name/ticker."""
    parsed = parse_research_query(query)
    ticker = parsed["ticker"]
    question = question or (
        f"{parsed['task']}\n\n"
        f"Resolved issuer: {parsed['name']} ({ticker}). "
        "Use this ticker in every tool call. "
        "Horizon for the three risks and the hedge is the next 90 days. "
        "Ground the hedge in realized volatility and the risks you found. "
        "Use tools only as needed."
    )
    graph = build_agent_a()
    result = graph.invoke(
        {
            "ticker": ticker,
            "messages": [
                SystemMessage(content=_system_prompt()),
                HumanMessage(content=question),
            ],
        },
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    last = messages[-1]
    final_text = last.content if isinstance(last.content, str) else str(last.content)
    brief = _parse_brief(final_text)
    return {
        "query": query,
        "parsed": parsed,
        "resolved": parsed,
        "ticker": ticker,
        "question": question,
        "tool_calls": tool_call_sequence(messages),
        "final_text": final_text,
        "brief": brief.model_dump() if brief else None,
        "messages": messages,
    }


def format_answer(result: dict[str, Any]) -> str:
    """Readable notebook output: health, sentiment, 3 risks, one hedge."""
    brief = result.get("brief") or {}
    ticker = result.get("ticker") or brief.get("ticker") or "?"
    lines = [
        f"{ticker} — 90-day research brief",
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
    return "\n".join(lines)
