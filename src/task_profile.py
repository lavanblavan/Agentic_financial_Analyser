"""Decide which tool observations a question needs.

The agent still picks call order. The graph only nudges for observations
required by the user's question, not always all five tools.
"""

from __future__ import annotations

import re

from src.ticker import _clean_query, _looks_like_full_task

# Tokens used by agent_a.missing_observations
OBS_PRICE = "get_price_data"
OBS_VOL_30 = "calculate_volatility:30"
OBS_VOL_90 = "calculate_volatility:90"
OBS_NEWS = "get_news"
OBS_SENTIMENT = "llm_sentiment"
OBS_WEB = "web_search"

FULL_RESEARCH = [OBS_PRICE, OBS_VOL_30, OBS_VOL_90, OBS_NEWS, OBS_SENTIMENT]


def infer_task_profile(query: str) -> dict[str, object]:
    """Map a user question to a task mode and required tool observations."""
    raw = _clean_query(query)
    lower = raw.lower()

    if _looks_like_full_task(raw):
        return _profile(
            "full_research",
            FULL_RESEARCH,
            "Full 90-day health, sentiment, three risks, and one vol-grounded hedge.",
        )

    if re.search(r"\b(news|headlines?|headline|catalyst|what happened)\b", lower):
        required = [OBS_NEWS]
        if re.search(r"\b(sentiment|bullish|bearish|mood)\b", lower):
            required.append(OBS_SENTIMENT)
        return _profile("news", required, "Answer with recent headlines from get_news.")

    if re.search(r"\b(price|share price|stock price|trading at|latest price|how much)\b", lower):
        required = [OBS_PRICE]
        if re.search(r"\b(volatility|vol\b)\b", lower):
            required.append(OBS_VOL_30)
        return _profile("price", required, "Answer with price and technicals from get_price_data.")

    if re.search(r"\b(volatility|realized vol|vol)\b", lower):
        required: list[str] = []
        if re.search(r"\b90|quarter|term structure|90-day|90 day\b", lower):
            required.append(OBS_VOL_90)
        if re.search(r"\b30|short[- ]term\b", lower) or not required:
            required.append(OBS_VOL_30)
        if OBS_VOL_30 in required and re.search(r"\b90|quarter|term structure|90-day|90 day\b", lower):
            if OBS_VOL_90 not in required:
                required.append(OBS_VOL_90)
        return _profile("volatility", list(dict.fromkeys(required)), "Use calculate_volatility for the horizon asked.")

    if re.search(r"\b(sentiment|market mood|bullish|bearish)\b", lower):
        return _profile(
            "sentiment",
            [OBS_NEWS, OBS_SENTIMENT],
            "Fetch headlines, then score them with llm_sentiment.",
        )

    # Bare name/ticker with no question words → full research brief.
    words = raw.split()
    if len(words) <= 2 and "?" not in raw and not re.search(r"\b(what|why|how|show|tell)\b", lower):
        return _profile("full_research", FULL_RESEARCH, "Bare issuer label — produce the full research brief.")

    # Open question: agent decides; graph does not force a checklist.
    return _profile(
        "adaptive",
        [],
        "Read the question and call only the tools you need. Skip tools that do not help answer it.",
    )


def _profile(mode: str, required: list[str], tool_guidance: str) -> dict[str, object]:
    return {
        "mode": mode,
        "required": required,
        "tool_guidance": tool_guidance,
    }
