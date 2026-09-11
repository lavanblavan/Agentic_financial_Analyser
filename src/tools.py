"""Five agent tools. Implementations reuse Task 1 Financial_AI modules.

The LLM chooses which of these to call and in what order. Do not hard-code
a call sequence in the graph.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import project_root
from src.data import fetch_prices
from src.groq_client import call_groq_json
from src.indicators import add_indicators, latest_snapshot, realized_vol_pct
from src.news import fetch_news, search_news_query
from src.memory import disk_cached
from src.ticker import resolve_ticker
from src.tracing import traced


def _symbol(query: str) -> str:
    return resolve_ticker(query)["ticker"]


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=str)


def _normalize_headlines(headlines: Any) -> str:
    """Accept a title list, get_news JSON, or a raw string. Groq often sends an array."""
    if headlines is None or headlines == "":
        return "none"
    if isinstance(headlines, str):
        text = headlines.strip()
        if not text:
            return "none"
        if text.startswith("{") or text.startswith("["):
            try:
                return _normalize_headlines(json.loads(text))
            except json.JSONDecodeError:
                return text
        return text
    if isinstance(headlines, dict):
        items = headlines.get("headlines", headlines)
        return _normalize_headlines(items)
    if isinstance(headlines, list):
        lines: list[str] = []
        for item in headlines:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                if title:
                    lines.append(title)
            elif item:
                lines.append(str(item).strip())
        return "\n".join(lines) if lines else "none"
    return str(headlines)


class SentimentArgs(BaseModel):
    ticker: str = Field(description="Company name or ticker, e.g. AAPL or apple")
    headlines: list[str] = Field(
        default_factory=list,
        description="Headline title strings from get_news (a list, not a JSON string).",
    )


@traced
@disk_cached
def _get_price_data(ticker: str, period: str = "1y") -> str:
    """Latest price plus SMA/RSI/MACD/Bollinger snapshot. Accepts 'apple' or 'AAPL'."""
    symbol = _symbol(ticker)
    frame = add_indicators(fetch_prices(symbol, period=period))
    snap = latest_snapshot(frame)
    return _dumps({"query": ticker, "ticker": symbol, **snap, "bars": int(len(frame))})


@traced
@disk_cached
def _calculate_volatility(ticker: str, window_days: int = 30, period: str = "1y") -> str:
    """Annualized realized volatility (%) over window_days. Use 30 or 90. Accepts a name or ticker."""
    window = int(window_days)
    if window not in {20, 30, 60, 90}:
        window = 30
    symbol = _symbol(ticker)
    close = fetch_prices(symbol, period=period)["Close"]
    vol = realized_vol_pct(close, window=window)
    return _dumps(
        {
            "query": ticker,
            "ticker": symbol,
            "window_days": window,
            "vol_pct": round(vol, 4),
            "method": "close-to-close annualized stdev * sqrt(252) * 100",
        }
    )


@traced
@disk_cached
def _get_news(ticker: str, min_items: int = 8) -> str:
    """Recent headlines from Yahoo Finance, with Google News RSS fallback. Accepts a name or ticker."""
    symbol = _symbol(ticker)
    headlines = fetch_news(symbol, min_items=int(min_items))
    return _dumps({"query": ticker, "ticker": symbol, "count": len(headlines), "headlines": headlines})


@traced
@disk_cached
def _llm_sentiment(ticker: str, headlines: list[str] | None = None) -> str:
    """Score news sentiment from -1 to 1. headlines must be a list of title strings."""
    symbol = _symbol(ticker)
    prompt = (project_root() / "prompts" / "sentiment.md").read_text(encoding="utf-8")
    user = f"TICKER: {symbol}\nHEADLINES:\n{_normalize_headlines(headlines)}"
    payload = call_groq_json(prompt, user)
    score = payload.get("sentiment_score", 0.0)
    try:
        score = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    payload["query"] = ticker
    payload["ticker"] = symbol
    payload["sentiment_score"] = score
    payload.setdefault("label", "neutral")
    payload.setdefault("rationale", "")
    payload.setdefault("key_themes", [])
    return _dumps(payload)


@traced
@disk_cached
def _web_search(query: str, max_results: int = 5) -> str:
    """Search the open web for analyst views, catalysts, or recent events."""
    results = _ddg_search(query, max_results=int(max_results))
    source = "ddgs"
    if not results:
        results = search_news_query(query, min_items=int(max_results))
        source = "google-news-rss"
    return _dumps({"query": query, "source": source, "count": len(results), "results": results})


def _ddg_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    try:
        rows = DDGS().text(query, max_results=max_results) or []
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "title": str(row.get("title") or ""),
                "url": str(row.get("href") or ""),
                "snippet": str(row.get("body") or ""),
                "source": "ddgs",
            }
        )
    return out


get_price_data = StructuredTool.from_function(
    func=_get_price_data,
    name="get_price_data",
    description=(
        "Fetch the latest OHLCV snapshot and technicals (SMA 50/200, RSI-14, "
        "MACD, Bollinger position, momentum bias). "
        "ticker may be a company name or symbol, e.g. apple, AAPL, nvidia. "
        "Use this first when you need price context."
    ),
)

calculate_volatility = StructuredTool.from_function(
    func=_calculate_volatility,
    name="calculate_volatility",
    description=(
        "Compute annualized realized volatility in percent. "
        "ticker may be a company name or symbol. "
        "Set window_days to 30 for short-term vol or 90 for term-structure context. "
        "Call this when RSI or price alone is not enough to judge the vol regime."
    ),
)

get_news = StructuredTool.from_function(
    func=_get_news,
    name="get_news",
    description=(
        "Fetch recent company headlines from Yahoo Finance and Google News RSS. "
        "ticker may be a company name or symbol. Use before scoring sentiment."
    ),
)

llm_sentiment = StructuredTool.from_function(
    func=_llm_sentiment,
    name="llm_sentiment",
    args_schema=SentimentArgs,
    description=(
        "Score headline sentiment from -1 (bearish) to 1 (bullish). "
        "ticker may be a company name or symbol. "
        "Pass headlines as a list of title strings from get_news, e.g. "
        "[\"Apple launches iPhone 18 Pro\", \"...\"]. Do not pass a JSON string."
    ),
)

web_search = StructuredTool.from_function(
    func=_web_search,
    name="web_search",
    description=(
        "Search the web for analyst views, catalysts, or corroborating events. "
        "Use after price/news when you need outside context."
    ),
)

ALL_TOOLS = [
    get_price_data,
    calculate_volatility,
    get_news,
    llm_sentiment,
    web_search,
]
