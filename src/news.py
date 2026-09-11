"""Fetch recent headlines without a paid news API. Reused from Task 1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests


def fetch_news(ticker: str, min_items: int = 10) -> list[dict[str, str]]:
    headlines = _from_yfinance(ticker)
    if len(headlines) < min_items:
        headlines.extend(_from_google_news(ticker))
    return _dedupe(headlines, min_items)


def search_news_query(query: str, min_items: int = 5) -> list[dict[str, str]]:
    """Free-form Google News RSS search. Fallback for web_search."""
    return _dedupe(_from_google_news(query, query=query), min_items)


def _dedupe(headlines: list[dict[str, str]], min_items: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in headlines:
        title = item["title"].strip()
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= min_items:
            break
    return unique


def _from_yfinance(ticker: str) -> list[dict[str, str]]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items: list[dict[str, str]] = []
    for row in raw:
        title = _headline_title(row)
        if not title:
            continue
        items.append(
            {
                "title": title,
                "publisher": _nested(row, "publisher")
                or _nested(row, "content", "provider", "displayName")
                or "Yahoo",
                "date": _format_ts(row),
                "source": "yfinance",
            }
        )
    return items


def _from_google_news(ticker: str, query: str | None = None) -> list[dict[str, str]]:
    query = quote_plus(query or f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return []

    parsed = feedparser.parse(response.content)
    items: list[dict[str, str]] = []
    for entry in parsed.entries:
        title = str(getattr(entry, "title", "")).strip()
        if not title:
            continue
        source = getattr(entry, "source", None)
        publisher = "Google News"
        if isinstance(source, dict):
            publisher = str(source.get("title", "Google News"))
        items.append(
            {
                "title": title,
                "publisher": publisher,
                "date": str(getattr(entry, "published", ""))[:16],
                "source": "google-news-rss",
                "link": str(getattr(entry, "link", "")),
            }
        )
    return items


def _headline_title(row: dict[str, Any]) -> str:
    return (
        _nested(row, "title")
        or _nested(row, "content", "title")
        or _nested(row, "content", "contentTitle")
        or ""
    )


def _nested(row: dict[str, Any], *keys: str) -> str | None:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    text = str(current).strip()
    return text or None


def _format_ts(row: dict[str, Any]) -> str:
    ts = row.get("providerPublishTime") or _nested(row, "content", "pubDate")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    if ts:
        return str(ts)[:16]
    return ""
