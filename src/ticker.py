"""Resolve a company name or ticker to a Yahoo Finance equity symbol."""

from __future__ import annotations

import re

import yfinance as yf

PLACEHOLDER_TOKENS = {"TICKER", "SYMBOL", "COMPANY", "NAME"}

DEFAULT_TASK = (
    "Analyse the current financial health and market sentiment of {subject}. "
    "Identify the top three risks to its share price over the next 90 days "
    "and suggest one data-driven hedge strategy."
)

_STOP_TICKERS = {
    "AND",
    "THE",
    "FOR",
    "ITS",
    "NOT",
    "NEXT",
    "TOP",
    "ONE",
    "LLM",
    "RSI",
    "SMA",
    "MACD",
}

_ALIASES = {
    "apple": "AAPL",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "broadcom": "AVGO",
}


def _clean_query(query: str) -> str:
    text = (query or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_name(query: str) -> str:
    text = _clean_query(query).lower()
    for suffix in (
        " incorporated",
        " corporation",
        " company",
        " limited",
        " inc.",
        " inc",
        " corp.",
        " corp",
        " ltd.",
        " ltd",
        " co.",
        " co",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _is_ticker_like(query: str) -> bool:
    compact = query.strip().upper().replace("-", ".")
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?", compact))


def _yahoo_search(query: str) -> list[dict]:
    try:
        result = yf.Search(query, max_results=8, news_count=0)
        return list(result.quotes or [])
    except Exception:
        return []


def _pick_equity(quotes: list[dict]) -> dict | None:
    equities = [
        row
        for row in quotes
        if str(row.get("quoteType") or row.get("typeDisp") or "").upper()
        in {"EQUITY", "STOCK", ""}
    ]
    pool = equities or quotes
    for row in pool:
        symbol = str(row.get("symbol") or "").strip()
        if symbol:
            return row
    return None


def extract_subject(text: str) -> str:
    """Pull a company name or ticker out of a research prompt or a short label."""
    raw = _clean_query(text)
    if not raw:
        raise ValueError("Paste a research prompt or a name/ticker, e.g. apple or NVDA.")

    bracket = re.search(r"\[([^\]]+)\]", raw)
    if bracket:
        inner = bracket.group(1).strip()
        if inner.upper() in PLACEHOLDER_TOKENS:
            raise ValueError(
                "Replace [TICKER] with a name or symbol, e.g. apple, AAPL, nvidia."
            )
        return inner

    sentiment_of = re.search(
        r"market sentiment of\s+(.+?)(?:\.|Identify|identify|$)",
        raw,
        flags=re.I | re.S,
    )
    if sentiment_of:
        subject = sentiment_of.group(1).strip(" \n\t")
        if subject:
            return subject

    if len(raw) < 48 and "analyse" not in raw.lower() and "analyze" not in raw.lower():
        return raw

    lower = raw.lower()
    for name in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return name

    for token in re.findall(r"\b[A-Z]{1,5}\b", raw):
        if token not in _STOP_TICKERS:
            return token

    raise ValueError(
        "Could not find a company in that text. "
        "Use: Analyse ... of NVDA. ... or just type apple / AAPL."
    )


def _looks_like_full_task(text: str) -> bool:
    lower = text.lower()
    return "hedge" in lower and ("sentiment" in lower or "risk" in lower)


def parse_research_query(text: str) -> dict[str, str]:
    """Accept the assessment prompt, or a bare name/ticker, and resolve the issuer."""
    subject = extract_subject(text)
    resolved = resolve_ticker(subject)
    raw = _clean_query(text)
    if _looks_like_full_task(raw):
        task = re.sub(r"\[[^\]]+\]", resolved["ticker"], raw)
    else:
        task = DEFAULT_TASK.format(subject=f"{resolved['name']} ({resolved['ticker']})")
    return {
        **resolved,
        "subject": subject,
        "task": task,
    }


def normalize_symbol(query: str) -> str:
    """Map apple/AAPL to AAPL without a network call. Used for cache keys."""
    raw = _clean_query(query)
    alias = _ALIASES.get(_normalize_name(raw))
    if alias:
        return alias
    if _is_ticker_like(raw):
        return raw.strip().upper().replace(".", "-")
    return raw.upper()


def resolve_ticker(query: str) -> dict[str, str]:
    """Map 'apple', 'AAPL', or 'nvidia' to a ticker the tools can fetch.

    Returns ticker, name, and how it was resolved.
    """
    raw = _clean_query(query)
    if not raw:
        raise ValueError("Give a company name or ticker, e.g. apple, AAPL, nvidia.")

    alias = _ALIASES.get(_normalize_name(raw))
    if alias:
        quotes = _yahoo_search(alias)
        picked = _pick_equity(quotes)
        name = str((picked or {}).get("shortname") or (picked or {}).get("longname") or raw)
        return {"query": raw, "ticker": alias, "name": name, "resolved_via": "alias"}

    quotes = _yahoo_search(raw)
    picked = _pick_equity(quotes)
    if picked:
        ticker = str(picked["symbol"]).upper()
        name = str(picked.get("shortname") or picked.get("longname") or ticker)
        return {"query": raw, "ticker": ticker, "name": name, "resolved_via": "yahoo_search"}

    if _is_ticker_like(raw):
        ticker = raw.strip().upper().replace(".", "-")
        return {"query": raw, "ticker": ticker, "name": ticker, "resolved_via": "as_ticker"}

    raise ValueError(
        f"Could not map {raw!r} to a stock ticker. Try a name like 'apple' or a symbol like 'AAPL'."
    )
