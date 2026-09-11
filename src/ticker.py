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
    "WHAT",
    "HOW",
    "WHY",
    "GET",
    "ASK",
}

_GENERIC_SUBJECT = {
    "analysis",
    "company",
    "financial",
    "health",
    "it",
    "news",
    "report",
    "research",
    "sentiment",
    "summary",
    "that",
    "them",
    "this",
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


def ticker_mentioned(text: str) -> str | None:
    """Find a known issuer in text without a network call. None if only pronouns."""
    raw = _clean_query(text)
    if not raw:
        return None
    lower = raw.lower()
    for name in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return _ALIASES[name]
    for token in re.findall(r"\b[A-Z]{1,5}(?:[.-][A-Z]{1,2})?\b", raw):
        if token in _STOP_TICKERS or token in PLACEHOLDER_TOKENS:
            continue
        if _is_ticker_like(token):
            return token.strip().upper().replace(".", "-")
    return None


def _alias_in_text(raw: str) -> str | None:
    lower = raw.lower()
    for name in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return name
    return None


def _trailing_company(raw: str) -> str | None:
    """Last 'of/for <name>' in a question, e.g. financial summary of tesla."""
    matches = list(
        re.finditer(
            r"\b(?:of|for)\s+([A-Za-z][\w.&'-]*(?:\s+[A-Za-z][\w.&'-]*){0,4})",
            raw,
            flags=re.I,
        )
    )
    if not matches:
        return None
    subject = matches[-1].group(1).strip(" .?'\"")
    tokens = [part.lower() for part in re.findall(r"[a-z0-9]+", subject.lower())]
    if not tokens or all(part in _GENERIC_SUBJECT for part in tokens):
        return None
    return subject


def _llm_extract_company(text: str) -> str:
    """Ask the LLM for the issuer name only. Yahoo still maps it to a ticker."""
    from src.config import project_root
    from src.groq_client import LLMNotConfiguredError, call_groq_json

    prompt_path = project_root() / "prompts" / "extract_issuer.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    try:
        payload = call_groq_json(system_prompt, text)
    except LLMNotConfiguredError as err:
        raise ValueError(
            "Could not find a company in that text, and no LLM key is set to rewrite it. "
            "Use a name/ticker (tesla, TSLA) or set OPENROUTER_API_KEY / GROQ_API_KEY."
        ) from err
    company = payload.get("company") if isinstance(payload, dict) else None
    if company is None:
        raise ValueError(
            "Could not find a company in that text. "
            "Use: Analyse ... of NVDA. ... or just type apple / AAPL."
        )
    subject = _clean_query(str(company))
    if not subject or subject.lower() in {"null", "none"}:
        raise ValueError(
            "Could not find a company in that text. "
            "Use: Analyse ... of NVDA. ... or just type apple / AAPL."
        )
    return subject


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

    alias = _alias_in_text(raw)
    if alias:
        return alias

    trailing = _trailing_company(raw)
    if trailing:
        return trailing

    if _is_ticker_like(raw):
        return raw.strip().upper().replace(".", "-")

    words = raw.split()
    if (
        len(words) <= 4
        and "analyse" not in raw.lower()
        and "analyze" not in raw.lower()
    ):
        return raw

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
    """Accept a messy question, the assessment prompt, or a bare name/ticker.

    Rules extract the issuer when they can. If that fails, an LLM rewrites the
    question down to a company name — Yahoo still maps the name to a ticker.
    """
    raw = _clean_query(text)
    extracted_via = "rules"
    subject: str | None = None
    resolved: dict[str, str] | None = None
    try:
        subject = extract_subject(raw)
        resolved = resolve_ticker(subject)
    except ValueError:
        subject = _llm_extract_company(raw)
        resolved = resolve_ticker(subject)
        extracted_via = "llm"

    if _looks_like_full_task(raw):
        task = re.sub(r"\[[^\]]+\]", resolved["ticker"], raw)
    else:
        # Pass the user's wording through; Agent A decides which tools to call.
        task = raw
    return {
        **resolved,
        "subject": subject,
        "task": task,
        "extracted_via": extracted_via,
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
