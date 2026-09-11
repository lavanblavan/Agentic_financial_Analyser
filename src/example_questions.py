"""Demo and evaluation questions for the agentic workflow notebook."""

from __future__ import annotations

PRIMARY_QUESTIONS: dict[str, str] = {
    "full_research_apple": (
        "Analyse the current financial health and market sentiment of apple. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    ),
    "full_research_nvidia": (
        "Analyse the current financial health and market sentiment of nvidia. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    ),
    "news_only": "what is the news for apple",
    "news_sentiment": "what is the news sentiment for tesla",
    "price_only": "what is the latest apple share price",
    "volatility_90d": "what is the 90 day volatility of microsoft",
    "volatility_30d": "what is the 30 day volatility of NVDA",
    "messy_summary": "what is the financial summary of tesla",
    "messy_search": "search for financial report of apple",
    "small_cap_news": "what is the news for enphase",
    "bare_name": "amazon",
    "bare_ticker": "AAPL",
    "adaptive_open": "tell me how apple compares to peers on growth",
    "switch_issuer": "what is the news for boeing",
}

FOLLOWUP_QUESTIONS: dict[str, str] = {
    "recall_hedge": "Remind me of the hedge and why 90-day volatility matters.",
    "recall_risks": "What were the top three risks again?",
    "recall_headlines": "Remind me of the headlines.",
    "refresh_price": "What is the latest apple price?",
    "refresh_vol": "What is the latest volatility right now?",
    "extend_news": "more news on apple — any catalyst this week?",
    "extend_search": "search for analyst views on apple share price",
    "switch_after_apple": "what is the news for tesla",
}

DEFAULT_PRIMARY = "full_research_apple"
DEFAULT_FOLLOWUP_A = "refresh_price"
DEFAULT_FOLLOWUP_B = "recall_hedge"


def list_primary() -> list[str]:
    return list(PRIMARY_QUESTIONS.keys())


def list_followups() -> list[str]:
    return list(FOLLOWUP_QUESTIONS.keys())


def get_primary(key: str = DEFAULT_PRIMARY) -> str:
    return PRIMARY_QUESTIONS.get(key, PRIMARY_QUESTIONS[DEFAULT_PRIMARY])


def get_followup(key: str) -> str:
    return FOLLOWUP_QUESTIONS[key]
