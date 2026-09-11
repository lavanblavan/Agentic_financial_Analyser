import json

from src.memory import (
    ask_agent_a,
    cache_get,
    cache_put,
    classify_intent,
    describe_memory,
    disk_cached,
    is_followup,
    load_session,
    recall,
    relate,
    remember,
)
from src.ticker import normalize_symbol, ticker_mentioned


def test_normalize_symbol_does_not_need_network():
    assert normalize_symbol("apple") == "AAPL"
    assert normalize_symbol("AAPL") == "AAPL"
    assert ticker_mentioned("Remind me of the hedge.") is None
    assert ticker_mentioned("what about tesla") == "TSLA"


def test_disk_cache_hit_marks_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.CACHE_DIR", tmp_path)
    calls = {"n": 0}

    @disk_cached
    def _get_price_data(ticker: str, period: str = "1y") -> str:
        calls["n"] += 1
        return json.dumps({"ticker": ticker, "close": 10})

    first = json.loads(_get_price_data("AAPL"))
    second = json.loads(_get_price_data("apple"))
    assert calls["n"] == 1
    assert first.get("cached") is None
    assert second["cached"] is True
    assert second["close"] == 10


def test_cache_expires(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.CACHE_DIR", tmp_path)
    args = {"ticker": "AAPL", "period": "1y"}
    cache_put("_get_price_data", args, json.dumps({"close": 1}))
    assert cache_get("_get_price_data", args, ttl=60) is not None
    assert cache_get("_get_price_data", args, ttl=-1) is None


def test_followup_uses_last_ticker_without_new_research(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.SESSION_PATH", tmp_path / "session.json")
    remember(
        {
            "query": "Analyse apple ...",
            "ticker": "AAPL",
            "brief": {"ticker": "AAPL", "current_price": 1, "vol_30d_pct": 2, "momentum": "mixed", "sentiment_score": 0},
            "report": {"hedge_or_strategy": "90d put"},
        }
    )
    session = load_session()
    assert is_followup("Remind me of the hedge.", session)
    assert is_followup("what is the hedge?", session)
    assert not is_followup(
        "Analyse the current financial health and market sentiment of tesla. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy.",
        session,
    )
    stored = recall("apple", session)
    assert stored is not None
    assert stored["report"]["hedge_or_strategy"] == "90d put"
    status = describe_memory(session)
    assert status["last_ticker"] == "AAPL"
    assert status["brief_cached"] is True

    recall_plan = relate("Remind me of the hedge.", session)
    assert recall_plan["related"] is True
    assert recall_plan["intent"] == "recall"
    assert recall_plan["need_tools"] == []
    assert "hedge" in recall_plan["reuse"]

    refresh_plan = relate("What is the latest AAPL price?", session)
    assert refresh_plan["related"] is True
    assert refresh_plan["intent"] == "refresh"
    assert "get_price_data" in refresh_plan["need_tools"]

    new_plan = relate(
        "Analyse the current financial health and market sentiment of tesla. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy.",
        session,
    )
    assert new_plan["related"] is False
    assert classify_intent(new_plan.get("previous_query") or "Analyse apple") == "research"


def test_ask_agent_a_reuses_memory_for_followup(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.SESSION_PATH", tmp_path / "session.json")
    remember(
        {
            "query": "what is the news for apple",
            "ticker": "AAPL",
            "task_mode": "news",
            "brief": {
                "ticker": "AAPL",
                "current_price": 200,
                "vol_30d_pct": 22,
                "momentum": "mixed",
                "sentiment_score": 0.1,
                "headlines": ["Apple launches iPhone 18 Pro"],
            },
        }
    )
    result = ask_agent_a("Remind me of the headlines.")
    assert result["from_memory"] is True
    assert result["need_tools"] == []
    assert result.get("followup_answer")

