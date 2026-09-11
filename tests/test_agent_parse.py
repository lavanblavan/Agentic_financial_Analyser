from langchain_core.messages import AIMessage, HumanMessage

from src.agent_a import _extract_json_object, missing_observations, _parse_brief
from src.task_profile import FULL_RESEARCH


def test_missing_observations_lists_required_tools():
    messages = [HumanMessage(content="go")]
    missing = missing_observations(messages, FULL_RESEARCH)
    assert "get_price_data" in missing
    assert "get_news" in missing
    assert any("90" in item for item in missing)


def test_missing_observations_respects_news_only_profile():
    messages = [HumanMessage(content="go")]
    missing = missing_observations(messages, ["get_news"])
    assert missing == ["get_news"]


def test_missing_observations_clears_when_tools_ran():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_price_data", "args": {"ticker": "AAPL"}, "id": "1", "type": "tool_call"},
                {"name": "calculate_volatility", "args": {"ticker": "AAPL", "window_days": 30}, "id": "2", "type": "tool_call"},
                {"name": "calculate_volatility", "args": {"ticker": "AAPL", "window_days": 90}, "id": "3", "type": "tool_call"},
                {"name": "get_news", "args": {"ticker": "AAPL"}, "id": "4", "type": "tool_call"},
                {"name": "llm_sentiment", "args": {"ticker": "AAPL"}, "id": "5", "type": "tool_call"},
            ],
        )
    ]
    assert missing_observations(messages, FULL_RESEARCH) == []


def test_extract_json_from_prose():
    text = 'Here is the brief.\n{"ticker": "AAPL", "current_price": 1, "vol_30d_pct": 2, "momentum": "mixed"}'
    payload = _extract_json_object(text)
    assert payload is not None
    assert payload["ticker"] == "AAPL"


def test_parse_brief_accepts_core_fields():
    raw = json_blob()
    brief = _parse_brief(raw)
    assert brief is not None
    assert brief.ticker == "AAPL"


def json_blob() -> str:
    return """
    {
      "ticker": "AAPL",
      "current_price": 200.0,
      "vol_30d_pct": 22.1,
      "vol_90d_pct": 24.0,
      "momentum": "bullish",
      "sentiment_score": 0.2,
      "quantitative_risks": [
        {"name": "vol", "severity": "medium", "evidence": "30d vol 22%"}
      ]
    }
    """
