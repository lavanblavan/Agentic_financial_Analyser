from src.task_profile import FULL_RESEARCH, infer_task_profile
from src.ticker import DEFAULT_TASK


def test_full_assessment_prompt_requires_all_tools():
    query = DEFAULT_TASK.format(subject="apple")
    profile = infer_task_profile(query)
    assert profile["mode"] == "full_research"
    assert profile["required"] == FULL_RESEARCH


def test_news_question_requires_news_only():
    profile = infer_task_profile("what is the news for apple")
    assert profile["mode"] == "news"
    assert profile["required"] == ["get_news"]


def test_news_with_sentiment_adds_sentiment_tool():
    profile = infer_task_profile("what is the news sentiment for apple")
    assert profile["mode"] == "news"
    assert profile["required"] == ["get_news", "llm_sentiment"]


def test_price_question_requires_price_tool():
    profile = infer_task_profile("what is the latest apple price")
    assert profile["mode"] == "price"
    assert profile["required"] == ["get_price_data"]


def test_bare_ticker_still_full_research():
    profile = infer_task_profile("apple")
    assert profile["mode"] == "full_research"
    assert profile["required"] == FULL_RESEARCH


def test_adaptive_for_open_questions():
    profile = infer_task_profile("tell me how apple compares to peers on growth")
    assert profile["mode"] == "adaptive"
    assert profile["required"] == []


def test_volatility_90d_question():
    profile = infer_task_profile("what is the 90 day volatility of microsoft")
    assert profile["mode"] == "volatility"
    assert "calculate_volatility:90" in profile["required"]


def test_messy_summary_maps_to_news_or_adaptive():
    profile = infer_task_profile("what is the financial summary of tesla")
    assert profile["mode"] in {"news", "adaptive", "price"}


def test_small_cap_news_question():
    profile = infer_task_profile("what is the news for enphase")
    assert profile["mode"] == "news"
    assert profile["required"] == ["get_news"]
