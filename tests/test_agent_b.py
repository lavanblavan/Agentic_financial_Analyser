from src.agent_b import brief_gaps, decision_from_gaps, format_final_report, report_from_brief
from src.schemas import DataBrief, RiskFactor


def _brief(**overrides) -> DataBrief:
    payload = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 326.57,
        "vol_30d_pct": 31.25,
        "vol_90d_pct": 29.3,
        "momentum": "bullish",
        "sentiment_score": 0.5,
        "sentiment_label": "bullish",
        "headlines": ["Apple launches iPhone 18 Pro and iPhone 18 Pro Max"],
        "quantitative_risks": [
            RiskFactor(name="iPhone 18 demand miss", severity="high", evidence="Launch headline; leasing as prices increase"),
            RiskFactor(name="Hardware vs AI narrative", severity="medium", evidence="Buyers prefer hardware over AI features"),
            RiskFactor(name="Macro CPI / yields", severity="medium", evidence="CPI inflation due in the tape"),
        ],
        "hedge_strategy": "90-day protective put sized off 29.3% 90d vol vs 31.25% 30d vol.",
    }
    payload.update(overrides)
    return DataBrief.model_validate(payload)


def test_missing_90d_vol_is_the_first_request():
    brief = _brief(vol_90d_pct=None, hedge_strategy="buy a put")
    gaps = brief_gaps(brief)
    assert "vol_90" in gaps
    decision = decision_from_gaps(brief, gaps)
    assert decision.need_more_data
    assert decision.request_kind == "vol_90"


def test_generic_risks_request_web_search():
    brief = _brief(
        quantitative_risks=[
            {"name": "Supply Chain Disruptions", "severity": "high", "evidence": "Potential issues with hardware supply chains could impact product availability."},
            {"name": "Market Volatility", "severity": "medium", "evidence": "Increased short-term volatility may lead to price fluctuations."},
            {"name": "Consumer Demand Fluctuations", "severity": "medium", "evidence": "Concerns about product issues may temper consumer enthusiasm."},
        ]
    )
    gaps = brief_gaps(brief)
    assert "generic_risks" in gaps
    decision = decision_from_gaps(brief, gaps)
    assert decision.request_kind == "web_search"


def test_hedge_without_90d_requests_revise():
    brief = _brief(hedge_strategy="Consider a protective put given 30-day volatility of 31.25%.")
    gaps = brief_gaps(brief)
    assert "hedge_90d" in gaps
    decision = decision_from_gaps(brief, gaps)
    assert decision.request_kind == "revise"


def test_grounded_brief_has_no_gaps():
    assert brief_gaps(_brief()) == []


def test_news_mode_only_checks_headlines():
    brief = _brief(
        vol_90d_pct=None,
        hedge_strategy="",
        quantitative_risks=[],
        headlines=["Apple launches iPhone 18 Pro"],
    )
    assert brief_gaps(brief, mode="news") == []
    assert brief_gaps(brief, mode="full_research")


def test_final_report_marks_90d_vol():
    brief = _brief()
    report = report_from_brief(brief, [{"request_kind": "revise", "reason": "hedge"}], revisions=1)
    assert report.used_90d_vol is True
    assert report.critique_rounds == 1
    text = format_final_report({"report": report.model_dump(), "critiques": [{"request_kind": "revise", "reason": "hedge"}]})
    assert "used_90d_vol=True" in text
    assert "Critique trail" in text
