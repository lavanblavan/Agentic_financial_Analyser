from unittest.mock import patch

from src.ticker import extract_subject, parse_research_query


PROMPT = (
    "Analyse the current financial health and market sentiment of {subject}. "
    "Identify the top three risks to its share price over the next 90 days "
    "and suggest one data-driven hedge strategy."
)


def test_extracts_ticker_from_assessment_prompt():
    assert extract_subject(PROMPT.format(subject="NVDA")) == "NVDA"
    assert extract_subject(PROMPT.format(subject="apple")) == "apple"
    assert extract_subject(PROMPT.format(subject="nvidia")) == "nvidia"


def test_extracts_bracketed_name():
    assert extract_subject(PROMPT.format(subject="[AAPL]")) == "AAPL"


def test_placeholder_ticker_is_rejected():
    try:
        extract_subject(PROMPT.format(subject="[TICKER]"))
    except ValueError as err:
        assert "Replace [TICKER]" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_bare_name_still_works():
    assert extract_subject("apple") == "apple"
    assert extract_subject("AAPL") == "AAPL"


def test_extracts_company_from_short_questions():
    assert extract_subject("what is the finanicalsummary of tesla") == "tesla"
    assert extract_subject("search for financial report of apple") == "apple"
    assert extract_subject("analyse financialsummary for tesla") == "tesla"
    assert extract_subject("what is the financial summary of enphase") == "enphase"


@patch("src.ticker.resolve_ticker")
@patch("src.ticker._llm_extract_company", return_value="Enphase Energy")
def test_llm_rewrites_when_rules_cannot_extract(llm, resolve):
    resolve.return_value = {
        "query": "Enphase Energy",
        "ticker": "ENPH",
        "name": "Enphase Energy",
        "resolved_via": "yahoo_search",
    }
    parsed = parse_research_query(
        "Tell me how that solar microinverter maker in Petaluma is doing this quarter"
    )
    llm.assert_called_once()
    assert parsed["ticker"] == "ENPH"
    assert parsed["subject"] == "Enphase Energy"
    assert parsed["extracted_via"] == "llm"


@patch("src.ticker.resolve_ticker")
@patch("src.ticker._llm_extract_company")
def test_known_question_skips_llm(llm, resolve):
    resolve.return_value = {
        "query": "tesla",
        "ticker": "TSLA",
        "name": "Tesla, Inc.",
        "resolved_via": "alias",
    }
    parsed = parse_research_query("what is the finanicalsummary of tesla")
    llm.assert_not_called()
    assert parsed["ticker"] == "TSLA"
    assert parsed["extracted_via"] == "rules"
