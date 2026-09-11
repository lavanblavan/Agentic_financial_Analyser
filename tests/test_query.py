from src.ticker import extract_subject


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
