from src.tools import _normalize_headlines


def test_normalize_headlines_accepts_title_list():
    text = _normalize_headlines(["Apple launches iPhone", "Yields surge"])
    assert "Apple launches iPhone" in text
    assert "Yields surge" in text


def test_normalize_headlines_accepts_get_news_objects():
    text = _normalize_headlines(
        [{"title": "Apple launches iPhone", "publisher": "Yahoo"}]
    )
    assert text == "Apple launches iPhone"


def test_normalize_headlines_accepts_json_string():
    text = _normalize_headlines('{"headlines": [{"title": "Hello"}]}')
    assert text == "Hello"
