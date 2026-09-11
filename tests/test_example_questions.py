from src.example_questions import (
    FOLLOWUP_QUESTIONS,
    PRIMARY_QUESTIONS,
    get_followup,
    get_primary,
)
from src.task_profile import infer_task_profile


def test_primary_bank_covers_all_modes():
    modes = {infer_task_profile(text)["mode"] for text in PRIMARY_QUESTIONS.values()}
    assert "full_research" in modes
    assert "news" in modes
    assert "price" in modes
    assert "volatility" in modes
    assert "adaptive" in modes


def test_followup_bank_is_non_empty():
    assert len(FOLLOWUP_QUESTIONS) >= 6
    assert get_followup("recall_hedge").startswith("Remind")


def test_get_primary_default():
    assert "apple" in get_primary().lower() or "Analyse" in get_primary()
