from src.config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_MODEL_FALLBACKS, canonical_groq_model


def test_retired_llama_is_remapped():
    assert canonical_groq_model("llama-3.3-70b-versatile") == "openai/gpt-oss-20b"
    assert canonical_groq_model("llama-3.1-8b-instant") == "openai/gpt-oss-20b"


def test_current_model_is_unchanged():
    assert canonical_groq_model("qwen/qwen3.6-27b") == "qwen/qwen3.6-27b"
    assert canonical_groq_model("openai/gpt-oss-20b") == "openai/gpt-oss-20b"


def test_openrouter_defaults_prefer_stronger_free_model():
    assert DEFAULT_OPENROUTER_MODEL == "openai/gpt-oss-120b:free"
    assert "meta-llama/llama-3.3-70b-instruct:free" in OPENROUTER_MODEL_FALLBACKS
