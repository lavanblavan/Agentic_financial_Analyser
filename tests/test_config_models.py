from src.config import canonical_groq_model


def test_retired_llama_is_remapped():
    assert canonical_groq_model("llama-3.3-70b-versatile") == "openai/gpt-oss-20b"
    assert canonical_groq_model("llama-3.1-8b-instant") == "openai/gpt-oss-20b"


def test_current_model_is_unchanged():
    assert canonical_groq_model("qwen/qwen3.6-27b") == "qwen/qwen3.6-27b"
    assert canonical_groq_model("openai/gpt-oss-20b") == "openai/gpt-oss-20b"
