"""Load settings from Colab Secrets or a local .env file.

Same rule as Task 1: never put API keys in the notebook or in git.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def running_in_colab() -> bool:
    return "google.colab" in sys.modules


def project_root() -> Path:
    """Repo root: folder that contains src/ and requirements.txt."""
    return Path(__file__).resolve().parent.parent


def probe_colab_secret(name: str) -> str:
    """Why a Colab secret is missing. Never returns the secret value."""
    try:
        from google.colab import userdata
    except ImportError:
        return "not_colab"
    try:
        value = userdata.get(name)
    except Exception as exc:
        kind = type(exc).__name__
        message = str(exc).lower()
        if "Access" in kind or "access" in message or "grant" in message:
            return "access_denied"
        if "NotFound" in kind or "not found" in message:
            return "not_found"
        return f"error:{kind}"
    if value and str(value).strip():
        return "present"
    return "empty"


def _secret_from_colab(name: str) -> str | None:
    if probe_colab_secret(name) != "present":
        return None
    from google.colab import userdata

    return str(userdata.get(name)).strip()


def normalize_secret(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
    return text or None


RETIRED_GROQ_MODELS = {
    "llama-3.3-70b-versatile": "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "llama-3.1-70b-versatile": "qwen/qwen3.6-27b",
}

# Groq's Llama 3.3 replacement, then OSS fallbacks if a preview ID is unavailable.
DEFAULT_AGENT_MODEL = "qwen/qwen3.6-27b"
AGENT_MODEL_FALLBACKS = (
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)


def canonical_groq_model(model: str | None) -> str:
    raw = (model or DEFAULT_AGENT_MODEL).strip()
    return RETIRED_GROQ_MODELS.get(raw, raw)


def groq_key_shape(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    if api_key.startswith("gsk_"):
        return "gsk_*"
    return "unexpected"


def _secret_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_secret(name: str) -> str | None:
    if running_in_colab():
        value = _secret_from_colab(name)
        if value:
            return normalize_secret(value)
    return normalize_secret(_secret_from_env(name))


@dataclass(frozen=True)
class Settings:
    ticker: str
    lookback: str
    groq_api_key: str | None
    groq_model: str
    groq_agent_model: str

    @property
    def llm_ready(self) -> bool:
        return bool(self.groq_api_key)


def load_settings() -> Settings:
    if not running_in_colab():
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv:
            load_dotenv(project_root() / ".env", override=False)

    json_model = canonical_groq_model(get_secret("GROQ_MODEL") or "openai/gpt-oss-20b")
    agent_model = canonical_groq_model(
        get_secret("GROQ_AGENT_MODEL") or DEFAULT_AGENT_MODEL
    )

    return Settings(
        ticker=(get_secret("TICKER") or "NVDA").upper(),
        lookback=get_secret("LOOKBACK") or "1y",
        groq_api_key=get_secret("GROQ_API_KEY"),
        groq_model=json_model,
        groq_agent_model=agent_model,
    )


def describe_env(settings: Settings) -> dict[str, str]:
    source = "colab-secrets" if running_in_colab() else "local-dotenv"
    status = {
        "runtime": "colab" if running_in_colab() else "local",
        "secret_source": source,
        "ticker": settings.ticker,
        "lookback": settings.lookback,
        "llm_provider": "groq",
        "llm_model": settings.groq_model,
        "agent_model": settings.groq_agent_model,
        "llm_key_present": "yes" if settings.llm_ready else "no",
        "groq_key_shape": groq_key_shape(settings.groq_api_key),
    }
    if running_in_colab():
        status["groq_secret_status"] = probe_colab_secret("GROQ_API_KEY")
    return status


def missing_key_help(settings: Settings) -> str:
    if settings.llm_ready:
        return ""
    if running_in_colab():
        return (
            "Colab Secret GROQ_API_KEY is missing or not granted. "
            "Add it under the key icon and enable Notebook access."
        )
    return "Local: copy .env.example to .env and set GROQ_API_KEY."
