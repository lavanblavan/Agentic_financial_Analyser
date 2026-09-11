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
    "llama-3.3-70b-versatile": "openai/gpt-oss-20b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "llama-3.1-70b-versatile": "openai/gpt-oss-20b",
}

# Groq free OTPM is 1000; keep Groq completions under that. OpenRouter is not capped the same way.
DEFAULT_AGENT_MODEL = "openai/gpt-oss-20b"
AGENT_MAX_TOKENS = 800
OPENROUTER_MAX_TOKENS = 2048
AGENT_MODEL_FALLBACKS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
)
# OpenRouter first when a key is present. 120B free is stronger at tool calling than 20B.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_MODEL_FALLBACKS = (
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-4o-mini",
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
    openrouter_api_key: str | None
    openrouter_model: str
    max_tokens: int

    @property
    def llm_ready(self) -> bool:
        return bool(self.groq_api_key or self.openrouter_api_key)

    @property
    def prefer_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)


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
        openrouter_api_key=get_secret("OPENROUTER_API_KEY"),
        openrouter_model=get_secret("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
        max_tokens=int(
            get_secret("LLM_MAX_TOKENS")
            or (OPENROUTER_MAX_TOKENS if get_secret("OPENROUTER_API_KEY") else AGENT_MAX_TOKENS)
        ),
    )


def describe_env(settings: Settings) -> dict[str, str]:
    source = "colab-secrets" if running_in_colab() else "local-dotenv"
    if settings.groq_api_key and settings.openrouter_api_key:
        provider = "groq+openrouter"
    elif settings.groq_api_key:
        provider = "groq"
    elif settings.openrouter_api_key:
        provider = "openrouter"
    else:
        provider = "none"
    status = {
        "runtime": "colab" if running_in_colab() else "local",
        "secret_source": source,
        "ticker": settings.ticker,
        "lookback": settings.lookback,
        "llm_provider": provider,
        "llm_model": settings.openrouter_model if settings.prefer_openrouter else settings.groq_model,
        "agent_model": settings.openrouter_model if settings.prefer_openrouter else settings.groq_agent_model,
        "openrouter_model": settings.openrouter_model,
        "openrouter_key_present": "yes" if settings.openrouter_api_key else "no",
        "max_tokens": str(settings.max_tokens),
        "llm_key_present": "yes" if settings.llm_ready else "no",
        "groq_key_shape": groq_key_shape(settings.groq_api_key),
    }
    if running_in_colab():
        status["groq_secret_status"] = probe_colab_secret("GROQ_API_KEY")
        status["openrouter_secret_status"] = probe_colab_secret("OPENROUTER_API_KEY")
    return status


def missing_key_help(settings: Settings) -> str:
    if settings.llm_ready:
        return ""
    if running_in_colab():
        return (
            "No LLM key found. Add OPENROUTER_API_KEY (preferred) or GROQ_API_KEY "
            "under the key icon and enable Notebook access."
        )
    return (
        "No LLM key found. Local: set OPENROUTER_API_KEY in .env "
        "(optional GROQ_API_KEY as fallback). "
        "Colab: add the same names in Secrets and grant access."
    )
