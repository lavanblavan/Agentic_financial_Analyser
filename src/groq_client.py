"""Small Groq JSON client. Same key-loading rules as Task 1."""

from __future__ import annotations

import json
from typing import Any

import requests

from src.config import Settings, groq_key_shape, load_settings


class LLMNotConfiguredError(RuntimeError):
    pass


class GroqAPIError(RuntimeError):
    pass


def call_groq_json(system_prompt: str, user_content: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    if not settings.llm_ready:
        raise LLMNotConfiguredError(
            "No LLM key found. Local: set GROQ_API_KEY in .env. "
            "Colab: add GROQ_API_KEY in Secrets and grant access."
        )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.groq_model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise GroqAPIError(_groq_error_message(response, settings.groq_api_key or ""))
    raw = response.json()["choices"][0]["message"]["content"]
    return json.loads(raw)


def _groq_error_message(response: requests.Response, api_key: str) -> str:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or str(payload)
    except ValueError:
        detail = response.text[:300]
    shape = groq_key_shape(api_key)
    if response.status_code == 401:
        return (
            f"Groq 401 Unauthorized (key shape {shape}). "
            "Create a new key at https://console.groq.com/keys."
        )
    return f"Groq HTTP {response.status_code}: {detail}"
