"""JSON chat client: OpenRouter first when a key is set, else Groq."""

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
    last_error: Exception | None = None
    if settings.openrouter_api_key:
        try:
            return _chat_json(
                url="https://openrouter.ai/api/v1/chat/completions",
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=settings.max_tokens,
            )
        except GroqAPIError as exc:
            last_error = exc
            if not settings.groq_api_key:
                raise
    if settings.groq_api_key:
        try:
            return _chat_json(
                url="https://api.groq.com/openai/v1/chat/completions",
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=min(settings.max_tokens, 800),
            )
        except GroqAPIError:
            if last_error:
                raise last_error
            raise
    if last_error:
        raise last_error
    raise LLMNotConfiguredError(
        "No LLM key found. Local: set OPENROUTER_API_KEY in .env "
        "(optional GROQ_API_KEY). Colab: add Secrets and grant access."
    )


def _chat_json(
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> dict[str, Any]:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise GroqAPIError(_http_error_message(response, api_key))
    raw = response.json()["choices"][0]["message"]["content"]
    return json.loads(raw)


def _http_error_message(response: requests.Response, api_key: str) -> str:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or str(payload)
    except ValueError:
        detail = response.text[:300]
    shape = groq_key_shape(api_key)
    if response.status_code == 401:
        return (
            f"HTTP 401 Unauthorized (key shape {shape}). "
            "Create a Groq key at https://console.groq.com/keys "
            "or an OpenRouter key at https://openrouter.ai/keys."
        )
    return f"LLM HTTP {response.status_code}: {detail}"
