"""Tests for OpenAICompatBackend (local/cloud OpenAI-protocol servers)."""

from __future__ import annotations

import httpx
import pytest
from kindalive.interpreter.openai_backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatBackend,
)


def test_defaults_point_at_local_ollama(monkeypatch):
    monkeypatch.delenv("KINDALIVE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("KINDALIVE_LLM_MODEL", raising=False)
    backend = OpenAICompatBackend()
    assert backend.base_url == DEFAULT_BASE_URL.rstrip("/")
    assert backend.model == DEFAULT_MODEL


def test_env_configuration(monkeypatch):
    monkeypatch.setenv("KINDALIVE_LLM_BASE_URL", "http://box:1234/v1/")
    monkeypatch.setenv("KINDALIVE_LLM_MODEL", "qwen2.5")
    backend = OpenAICompatBackend()
    assert backend.base_url == "http://box:1234/v1"
    assert backend.model == "qwen2.5"


def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("KINDALIVE_LLM_BASE_URL", "http://box:1234/v1")
    backend = OpenAICompatBackend(base_url="http://other:8000/v1",
                                  model="phi3")
    assert backend.base_url == "http://other:8000/v1"
    assert backend.model == "phi3"


def test_key_alias_prefers_kindalive_llm_key(monkeypatch):
    """KINDALIVE_LLM_KEY wins over OPENAI_API_KEY (so an OpenRouter key
    doesn't have to masquerade as an OpenAI key)."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("KINDALIVE_LLM_KEY", "openrouter-key")
    backend = OpenAICompatBackend(base_url="https://openrouter.ai/api/v1")
    assert backend._api_key == "openrouter-key"

    monkeypatch.delenv("KINDALIVE_LLM_KEY")
    fallback = OpenAICompatBackend(base_url="https://openrouter.ai/api/v1")
    assert fallback._api_key == "openai-key"


@pytest.mark.asyncio
async def test_attribution_headers_for_openrouter(monkeypatch):
    """X-Title defaults to Kindalive; HTTP-Referer is sent when set."""
    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={"choices": [{"message": {"content": "[]"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.delenv("KINDALIVE_LLM_TITLE", raising=False)
    backend = OpenAICompatBackend(
        base_url="https://openrouter.ai/api/v1",
        referer="https://example.com/kindalive",
    )
    await backend.call("s", "u")
    assert captured["headers"]["X-Title"] == "Kindalive"
    assert captured["headers"]["HTTP-Referer"] == "https://example.com/kindalive"


@pytest.mark.asyncio
async def test_call_posts_chat_completion(monkeypatch):
    """The backend speaks the /chat/completions protocol."""
    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {"message": {"content": '[{"chemical": "dopamine"}]'}}
                ]
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    backend = OpenAICompatBackend(
        base_url="http://localhost:11434/v1", model="llama3.1",
        api_key="secret",
    )
    result = await backend.call("system text", "user text")

    assert result == '[{"chemical": "dopamine"}]'
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    payload = captured["payload"]
    assert payload["model"] == "llama3.1"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0] == {
        "role": "system", "content": "system text",
    }
    assert payload["messages"][1] == {
        "role": "user", "content": "user text",
    }
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_call_includes_conversation_history(monkeypatch):
    """Prior turns are inserted between the system prompt and the new
    user message, in order."""
    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={"choices": [{"message": {"content": "[]"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    backend = OpenAICompatBackend(base_url="http://localhost:11434/v1")
    history = [
        {"role": "user", "content": "I won the lottery"},
        {"role": "assistant", "content": '{"reply": "Congrats!"}'},
    ]
    await backend.call("system text", "it was only five dollars",
                       history=history)
    msgs = captured["payload"]["messages"]
    assert msgs[0] == {"role": "system", "content": "system text"}
    assert msgs[1] == history[0]
    assert msgs[2] == history[1]
    assert msgs[3] == {"role": "user", "content": "it was only five dollars"}


@pytest.mark.asyncio
async def test_call_omits_auth_header_without_key(monkeypatch):
    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={"choices": [{"message": {"content": "[]"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = OpenAICompatBackend(base_url="http://localhost:11434/v1")
    await backend.call("s", "u")
    assert "Authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_call_raises_on_http_error(monkeypatch):
    async def fake_post(self, url, json=None, headers=None):
        request = httpx.Request("POST", url)
        return httpx.Response(500, request=request, text="boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    backend = OpenAICompatBackend(base_url="http://localhost:11434/v1")
    with pytest.raises(httpx.HTTPStatusError):
        await backend.call("s", "u")
    assert backend.call_count == 0
