"""Tests for the local Ollama LLM adapter."""

from __future__ import annotations
import json
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from reachy_mini_conversation_app.local_llm import (
    OllamaLLMAdapter,
    ollama_tool_schemas,
    ollama_tool_call_messages,
)


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "dance", "arguments": {"style": "wave"}},
                        }
                    ],
                }
            }
        ).encode("utf-8")


def test_ollama_tool_schemas_wrap_existing_tool_specs() -> None:
    """Local tool specs should be sent in Ollama's function wrapper format."""
    converted = ollama_tool_schemas(
        [
            {
                "type": "function",
                "name": "dance",
                "description": "Do a dance",
                "parameters": {"type": "object"},
            }
        ]
    )

    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "dance",
                "description": "Do a dance",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_ollama_tool_call_messages_use_tool_history_shape() -> None:
    """Normalized tool calls should be restorable into Ollama chat history."""
    assert ollama_tool_call_messages([{"name": "dance", "arguments": {"style": "wave"}}]) == [
        {
            "type": "function",
            "function": {"index": 0, "name": "dance", "arguments": {"style": "wave"}},
        }
    ]


def test_ollama_llm_posts_wrapped_tools_and_normalizes_response() -> None:
    """Ollama adapter should send wrapped tools and return normalized tool calls."""
    captured = {}

    def fake_urlopen(req: object, timeout: float) -> _Response:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    adapter = OllamaLLMAdapter(base_url="http://ollama.test", model="gemma3:test", timeout_seconds=9)
    with patch("reachy_mini_conversation_app.local_llm.urlopen", fake_urlopen):
        response = adapter._chat_sync(
            [{"role": "user", "content": "dance"}],
            [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
        )

    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["timeout"] == 9
    assert captured["payload"]["tools"][0]["function"]["name"] == "dance"
    assert response.tool_calls == [{"name": "dance", "arguments": {"style": "wave"}}]


def test_ollama_llm_retries_without_tools_when_schema_is_rejected() -> None:
    """Tool schema rejection should not prevent a spoken local response."""
    payloads = []

    class TextResponse:
        def __enter__(self) -> "TextResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "hello", "tool_calls": []}}).encode("utf-8")

    def fake_urlopen(req: object, timeout: float) -> TextResponse:
        payload = json.loads(req.data.decode("utf-8"))
        payloads.append(payload)
        if "tools" in payload:
            raise HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(b"invalid tools"))
        return TextResponse()

    adapter = OllamaLLMAdapter(base_url="http://ollama.test", model="gemma3:test")
    with patch("reachy_mini_conversation_app.local_llm.urlopen", fake_urlopen):
        response = adapter._chat_sync(
            [{"role": "user", "content": "say hi"}],
            [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
        )

    assert response.content == "hello"
    assert "tools" in payloads[0]
    assert "tools" not in payloads[1]
