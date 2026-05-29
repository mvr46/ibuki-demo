"""Tests for the local Ollama LLM adapter."""

from __future__ import annotations
import json
import asyncio
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from reachy_mini_conversation_app.backends.local_llm import (
    OllamaLLMAdapter,
    OllamaToolRouter,
    LocalToolRoutingResult,
    _parse_router_content,
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
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        response = adapter._chat_sync(
            [{"role": "user", "content": "dance"}],
            [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
        )

    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["timeout"] == 9
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_predict"] == 192
    assert captured["payload"]["tools"][0]["function"]["name"] == "dance"
    assert response.tool_calls == [{"name": "dance", "arguments": {"style": "wave"}}]


def test_ollama_llm_retries_without_tools_when_schema_is_rejected() -> None:
    """Tool schema rejection should not prevent a spoken local response."""
    payloads = []
    diagnostics = {}

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

    adapter = OllamaLLMAdapter(
        base_url="http://ollama.test",
        model="gemma3:test",
        diagnostics=type("Diagnostics", (), {"set_local_model": lambda self, **payload: diagnostics.update(payload)})(),
    )
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        response = adapter._chat_sync(
            [{"role": "user", "content": "say hi"}],
            [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
        )

    assert response.content == "hello"
    assert "tools" in payloads[0]
    assert "tools" not in payloads[1]
    assert diagnostics["configured_model"] == "gemma3:test"
    assert diagnostics["tools_disabled_by_model"] is True
    assert diagnostics["last_tool_status"] == "rejected"


def test_ollama_tool_router_posts_compact_structured_payload() -> None:
    """Qwen router should use a compact JSON payload and normalize one tool call."""
    captured = {}

    class RouterResponse:
        def __enter__(self) -> "RouterResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {"tool_calls": [{"name": "look_at_person", "arguments": {"name": "Matt"}}]}
                        )
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(req: object, timeout: float) -> RouterResponse:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return RouterResponse()

    router = OllamaToolRouter(base_url="http://ollama.test", model="qwen3.5:test", timeout_seconds=3)
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        result = router._route_sync(
            "look at Matt",
            [
                {
                    "type": "function",
                    "name": "look_at_person",
                    "description": "Turn toward a named visible person.",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            ],
        )

    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["timeout"] == 3
    assert captured["payload"]["model"] == "qwen3.5:test"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_predict"] <= 64
    assert captured["payload"]["format"]["required"] == ["tool_calls"]
    user_content = json.loads(captured["payload"]["messages"][1]["content"])
    assert "tools" in user_content
    assert "available_tools" not in user_content
    assert user_content["tools"][0]["arguments"]["name"]["type"] == "string"
    assert result.tool_calls == [{"name": "look_at_person", "arguments": {"name": "Matt"}}]


def test_qwen_router_parser_accepts_raw_json() -> None:
    """Router parser should accept the exact raw JSON shape requested from Qwen."""
    parsed = _parse_router_content('{"tool_calls":[{"name":"who_am_i","arguments":{}}]}')

    assert parsed == [{"name": "who_am_i", "arguments": {}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_strips_fenced_json() -> None:
    """Router parser should tolerate Markdown fences returned by Qwen."""
    parsed = _parse_router_content('```json\n{"tool_calls":[{"name":"camera","arguments":{"question":"what do you see"}}]}\n```')

    assert parsed == [{"name": "camera", "arguments": {"question": "what do you see"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_accepts_parameters_alias() -> None:
    """Router parser should normalize Qwen's observed parameters alias to arguments."""
    parsed = _parse_router_content('{"tool_calls":[{"name":"move_head","parameters":{"direction":"left"}}]}')

    assert parsed == [{"name": "move_head", "arguments": {"direction": "left"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_marks_invalid_json() -> None:
    """Invalid router output should be logged as a parse failure without a fallback call."""
    parsed = _parse_router_content("not json")

    assert parsed == []
    assert parsed.parse_error is True


def test_ollama_tool_router_ignores_unknown_qwen_tool() -> None:
    """Unknown Qwen-selected tools should be dropped without substituting a deterministic call."""
    class RouterResponse:
        def __enter__(self) -> "RouterResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": '{"tool_calls":[{"name":"unknown_tool","arguments":{}}]}'}}).encode(
                "utf-8"
            )

    def fake_urlopen(req: object, timeout: float) -> RouterResponse:
        return RouterResponse()

    router = OllamaToolRouter(base_url="http://ollama.test", model="qwen3.5:test")
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        result = router._route_sync(
            "do something",
            [{"type": "function", "name": "camera", "description": "Camera", "parameters": {}}],
        )

    assert result.tool_calls == []
    assert result.ignored_tool_name == "unknown_tool"
    assert result.parse_error is False


def test_ollama_tool_router_reports_qwen_no_tool_without_deterministic_shortcut() -> None:
    """Qwen no-tool responses should stay no-tool, even for action-like utterances."""
    async def run() -> None:
        diagnostics = {}
        router = OllamaToolRouter(
            diagnostics=type(
                "Diagnostics",
                (),
                {"set_local_model": lambda self, **payload: diagnostics.update(payload)},
            )()
        )
        router._route_sync = lambda _text, _tools: LocalToolRoutingResult(tool_calls=[])  # type: ignore[method-assign]
        tools = [{"type": "function", "name": "move_head", "description": "Move head", "parameters": {}}]

        result = await router.route("move your head left", tools)

        assert result.tool_calls == []
        assert diagnostics["qwen_router_status"] == "qwen_no_tool"

    asyncio.run(run())


def test_ollama_tool_router_reports_qwen_parse_error() -> None:
    """Malformed Qwen output should surface as qwen_parse_error for diagnostics."""
    async def run() -> None:
        diagnostics = {}
        router = OllamaToolRouter(
            diagnostics=type(
                "Diagnostics",
                (),
                {"set_local_model": lambda self, **payload: diagnostics.update(payload)},
            )()
        )
        router._route_sync = lambda _text, _tools: LocalToolRoutingResult(  # type: ignore[method-assign]
            tool_calls=[],
            raw_content="```json\n{bad",
            parse_error=True,
        )
        tools = [{"type": "function", "name": "camera", "description": "Camera", "parameters": {}}]

        result = await router.route("what do you see?", tools)

        assert result.tool_calls == []
        assert diagnostics["qwen_router_status"] == "qwen_parse_error"
        assert diagnostics["last_ollama_error"] == "```json\n{bad"

    asyncio.run(run())
