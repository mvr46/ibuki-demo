"""Tests for the local llama.cpp/OpenAI-compatible LLM adapter."""

from __future__ import annotations
import json
from unittest.mock import patch

import httpx
import pytest

from reachy_mini_conversation_app.backends.local_llm import (
    LocalToolRoutingResult,
    OpenAICompatibleLLMAdapter,
    OpenAICompatibleToolRouter,
    _parse_router_content,
    create_local_llm_adapter,
    create_local_tool_router,
    _openai_compatible_messages,
)


@pytest.mark.asyncio
async def test_openai_compatible_llm_stream_chat_yields_sse_deltas() -> None:
    """llama.cpp-style streaming should parse OpenAI-compatible SSE chunks."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"there"}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8"),
        )

    adapter = OpenAICompatibleLLMAdapter(
        base_url="http://llama.test/v1",
        model="gemma-gguf",
        http_transport=httpx.MockTransport(handler),
    )
    chunks = [chunk async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}], [])]

    assert chunks == ["hello ", "there"]
    assert captured["url"] == "http://llama.test/v1/chat/completions"
    assert captured["payload"]["model"] == "gemma-gguf"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["max_tokens"] == 96
    assert "keep_alive" not in captured["payload"]
    assert "options" not in captured["payload"]


@pytest.mark.asyncio
async def test_openai_compatible_llm_chat_normalizes_response() -> None:
    """Non-streaming OpenAI-compatible responses should match the local adapter contract."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured["payload"] = payload
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello", "tool_calls": []}}]},
        )

    adapter = OpenAICompatibleLLMAdapter(
        base_url="http://llama.test/v1",
        model="gemma-gguf",
        http_transport=httpx.MockTransport(handler),
    )

    response = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
    )

    assert response.content == "hello"
    assert response.tool_calls == []
    assert "tools" not in captured["payload"]
    assert "format" not in captured["payload"]


def test_create_local_llm_adapter_uses_openai_compatible_provider() -> None:
    """The local backend always uses the llama.cpp/OpenAI-compatible chat path."""
    adapter = create_local_llm_adapter()

    assert isinstance(adapter, OpenAICompatibleLLMAdapter)


def test_create_local_tool_router_uses_openai_compatible_provider() -> None:
    """The local backend keeps Qwen on the llama.cpp router server."""
    router = create_local_tool_router()

    assert isinstance(router, OpenAICompatibleToolRouter)


def test_openai_compatible_messages_fold_intermediate_system_context() -> None:
    """Gemma llama.cpp templates require strict alternation after the first system message."""
    converted = _openai_compatible_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "move left"},
            {"role": "assistant", "content": "Okay."},
            {"role": "tool", "content": "Tool result: movement completed"},
            {"role": "user", "content": "say hello"},
        ]
    )

    assert [message["role"] for message in converted] == ["system", "user", "assistant", "user"]
    assert "Context:" in converted[-1]["content"]
    assert "Tool result: movement completed" in converted[-1]["content"]


def test_openai_compatible_tool_router_posts_compact_completion_payload() -> None:
    """llama.cpp router should use a tiny completion payload with no tool schemas."""
    captured = {}

    class RouterResponse:
        def __enter__(self) -> "RouterResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"text": " move_head|left"}]}).encode("utf-8")

    def fake_urlopen(req: object, timeout: float) -> RouterResponse:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return RouterResponse()

    router = OpenAICompatibleToolRouter(base_url="http://llama.test/v1", model="qwen3-0.6b", timeout_seconds=3)
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        result = router._route_sync(
            "move your head left",
            [{"type": "function", "name": "move_head", "description": "Move head", "parameters": {}}],
        )

    assert captured["url"] == "http://llama.test/v1/completions"
    assert captured["timeout"] == 3
    assert captured["payload"]["model"] == "qwen3-0.6b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 18
    assert captured["payload"]["stop"] == ["\n"]
    assert "tools" not in captured["payload"]
    assert "format" not in captured["payload"]
    assert "messages" not in captured["payload"]
    assert "move_head" in captured["payload"]["prompt"]
    assert '"enum"' not in captured["payload"]["prompt"]
    assert result.tool_calls == [{"name": "move_head", "arguments": {"direction": "left"}}]


def test_qwen_router_parser_accepts_none_contract() -> None:
    """Router parser should accept the exact no-tool contract requested from Qwen."""
    parsed = _parse_router_content("none|")

    assert parsed == []
    assert parsed.parse_error is False


def test_qwen_router_parser_accepts_loose_none_text() -> None:
    """Small completion routers sometimes omit the pipe for no-tool turns."""
    parsed = _parse_router_content("none none")

    assert parsed == []
    assert parsed.parse_error is False


def test_qwen_router_parser_maps_camera_arg() -> None:
    """Router parser should map camera args into the existing tool schema."""
    parsed = _parse_router_content("camera|what do you see?", user_text="what do you see?")

    assert parsed == [{"name": "camera", "arguments": {"question": "what do you see?"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_maps_camera_question_sentinel_with_punctuation() -> None:
    """Completion routers may add a period after the sentinel; keep original question."""
    parsed = _parse_router_content("camera|question.", user_text="what do you see?")

    assert parsed == [{"name": "camera", "arguments": {"question": "what do you see?"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_maps_move_head_arg() -> None:
    """Router parser should validate move_head directions locally."""
    parsed = _parse_router_content("move_head|left")

    assert parsed == [{"name": "move_head", "arguments": {"direction": "left"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_maps_generic_dance_to_random() -> None:
    """Generic dance should preserve random move behavior."""
    parsed = _parse_router_content("dance|")

    assert parsed == [{"name": "dance", "arguments": {}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_maps_dance_none_to_random() -> None:
    """Small completion routers may write a none arg for random optional choices."""
    parsed = _parse_router_content("dance|none")

    assert parsed == [{"name": "dance", "arguments": {}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_drops_non_explicit_dance_move() -> None:
    """A model-selected dance move should be cleared unless the user said it."""
    tools = [
        {
            "name": "dance",
            "parameters": {"properties": {"move": {"enum": ["simple_nod", "robot_wave"]}}},
        }
    ]
    parsed = _parse_router_content("dance|simple_nod", user_text="can you dance?", tools=tools)

    assert parsed == [{"name": "dance", "arguments": {}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_accepts_explicit_dance_move() -> None:
    """A named dance move should pass only when explicitly present in the utterance."""
    tools = [
        {
            "name": "dance",
            "parameters": {"properties": {"move": {"enum": ["simple_nod", "robot_wave"]}}},
        }
    ]
    parsed = _parse_router_content("dance|robot_wave", user_text="please do robot wave", tools=tools)

    assert parsed == [{"name": "dance", "arguments": {"move": "robot_wave"}}]
    assert parsed.parse_error is False


def test_qwen_router_parser_marks_invalid_contract() -> None:
    """Invalid router output should be logged as a parse failure without a fallback call."""
    parsed = _parse_router_content("not a route")

    assert parsed == []
    assert parsed.parse_error is True


def test_openai_compatible_tool_router_ignores_unknown_qwen_tool() -> None:
    """Unknown Qwen-selected tools should be dropped without substituting a deterministic call."""
    class RouterResponse:
        def __enter__(self) -> "RouterResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"text": "unknown_tool|"}]}).encode("utf-8")

    def fake_urlopen(req: object, timeout: float) -> RouterResponse:
        return RouterResponse()

    router = OpenAICompatibleToolRouter(base_url="http://llama.test/v1", model="qwen3-0.6b")
    with patch("reachy_mini_conversation_app.backends.local_llm.urlopen", fake_urlopen):
        result = router._route_sync(
            "do something",
            [{"type": "function", "name": "camera", "description": "Camera", "parameters": {}}],
        )

    assert result.tool_calls == []
    assert result.ignored_tool_name == "unknown_tool"
    assert result.parse_error is False


@pytest.mark.asyncio
async def test_openai_compatible_tool_router_reports_qwen_no_tool_without_deterministic_shortcut() -> None:
    """Qwen no-tool responses should stay no-tool, even for action-like utterances."""
    diagnostics = {}
    router = OpenAICompatibleToolRouter(
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
    assert diagnostics["qwen_router_status"] == "router_no_tool"
    assert diagnostics["router_provider"] == "openai_compatible"


@pytest.mark.asyncio
async def test_openai_compatible_tool_router_reports_qwen_parse_error() -> None:
    """Malformed Qwen output should surface as a router parse error for diagnostics."""
    diagnostics = {}
    router = OpenAICompatibleToolRouter(
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
    assert diagnostics["qwen_router_status"] == "router_parse_error"
    assert diagnostics["last_local_model_error"] == "```json\n{bad"
