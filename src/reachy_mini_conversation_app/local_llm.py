"""Local Ollama LLM adapter."""

from __future__ import annotations
import json
import asyncio
import logging
from typing import Any, Protocol
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalLLMResponse:
    """One local model response."""

    content: str
    tool_calls: list[dict[str, Any]]


class LocalLLMAdapter(Protocol):
    """Interface for local chat models."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LocalLLMResponse:
        """Return a chat response and any tool calls."""
        ...


class OllamaLLMAdapter:
    """Ollama `/api/chat` adapter with OpenAI-style tool schema input."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialize the Ollama HTTP target."""
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LocalLLMResponse:
        """Return one Ollama chat response."""
        return await asyncio.to_thread(self._chat_sync, messages, tools)

    def _chat_sync(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LocalLLMResponse:
        tool_schemas = ollama_tool_schemas(tools)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
        try:
            data = self._post_chat(payload)
        except HTTPError as exc:
            body = _read_http_error_body(exc)
            if exc.code == 400 and tool_schemas:
                logger.warning("Ollama rejected tool schema (%s); retrying without tools. Body: %s", exc.code, body)
                payload.pop("tools", None)
                data = self._post_chat(payload)
            else:
                raise RuntimeError(f"Ollama chat failed with HTTP {exc.code}: {body}") from exc

        return _parse_chat_response(data)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a chat payload to Ollama."""
        req = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else {}


def _parse_chat_response(data: dict[str, Any]) -> LocalLLMResponse:
    """Parse an Ollama chat response."""
    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict):
        return LocalLLMResponse(content="", tool_calls=[])

    return LocalLLMResponse(
        content=str(message.get("content") or "").strip(),
        tool_calls=_normalize_tool_calls(message.get("tool_calls")),
    )


def _read_http_error_body(exc: HTTPError) -> str:
    """Read an HTTP error body without letting logging raise another exception."""
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


class EchoLLMAdapter:
    """Testing/local fallback adapter."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LocalLLMResponse:
        """Echo the latest user message."""
        latest = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        return LocalLLMResponse(content=f"I heard: {latest.get('content', '')}", tool_calls=[])


def _normalize_tool_calls(raw_tool_calls: object) -> list[dict[str, Any]]:
    """Normalize Ollama/OpenAI-ish tool calls into name/arguments dictionaries."""
    if not isinstance(raw_tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            arguments = fn.get("arguments") or {}
        else:
            name = item.get("name")
            arguments = item.get("arguments") or {}
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        normalized.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return normalized


def ollama_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert local/OpenAI-ish tool specs to Ollama chat tool schemas."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {},
                },
            }
        )
    return converted


def ollama_tool_call_messages(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized tool calls back to Ollama chat history shape."""
    messages: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        messages.append(
            {
                "type": "function",
                "function": {
                    "index": index,
                    "name": name,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                },
            }
        )
    return messages
