"""Local Ollama LLM adapter."""

from __future__ import annotations
import json
import asyncio
import logging
from typing import Any, Protocol
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from reachy_mini_conversation_app.runtime.config import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalLLMResponse:
    """One local model response."""

    content: str
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True)
class LocalToolRoutingResult:
    """One compact local tool-routing decision."""

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


class LocalToolRouter(Protocol):
    """Interface for compact local tool routing."""

    async def route(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        """Return tool calls for a user utterance, if any."""
        ...


class OllamaLLMAdapter:
    """Ollama `/api/chat` adapter with OpenAI-style tool schema input."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        diagnostics: Any | None = None,
    ) -> None:
        """Initialize the Ollama HTTP target."""
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds
        self.diagnostics = diagnostics

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
            "think": False,
            "options": {
                "num_predict": 192,
                "temperature": 0.4,
            },
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
        try:
            data = self._post_chat(payload)
        except HTTPError as exc:
            body = _read_http_error_body(exc)
            if exc.code == 400 and tool_schemas:
                _record_local_model(
                    self.diagnostics,
                    configured_model=self.model,
                    last_tool_status="rejected",
                    tools_disabled_by_model=True,
                    last_ollama_error=body,
                )
                logger.warning("Ollama rejected tool schema (%s); retrying without tools. Body: %s", exc.code, body)
                payload.pop("tools", None)
                data = self._post_chat(payload)
            else:
                _record_local_model(
                    self.diagnostics,
                    configured_model=self.model,
                    last_tool_status="error",
                    last_ollama_error=body,
                )
                raise RuntimeError(f"Ollama chat failed with HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            message = str(exc)
            _record_local_model(
                self.diagnostics,
                configured_model=self.model,
                last_tool_status="error",
                last_ollama_error=message,
            )
            raise RuntimeError(f"Ollama chat failed: {message}") from exc
        else:
            _record_local_model(
                self.diagnostics,
                configured_model=self.model,
                last_tool_status="ok" if tool_schemas else "not_requested",
                tools_disabled_by_model=False,
                last_ollama_error=None,
            )

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


class OllamaToolRouter:
    """Compact Qwen-backed JSON tool router for local robot actions."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        diagnostics: Any | None = None,
    ) -> None:
        """Initialize the router Ollama target."""
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_ROUTER_MODEL
        self.timeout_seconds = timeout_seconds
        self.diagnostics = diagnostics

    async def route(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        """Return a single compact tool call decision."""
        if not tools:
            return LocalToolRoutingResult(tool_calls=[])
        started = asyncio.get_running_loop().time()
        try:
            result = await asyncio.to_thread(self._route_sync, user_text, tools)
        except Exception as exc:
            latency_ms = (asyncio.get_running_loop().time() - started) * 1000
            _record_local_model(
                self.diagnostics,
                router_model=self.model,
                qwen_router_latency_ms=latency_ms,
                qwen_router_status="error",
                last_ollama_error=str(exc),
            )
            logger.warning("Local Qwen tool router failed: %s", exc)
            return LocalToolRoutingResult(tool_calls=[])
        latency_ms = (asyncio.get_running_loop().time() - started) * 1000
        _record_local_model(
            self.diagnostics,
            router_model=self.model,
            qwen_router_latency_ms=latency_ms,
            qwen_router_status="ok" if result.tool_calls else "no_tool",
            last_ollama_error=None,
        )
        return result

    def _route_sync(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        tool_names = {str(tool.get("name") or "") for tool in tools if isinstance(tool.get("name"), str)}
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": _router_output_schema(),
            "options": {
                "num_predict": 64,
                "temperature": 0,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fast JSON router for Reachy Mini robot tools. "
                        "Return exactly one JSON object matching the schema. "
                        "Call at most one tool. Use a tool only when the user asks for a robot action, "
                        "camera/vision lookup, face/person lookup, memory, dance, emotion, movement, or task status. "
                        "For ordinary conversation return an empty tool_calls array."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "utterance": user_text,
                            "available_tools": _compact_tool_specs(tools),
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        }
        data = self._post_chat(payload)
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else ""
        parsed = _parse_router_content(content)
        normalized = []
        for call in parsed:
            name = str(call.get("name") or "")
            if name not in tool_names:
                continue
            arguments = call.get("arguments")
            normalized.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
            break
        return LocalToolRoutingResult(tool_calls=normalized)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a router payload to Ollama."""
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


def _parse_router_content(content: object) -> list[dict[str, Any]]:
    """Parse router JSON content into normalized call dictionaries."""
    if isinstance(content, dict):
        parsed = content
    else:
        try:
            parsed = json.loads(str(content or "{}"))
        except json.JSONDecodeError:
            return []
    raw_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if isinstance(item, dict):
            calls.append(item)
    return calls


def _router_output_schema() -> dict[str, Any]:
    """Return Ollama structured-output schema for the tool router."""
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tool_calls"],
        "additionalProperties": False,
    }


def _compact_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool descriptions for the router prompt."""
    compact: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        compact.append(
            {
                "name": name,
                "description": _truncate(str(tool.get("description") or ""), 220),
                "parameters": _compact_schema(tool.get("parameters")),
            }
        )
    return compact


def _compact_schema(schema: object) -> dict[str, Any]:
    """Trim verbose JSON schema text while preserving argument names and requirements."""
    if not isinstance(schema, dict):
        return {}
    compact: dict[str, Any] = {"type": schema.get("type", "object")}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        compact_properties: dict[str, Any] = {}
        for key, value in properties.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            item: dict[str, Any] = {"type": value.get("type", "string")}
            if "enum" in value and isinstance(value["enum"], list):
                item["enum"] = value["enum"][:20]
            if value.get("description"):
                item["description"] = _truncate(str(value["description"]), 160)
            compact_properties[key] = item
        compact["properties"] = compact_properties
    if isinstance(schema.get("required"), list):
        compact["required"] = schema["required"]
    return compact


def _truncate(value: str, limit: int) -> str:
    """Return a compact single-line text value."""
    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."


def _read_http_error_body(exc: HTTPError) -> str:
    """Read an HTTP error body without letting logging raise another exception."""
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _record_local_model(diagnostics: Any | None, **payload: object) -> None:
    """Best-effort local model diagnostics update."""
    set_local_model = getattr(diagnostics, "set_local_model", None)
    if callable(set_local_model):
        set_local_model(**payload)


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
