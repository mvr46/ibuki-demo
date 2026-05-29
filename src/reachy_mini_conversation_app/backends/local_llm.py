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
    raw_content: str = ""
    parse_error: bool = False
    ignored_tool_name: str | None = None


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
                qwen_router_status="qwen_error",
                last_ollama_error=str(exc),
            )
            logger.warning("Local Qwen tool router failed: %s", exc)
            return LocalToolRoutingResult(tool_calls=[])
        if result.tool_calls:
            status = "qwen_ok"
        elif result.parse_error:
            status = "qwen_parse_error"
        else:
            status = "qwen_no_tool"
        latency_ms = (asyncio.get_running_loop().time() - started) * 1000
        _record_local_model(
            self.diagnostics,
            router_model=self.model,
            qwen_router_latency_ms=latency_ms,
            qwen_router_status=status,
            last_ollama_error=result.raw_content if result.parse_error else None,
        )
        if result.tool_calls:
            logger.info(
                "Qwen router selected tool=%s args=%s latency=%.0fms utterance=%r",
                result.tool_calls[0].get("name"),
                result.tool_calls[0].get("arguments"),
                latency_ms,
                _truncate(user_text, 160),
            )
        elif result.parse_error:
            logger.info(
                "Qwen router parse failed latency=%.0fms utterance=%r raw=%r",
                latency_ms,
                _truncate(user_text, 160),
                _truncate(result.raw_content, 500),
            )
        elif result.ignored_tool_name:
            logger.info(
                "Qwen router ignored unavailable tool=%s latency=%.0fms utterance=%r",
                result.ignored_tool_name,
                latency_ms,
                _truncate(user_text, 160),
            )
        else:
            logger.info("Qwen router selected no tool latency=%.0fms utterance=%r", latency_ms, _truncate(user_text, 160))
        return result

    def _route_sync(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        tool_names = {str(tool.get("name") or "") for tool in tools if isinstance(tool.get("name"), str)}
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": _router_output_schema(),
            "options": {
                "num_predict": 48,
                "temperature": 0,
            },
            "messages": [
                {
                    "role": "system",
                    "content": _router_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "utterance": user_text,
                            "tools": _compact_tool_specs(tools),
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
                return LocalToolRoutingResult(
                    tool_calls=[],
                    raw_content=str(content or ""),
                    parse_error=False,
                    ignored_tool_name=name or None,
                )
            arguments = call.get("arguments")
            normalized.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
            break
        return LocalToolRoutingResult(
            tool_calls=normalized,
            raw_content=str(content or ""),
            parse_error=getattr(parsed, "parse_error", False),
        )

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


class _RouterCalls(list[dict[str, Any]]):
    """Router calls with parse metadata attached."""

    def __init__(self, calls: list[dict[str, Any]], *, parse_error: bool = False) -> None:
        """Initialize normalized calls with parse status."""
        super().__init__(calls)
        self.parse_error = parse_error


def _parse_router_content(content: object) -> _RouterCalls:
    """Parse router JSON content into normalized call dictionaries."""
    if isinstance(content, dict):
        parsed = content
    else:
        try:
            parsed = json.loads(_strip_json_code_fence(str(content or "{}")))
        except json.JSONDecodeError:
            return _RouterCalls([], parse_error=True)
    raw_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(raw_calls, list):
        return _RouterCalls([], parse_error=True)
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("parameters")
        calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return _RouterCalls(calls)


def _strip_json_code_fence(content: str) -> str:
    """Return raw JSON from model output that may be wrapped in Markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _router_system_prompt() -> str:
    """Return the strict local Qwen tool-router instruction."""
    return (
        "You route one user utterance to Reachy Mini robot tools. "
        "Output RAW JSON only, no markdown, no code fences, no commentary. "
        'The JSON object must match: {"tool_calls":[{"name":"tool_name","arguments":{}}]}. '
        'Use the key "arguments" exactly; never use "parameters". '
        "Call at most one tool. Use [] for ordinary conversation. "
        'Use who_am_i for questions asking the speaker identity, such as "who am I". '
        "Use who_is_here for questions asking which people are visible. "
        'Use camera for visual scene/object questions like "what do you see" or "take a picture". '
        "Use remember_person when the user introduces or names themself and asks to be remembered. "
        "Use task_status when the user asks about running tools or background work. "
        "Use movement, dance, emotion, and tracking tools only for explicit robot action requests."
    )


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
                "description": _truncate(str(tool.get("description") or ""), 180),
                "arguments": _argument_hint(tool.get("parameters")),
            }
        )
    return compact


def _argument_hint(schema: object) -> dict[str, Any] | str:
    """Return a compact argument description without the confusing schema key name."""
    if not isinstance(schema, dict):
        return "none"
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "none"
    compact: dict[str, Any] = {}
    if isinstance(properties, dict):
        for key, value in properties.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            item: dict[str, Any] = {"type": value.get("type", "string")}
            if "enum" in value and isinstance(value["enum"], list):
                item["enum"] = value["enum"][:20]
            compact[key] = item
    required = schema.get("required")
    if isinstance(required, list) and required:
        compact["_required"] = [item for item in required if isinstance(item, str)]
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
