"""Local LLM, router, and OpenAI-compatible adapter implementations."""

from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import Any, Protocol
from dataclasses import dataclass
from urllib.request import Request, urlopen
from collections.abc import AsyncIterator

import httpx

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

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Yield chat response text deltas."""
        ...

    async def warm(self) -> None:
        """Warm latency-sensitive model state."""
        ...


class LocalToolRouter(Protocol):
    """Interface for compact local tool routing."""

    async def route(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        """Return tool calls for a user utterance, if any."""
        ...


class OpenAICompatibleLLMAdapter:
    """OpenAI-compatible chat adapter for llama.cpp/llama-server style local endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        diagnostics: Any | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible HTTP target."""
        self.base_url = (base_url or config.LOCAL_CHAT_BASE_URL).rstrip("/")
        self.model = model or config.LOCAL_CHAT_MODEL
        self.timeout_seconds = timeout_seconds
        self.diagnostics = diagnostics
        self.http_transport = http_transport

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LocalLLMResponse:
        """Return one OpenAI-compatible chat completion."""
        payload = self._chat_completion_payload(messages, stream=False)
        try:
            data = await self._post_chat_completion(payload)
        except Exception as exc:
            message = _local_http_error_message(exc)
            _record_local_model(
                self.diagnostics,
                configured_model=self.model,
                chat_provider="openai_compatible",
                last_tool_status="error",
                last_local_model_error=message,
            )
            raise RuntimeError(f"OpenAI-compatible local chat failed: {message}") from exc
        _record_local_model(
            self.diagnostics,
            configured_model=self.model,
            chat_provider="openai_compatible",
            last_tool_status="not_requested",
            last_local_model_error=None,
        )
        return _parse_openai_chat_completion_response(data)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Yield OpenAI-compatible streaming response text deltas."""
        payload = self._chat_completion_payload(messages, stream=True)
        try:
            async for chunk in self._post_chat_completion_stream(payload):
                yield chunk
        except Exception as exc:
            message = _local_http_error_message(exc)
            _record_local_model(
                self.diagnostics,
                configured_model=self.model,
                chat_provider="openai_compatible",
                last_tool_status="error",
                last_local_model_error=message,
            )
            raise RuntimeError(f"OpenAI-compatible local stream failed: {message}") from exc
        else:
            _record_local_model(
                self.diagnostics,
                configured_model=self.model,
                chat_provider="openai_compatible",
                last_tool_status="not_requested",
                last_local_model_error=None,
            )

    async def warm(self) -> None:
        """Warm the configured local chat server with a one-token request."""
        payload = self._chat_completion_payload(_warm_chat_messages(), stream=False, max_tokens=1, temperature=0)
        await self._post_chat_completion(payload)

    def _chat_completion_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        max_tokens: int | None = None,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        """Build a `/v1/chat/completions` payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_compatible_messages(messages),
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else config.LOCAL_CHAT_NUM_PREDICT,
        }
        return payload

    async def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a non-streaming OpenAI-compatible chat payload."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.http_transport) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
        if response.status_code >= 400:
            raise _HTTPResponseError(response.status_code, response.text.strip() or response.reason_phrase)
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def _post_chat_completion_stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Post a streaming OpenAI-compatible chat payload and yield content deltas."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.http_transport) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace").strip()
                    raise _HTTPResponseError(response.status_code, body or response.reason_phrase)
                async for line in response.aiter_lines():
                    chunk = _parse_openai_sse_line(line)
                    if chunk:
                        yield chunk


class OpenAICompatibleToolRouter:
    """Tiny intent router using a local OpenAI-compatible completions server."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        diagnostics: Any | None = None,
    ) -> None:
        """Initialize the router server target."""
        self.base_url = (base_url or config.LOCAL_ROUTER_BASE_URL).rstrip("/")
        self.model = model or config.LOCAL_ROUTER_MODEL
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
            _record_router_metrics(
                self.diagnostics,
                provider="openai_compatible",
                model=self.model,
                latency_ms=latency_ms,
                status="router_error",
                error=str(exc),
            )
            logger.warning("Local llama.cpp tool router failed: %s", exc)
            return LocalToolRoutingResult(tool_calls=[])
        latency_ms = (asyncio.get_running_loop().time() - started) * 1000
        _record_router_result(self.diagnostics, result, provider="openai_compatible", model=self.model, latency_ms=latency_ms)
        _log_router_result(result, latency_ms=latency_ms, user_text=user_text, provider="llama.cpp")
        return result

    async def warm(self) -> None:
        """Warm the configured router server with a one-token request."""
        await asyncio.to_thread(self._warm_sync)

    def _route_sync(self, user_text: str, tools: list[dict[str, Any]]) -> LocalToolRoutingResult:
        payload = {
            "model": self.model,
            "prompt": _router_generate_prompt(user_text, tools),
            "stream": False,
            "temperature": 0,
            "max_tokens": config.LOCAL_ROUTER_NUM_PREDICT,
            "stop": ["\n"],
        }
        data = self._post_completion(payload)
        return _router_result_from_content(_parse_openai_completion_text(data), user_text=user_text, tools=tools)

    def _warm_sync(self) -> None:
        payload = {
            "model": self.model,
            "prompt": "Return only label|arg and no punctuation.\nUser: hello\nOutput:",
            "stream": False,
            "temperature": 0,
            "max_tokens": 1,
            "stop": ["\n"],
        }
        self._post_completion(payload)

    def _post_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a compact prompt to a local OpenAI-compatible completions endpoint."""
        req = Request(
            f"{self.base_url}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else {}


def _parse_openai_chat_completion_response(data: dict[str, Any]) -> LocalLLMResponse:
    """Parse an OpenAI-compatible chat completion response."""
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return LocalLLMResponse(content="", tool_calls=[])
    return LocalLLMResponse(
        content=str(message.get("content") or "").strip(),
        tool_calls=_normalize_tool_calls(message.get("tool_calls")),
    )


def _parse_openai_completion_text(data: dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible completions response."""
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _router_result_from_content(
    content: object,
    *,
    user_text: str,
    tools: list[dict[str, Any]],
) -> LocalToolRoutingResult:
    """Parse compact router output into the public routing result type."""
    parsed = _parse_router_content(content, user_text=user_text, tools=tools)
    if parsed.ignored_tool_name:
        return LocalToolRoutingResult(
            tool_calls=[],
            raw_content=str(content or ""),
            parse_error=False,
            ignored_tool_name=parsed.ignored_tool_name,
        )
    for call in parsed:
        return LocalToolRoutingResult(
            tool_calls=[call],
            raw_content=str(content or ""),
            parse_error=False,
        )
    return LocalToolRoutingResult(
        tool_calls=[],
        raw_content=str(content or ""),
        parse_error=getattr(parsed, "parse_error", False),
    )


def _warm_chat_messages() -> list[dict[str, str]]:
    """Return warmup messages that cache the active profile prompt when available."""
    try:
        from reachy_mini_conversation_app.profiles.prompts import get_session_instructions

        return [
            {"role": "system", "content": get_session_instructions()},
            {"role": "user", "content": "."},
        ]
    except Exception:
        logger.debug("Falling back to minimal local chat warmup.", exc_info=True)
        return [{"role": "user", "content": "."}]


def _parse_openai_sse_line(line: str) -> str:
    """Return one OpenAI-compatible SSE content delta, if present."""
    text = line.strip()
    if not text or text.startswith(":"):
        return ""
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return ""
    data = json.loads(text)
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    delta = choice.get("delta") if isinstance(choice, dict) else None
    if isinstance(delta, dict):
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    return ""


def _openai_compatible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return text-only messages acceptable to OpenAI-compatible local servers."""
    converted: list[dict[str, Any]] = []
    pending_context: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "")
        if not content:
            continue
        if role == "system" and not converted:
            converted.append({"role": "system", "content": content})
            continue
        if role in {"system", "tool"}:
            pending_context.append(content)
            continue
        if role == "assistant":
            if _last_chat_role(converted) == "user":
                converted.append({"role": "assistant", "content": content})
            elif _last_chat_role(converted) == "assistant":
                converted[-1]["content"] = f"{converted[-1]['content']}\n\n{content}"
            continue

        user_content = content
        if pending_context:
            user_content = "Context:\n" + "\n".join(pending_context) + f"\n\nUser:\n{content}"
            pending_context = []
        if _last_chat_role(converted) == "user":
            converted[-1]["content"] = f"{converted[-1]['content']}\n\n{user_content}"
        else:
            converted.append({"role": "user", "content": user_content})
    if pending_context:
        context_text = "Context:\n" + "\n".join(pending_context)
        if _last_chat_role(converted) == "user":
            converted[-1]["content"] = f"{converted[-1]['content']}\n\n{context_text}"
        else:
            converted.append({"role": "user", "content": context_text})
    return converted


def _last_chat_role(messages: list[dict[str, Any]]) -> str | None:
    """Return the latest non-system chat role."""
    for message in reversed(messages):
        role = message.get("role")
        if role != "system":
            return str(role)
    return None


class _HTTPResponseError(RuntimeError):
    """Small status/body wrapper for async HTTP paths."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _local_http_error_message(exc: Exception) -> str:
    """Return a concise local HTTP failure message."""
    if isinstance(exc, _HTTPResponseError):
        return f"HTTP {exc.status_code}: {exc.body}"
    if isinstance(exc, httpx.HTTPError):
        return str(exc)
    return str(exc)


def create_local_llm_adapter(*, diagnostics: Any | None = None) -> LocalLLMAdapter:
    """Create the configured local chat adapter."""
    logger.info("Using OpenAI-compatible local chat provider at %s.", config.LOCAL_CHAT_BASE_URL)
    return OpenAICompatibleLLMAdapter(diagnostics=diagnostics)


def create_local_tool_router(*, diagnostics: Any | None = None) -> LocalToolRouter:
    """Create the configured compact local tool router."""
    logger.info("Using OpenAI-compatible local tool router at %s.", config.LOCAL_ROUTER_BASE_URL)
    return OpenAICompatibleToolRouter(diagnostics=diagnostics)


class _RouterCalls(list[dict[str, Any]]):
    """Router calls with parse metadata attached."""

    def __init__(
        self,
        calls: list[dict[str, Any]],
        *,
        parse_error: bool = False,
        ignored_tool_name: str | None = None,
    ) -> None:
        """Initialize normalized calls with parse status."""
        super().__init__(calls)
        self.parse_error = parse_error
        self.ignored_tool_name = ignored_tool_name


def _parse_router_content(
    content: object,
    *,
    user_text: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> _RouterCalls:
    """Parse and locally validate the router's `label|arg` contract."""
    text = str(content or "").strip().splitlines()[0].strip() if str(content or "").strip() else ""
    if "|" not in text and _clean_router_label(text).startswith("none"):
        return _RouterCalls([])
    if "|" not in text:
        label = _clean_router_label(text)
        arg = ""
    else:
        label, arg = text.split("|", 1)
        label = _clean_router_label(label)
        arg = _clean_router_arg(arg)
        if label == "none":
            return _RouterCalls([])

    active_tools = {str(tool.get("name") or ""): tool for tool in tools or [] if isinstance(tool.get("name"), str)}
    if active_tools and label not in active_tools:
        return _RouterCalls([], ignored_tool_name=label or None)

    arguments = _router_arguments_for(label, arg, user_text, active_tools.get(label))
    if arguments is None:
        return _RouterCalls([], parse_error="|" not in text, ignored_tool_name=label or None)
    return _RouterCalls([{"name": label, "arguments": arguments}])


def _clean_router_label(label: str) -> str:
    """Return a lowercase router label with common model cruft removed."""
    cleaned = label.strip().strip("`\"'").casefold()
    return re.sub(r"[^a-z0-9_]+", "", cleaned)


def _clean_router_arg(arg: str) -> str:
    """Return a compact router argument with model punctuation cruft removed."""
    cleaned = arg.strip().strip("`\"'").strip()
    return re.sub(r"[\s,.;:]+$", "", cleaned)


def _router_generate_prompt(user_text: str, tools: list[dict[str, Any]]) -> str:
    """Return the tiny `label|arg` router prompt without tool schemas or enums."""
    active = ", ".join(["none", *_active_tool_names(tools)])
    return (
        "Return only label|arg and no punctuation. Use none| for ordinary chat. "
        f"Labels: {active}. "
        "Examples: hello=>none|. how are you=>none|. what do you see=>camera|question. "
        "camera/see=>camera|question. move your head left=>move_head|left. "
        "move head=>move_head|left/right/up/down/front. dance=>dance|. emotion=>play_emotion|. "
        "who am i=>who_am_i|. who is here=>who_is_here|. remember me as Alice=>remember_person|Alice. "
        "look at Alice=>look_at_person|Alice. head tracking on=>head_tracking|start. "
        "head tracking off=>head_tracking|stop. stop dance=>stop_dance|. stop emotion=>stop_emotion|. "
        "task status=>task_status|. cancel task abc=>task_cancel|abc. "
        f"User: {user_text} Output:"
    )


def _active_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """Return active tool names only, avoiding schema/enum payload bloat."""
    names: list[str] = []
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _router_arguments_for(
    label: str,
    arg: str,
    user_text: str,
    tool: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate one routed label/arg pair and return dispatch arguments."""
    if label in {
        "who_am_i",
        "who_is_here",
        "task_status",
    }:
        return {}
    if label == "camera":
        return {"question": user_text if not arg or arg.casefold() == "question" else arg}
    if label == "move_head":
        direction = arg.casefold()
        return {"direction": direction} if direction in {"left", "right", "up", "down", "front"} else None
    if label == "dance":
        return _optional_exact_enum_argument("move", arg, user_text, tool)
    if label == "play_emotion":
        return _optional_exact_enum_argument("emotion", arg, user_text, tool)
    if label == "look_at_person":
        return {"name": arg} if arg else None
    if label == "remember_person":
        return {"name": arg} if arg else None
    if label == "head_tracking":
        lowered = arg.casefold()
        if lowered in {"start", "on", "enable", "enabled", "true"}:
            return {"start": True}
        if lowered in {"stop", "off", "disable", "disabled", "false"}:
            return {"start": False}
        return None
    if label in {"stop_dance", "stop_emotion"}:
        return {"dummy": True}
    if label == "task_cancel":
        return {"tool_id": arg} if arg else None
    return None


def _optional_exact_enum_argument(
    argument_name: str,
    arg: str,
    user_text: str,
    tool: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an optional enum argument only when the user explicitly said it."""
    if not arg or arg.casefold() in {"none", "null", "random", "any"}:
        return {}
    enum_values = _enum_values_for_argument(tool, argument_name)
    if not enum_values:
        return None
    exact_value = next((value for value in enum_values if arg == value), None)
    if exact_value is None:
        return None
    if _utterance_contains_exact_value(user_text, exact_value):
        return {argument_name: exact_value}
    return {}


def _enum_values_for_argument(tool: dict[str, Any] | None, argument_name: str) -> list[str]:
    """Return local enum values for validation without sending them to Qwen."""
    if not isinstance(tool, dict):
        return []
    parameters = tool.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    argument = properties.get(argument_name) if isinstance(properties, dict) else None
    enum_values = argument.get("enum") if isinstance(argument, dict) else None
    return [str(value) for value in enum_values] if isinstance(enum_values, list) else []


def _utterance_contains_exact_value(user_text: str, value: str) -> bool:
    """Return whether a user explicitly said an enum value."""
    folded_text = user_text.casefold()
    folded_value = value.casefold()
    if not folded_value:
        return False
    if re.search(rf"(?<!\w){re.escape(folded_value)}(?!\w)", folded_text):
        return True
    spaced = folded_value.replace("_", " ").replace("-", " ")
    return bool(spaced != folded_value and re.search(rf"(?<!\w){re.escape(spaced)}(?!\w)", folded_text))


def _truncate(value: str, limit: int) -> str:
    """Return a compact single-line text value."""
    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."


def _record_local_model(diagnostics: Any | None, **payload: object) -> None:
    """Best-effort local model diagnostics update."""
    set_local_model = getattr(diagnostics, "set_local_model", None)
    if callable(set_local_model):
        set_local_model(**payload)


def _record_router_result(
    diagnostics: Any | None,
    result: LocalToolRoutingResult,
    *,
    provider: str,
    model: str,
    latency_ms: float,
) -> None:
    """Record router diagnostics for the llama.cpp provider."""
    if result.tool_calls:
        status = "router_ok"
    elif result.parse_error:
        status = "router_parse_error"
    else:
        status = "router_no_tool"
    _record_router_metrics(
        diagnostics,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        status=status,
        error=result.raw_content if result.parse_error else None,
    )


def _record_router_metrics(
    diagnostics: Any | None,
    *,
    provider: str,
    model: str,
    latency_ms: float,
    status: str,
    error: str | None,
) -> None:
    """Record compact-router metrics while preserving dashboard field names."""
    _record_local_model(
        diagnostics,
        router_provider=provider,
        router_model=model,
        router_latency_ms=latency_ms,
        router_status=status,
        qwen_router_latency_ms=latency_ms,
        qwen_router_status=status,
        last_local_model_error=error,
    )


def _log_router_result(
    result: LocalToolRoutingResult,
    *,
    latency_ms: float,
    user_text: str,
    provider: str,
) -> None:
    """Log one compact-router decision."""
    if result.tool_calls:
        logger.info(
            "%s router selected tool=%s args=%s latency=%.0fms utterance=%r",
            provider,
            result.tool_calls[0].get("name"),
            result.tool_calls[0].get("arguments"),
            latency_ms,
            _truncate(user_text, 160),
        )
    elif result.parse_error:
        logger.info(
            "%s router parse failed latency=%.0fms utterance=%r raw=%r",
            provider,
            latency_ms,
            _truncate(user_text, 160),
            _truncate(result.raw_content, 500),
        )
    elif result.ignored_tool_name:
        logger.info(
            "%s router ignored unavailable tool=%s latency=%.0fms utterance=%r",
            provider,
            result.ignored_tool_name,
            latency_ms,
            _truncate(user_text, 160),
        )
    else:
        logger.info("%s router selected no tool latency=%.0fms utterance=%r", provider, latency_ms, _truncate(user_text, 160))


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
    """Normalize OpenAI-ish tool calls into name/arguments dictionaries."""
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
