from __future__ import annotations
import abc
import asyncio
import inspect
import logging
import importlib
from typing import TYPE_CHECKING, Any, Dict
from pathlib import Path
from dataclasses import dataclass

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.profiles.store import ProfileStore
from reachy_mini_conversation_app.runtime.config import config
from reachy_mini_conversation_app.tools.tool_constants import SystemTool


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any
    camera_worker: Any | None = None
    face_identity_worker: Any | None = None
    spatial_audio_source: Any | None = None
    speaker_attribution_worker: Any | None = None
    vision_processor: Any | None = None
    vision_analyzer: Any | None = None
    head_wobbler: Any | None = None
    performance_diagnostics: Any | None = None
    tool_registry: "ToolRegistry | None" = None
    motion_duration_s: float = 1.0


class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


def get_concrete_subclasses(base: type[Tool]) -> list[type[Tool]]:
    """Recursively find all concrete subclasses of a base class."""
    result: list[type[Tool]] = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        result.extend(get_concrete_subclasses(cls))
    return result


def list_core_tool_names() -> list[str]:
    """List model-selectable core tool module names."""
    tools_dir = Path(__file__).parent
    ignored = {
        "__init__",
        "core_tools",
        "background_tool_manager",
        "tool_constants",
        *{tool.value for tool in SystemTool},
    }
    return sorted(path.stem for path in tools_dir.glob("*.py") if path.stem not in ignored)


class ToolRegistry:
    """Explicit registry for the active profile's core tool Adapters."""

    def __init__(self, tool_names: list[str] | tuple[str, ...]) -> None:
        """Load the requested tool modules and instantiate their Tool classes."""
        deduped: list[str] = []
        for name in tool_names:
            if name not in deduped:
                deduped.append(name)
        self.requested_tool_names = tuple(deduped)
        self.tools = self._load_tools(self.requested_tool_names)
        self.tool_specs = [tool.spec() for tool in self.tools.values()]

    @classmethod
    def from_active_profile(cls, profile_store: ProfileStore | None = None) -> "ToolRegistry":
        """Build a registry from the current production profile."""
        store = profile_store or ProfileStore()
        profile = store.load(config.REACHY_MINI_CUSTOM_PROFILE)
        tool_names = [*profile.tools, *[tool.value for tool in SystemTool]]
        return cls(tool_names)

    def get_tool_specs(self, exclusion_list: list[str] | None = None) -> list[Dict[str, Any]]:
        """Return tool specs, optionally excluding unavailable tools."""
        excluded = set(exclusion_list or [])
        return [spec for spec in self.tool_specs if spec.get("name") not in excluded]

    def get_active_tool_specs(self, deps: ToolDependencies) -> list[Dict[str, Any]]:
        """Return tool specs filtered by what the current session deps support."""
        exclusion_list: list[str] = []
        if not (deps.camera_worker and deps.camera_worker.head_tracker):
            exclusion_list.append("head_tracking")
        if deps.face_identity_worker is None:
            exclusion_list.extend(["who_is_here", "remember_person", "look_at_person"])
        elif not bool(getattr(deps.face_identity_worker, "recognition_available", True)):
            exclusion_list.extend(["remember_person", "look_at_person"])
        if deps.camera_worker is None:
            exclusion_list.append("look_at_person")
        return self.get_tool_specs(exclusion_list)

    async def dispatch_tool_call(self, tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
        """Dispatch a tool call by name with JSON args and dependencies."""
        return await self._dispatch_tool_call(tool_name, _safe_load_obj(args_json), deps)

    async def dispatch_tool_call_with_manager(
        self,
        tool_name: str,
        args_json: str,
        deps: ToolDependencies,
        tool_manager: "BackgroundToolManager",
    ) -> Dict[str, Any]:
        """Dispatch a tool call, injecting a BackgroundToolManager into the args."""
        args = _safe_load_obj(args_json)
        args["tool_manager"] = tool_manager
        return await self._dispatch_tool_call(tool_name, args, deps)

    async def _dispatch_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"unknown tool: {tool_name}"}
        try:
            return await tool(deps, **args)
        except asyncio.CancelledError:
            logger.info("Tool cancelled: %s", tool_name)
            return {"error": "Tool cancelled"}
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            logger.exception("Tool error in %s: %s", tool_name, msg)
            return {"error": msg}

    def _load_tools(self, tool_names: tuple[str, ...]) -> dict[str, Tool]:
        """Import and instantiate only the requested core tool modules."""
        for tool_name in tool_names:
            try:
                importlib.import_module(f"reachy_mini_conversation_app.tools.{tool_name}")
            except ModuleNotFoundError as exc:
                raise RuntimeError(f"Unknown core tool(s) in active profile: {[tool_name]}") from exc
        tools_by_name = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}  # type: ignore[type-abstract]
        missing = [tool_name for tool_name in tool_names if tool_name not in tools_by_name]
        if missing:
            raise RuntimeError(f"Unknown core tool(s) in active profile: {missing}")
        loaded = {tool_name: tools_by_name[tool_name] for tool_name in tool_names}
        for tool_name, tool in loaded.items():
            logger.info("tool registered: %s - %s", tool_name, tool.description)
        return loaded


_DEFAULT_REGISTRY: ToolRegistry | None = None
ALL_TOOLS: Dict[str, Tool] = {}
ALL_TOOL_SPECS: list[Dict[str, Any]] = []
_TOOLS_INITIALIZED = False


def default_tool_registry() -> ToolRegistry:
    """Return a lazily-created registry for compatibility callers."""
    global ALL_TOOLS, ALL_TOOL_SPECS, _DEFAULT_REGISTRY, _TOOLS_INITIALIZED
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry.from_active_profile()
        ALL_TOOLS = _DEFAULT_REGISTRY.tools
        ALL_TOOL_SPECS = _DEFAULT_REGISTRY.tool_specs
        _TOOLS_INITIALIZED = True
    return _DEFAULT_REGISTRY


def get_tool_specs(exclusion_list: list[str] | None = None) -> list[Dict[str, Any]]:
    """Get tool specs from the active registry."""
    return default_tool_registry().get_tool_specs(exclusion_list)


def get_active_tool_specs(deps: ToolDependencies) -> list[Dict[str, Any]]:
    """Get active tool specs from the explicit registry when available."""
    registry = deps.tool_registry or default_tool_registry()
    return registry.get_active_tool_specs(deps)


def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    import json

    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    registry = deps.tool_registry or default_tool_registry()
    return await registry.dispatch_tool_call(tool_name, args_json, deps)


async def dispatch_tool_call_with_manager(
    tool_name: str, args_json: str, deps: ToolDependencies, tool_manager: "BackgroundToolManager"
) -> Dict[str, Any]:
    """Dispatch a tool call, injecting a BackgroundToolManager into the args."""
    registry = deps.tool_registry or default_tool_registry()
    return await registry.dispatch_tool_call_with_manager(tool_name, args_json, deps, tool_manager)
