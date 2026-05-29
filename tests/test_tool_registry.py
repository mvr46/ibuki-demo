from pathlib import Path

import pytest

from reachy_mini_conversation_app.profiles.store import ProfileStore
from reachy_mini_conversation_app.runtime.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolRegistry, ToolDependencies, list_core_tool_names


def _store(root: Path, tools_text: str) -> ProfileStore:
    store = ProfileStore(root)
    store.save_new("default", instructions="[default_prompt]", tools_text=tools_text, voice="local")
    return store


def test_tool_registry_loads_only_profile_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ToolRegistry should load the active profile's core tools plus system tools."""
    store = _store(tmp_path, "dance\ncamera\n")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    registry = ToolRegistry.from_active_profile(store)

    assert "dance" in registry.tools
    assert "camera" in registry.tools
    assert "task_status" in registry.tools
    assert "move_head" not in registry.tools


def test_tool_registry_rejects_unknown_profile_tools(tmp_path: Path) -> None:
    """Unknown tools should fail at the registry seam instead of being silently ignored."""
    store = _store(tmp_path, "not_a_real_tool\n")

    with pytest.raises(RuntimeError, match="Unknown core tool"):
        ToolRegistry.from_active_profile(store)


def test_active_tool_specs_filter_unavailable_dependencies(tmp_path: Path) -> None:
    """Active specs should hide tools that need absent camera or face deps."""
    registry = ToolRegistry.from_active_profile(_store(tmp_path, "camera\nhead_tracking\nwho_is_here\n"))
    deps = ToolDependencies(reachy_mini=object(), movement_manager=object(), tool_registry=registry)

    names = {spec["name"] for spec in registry.get_active_tool_specs(deps)}

    assert "camera" in names
    assert "head_tracking" not in names
    assert "who_is_here" not in names


def test_core_tool_names_exclude_registry_internals() -> None:
    """Dashboard tool checkboxes should show model-callable tools only."""
    names = list_core_tool_names()

    assert "dance" in names
    assert "core_tools" not in names
    assert "background_tool_manager" not in names
    assert "task_status" not in names
