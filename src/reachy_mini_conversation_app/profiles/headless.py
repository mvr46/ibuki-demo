"""Dashboard-facing production profile helpers."""

from __future__ import annotations
from typing import List
from pathlib import Path

from reachy_mini_conversation_app.profiles.store import DEFAULT_PROFILE_NAME, ProfileStore
from reachy_mini_conversation_app.runtime.config import get_default_voice_for_backend


DEFAULT_OPTION = DEFAULT_PROFILE_NAME
_store = ProfileStore()


def _profiles_root() -> Path:
    """Return the production profile root."""
    return _store.root


def _sanitize_name(name: str) -> str:
    """Return a profile-safe name."""
    return ProfileStore.sanitize_name(name)


def list_personalities() -> List[str]:
    """List editable production profile names."""
    return [summary.name for summary in _store.list_profiles()]


def resolve_profile_dir(selection: str) -> Path:
    """Resolve the directory path for the given profile selection."""
    return _store.profile_dir(selection or DEFAULT_PROFILE_NAME)


def read_instructions_for(name: str) -> str:
    """Read profile instructions."""
    return _store.load(name or DEFAULT_PROFILE_NAME).instructions


def read_tools_for(name: str) -> str:
    """Read profile tools as tools.txt content."""
    return _store.load(name or DEFAULT_PROFILE_NAME).tools_text


def available_tools_for(_selected: str = DEFAULT_PROFILE_NAME) -> List[str]:
    """List available core tool modules."""
    return ProfileStore.parse_tools_text(_core_tool_names_text())


def _core_tool_names_text() -> str:
    """Return core tool module names as newline-delimited text."""
    from reachy_mini_conversation_app.tools.core_tools import list_core_tool_names

    return "\n".join(list_core_tool_names())


def _write_profile(name_s: str, instructions: str, tools_text: str, voice: str | None = None) -> None:
    """Create or overwrite a production profile."""
    profile_name = _sanitize_name(name_s)
    profile_voice = voice or get_default_voice_for_backend()
    if _store.profile_dir(profile_name).exists():
        _store.overwrite(profile_name, instructions=instructions, tools_text=tools_text, voice=profile_voice)
    else:
        _store.save_new(profile_name, instructions=instructions, tools_text=tools_text, voice=profile_voice)
