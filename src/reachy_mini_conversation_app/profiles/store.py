"""Repo-backed production profile storage."""

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from importlib.resources import files


DEFAULT_PROFILE_NAME = "default"
INSTRUCTIONS_FILENAME = "instructions.txt"
TOOLS_FILENAME = "tools.txt"
VOICE_FILENAME = "voice.txt"


@dataclass(frozen=True)
class ProfileSummary:
    """One profile available to the production dashboard."""

    name: str
    is_default: bool = False


@dataclass(frozen=True)
class ProductionProfile:
    """Editable production prompt, tool policy, and voice preference."""

    name: str
    instructions: str
    tools: tuple[str, ...]
    voice: str | None = None

    @property
    def tools_text(self) -> str:
        """Return the persisted tools file content."""
        return "\n".join(self.tools) + ("\n" if self.tools else "")


def default_profiles_root() -> Path:
    """Return the source/package data directory containing production profiles."""
    return Path(str(files("reachy_mini_conversation_app.profiles")))


class ProfileStore:
    """Small Interface for production profile persistence."""

    def __init__(self, root: Path | None = None) -> None:
        """Initialize a store rooted at the package profile directory."""
        self.root = root or default_profiles_root()

    def list_profiles(self) -> list[ProfileSummary]:
        """List valid profiles, with the default profile first."""
        summaries: list[ProfileSummary] = []
        if not self.root.exists():
            return summaries
        for profile_dir in sorted(self.root.iterdir(), key=lambda path: (path.name != DEFAULT_PROFILE_NAME, path.name)):
            if profile_dir.is_dir() and (profile_dir / INSTRUCTIONS_FILENAME).is_file():
                summaries.append(
                    ProfileSummary(
                        name=profile_dir.name,
                        is_default=profile_dir.name == DEFAULT_PROFILE_NAME,
                    )
                )
        return summaries

    def resolve_startup_profile(self, requested: str | None = None) -> str:
        """Return an existing startup profile name, falling back to default."""
        candidate = (requested or "").strip() or DEFAULT_PROFILE_NAME
        if self.profile_dir(candidate).is_dir():
            return candidate
        return DEFAULT_PROFILE_NAME

    def load(self, name: str | None = None) -> ProductionProfile:
        """Load one profile from disk."""
        profile_name = self.resolve_startup_profile(name)
        profile_dir = self.profile_dir(profile_name)
        instructions_path = profile_dir / INSTRUCTIONS_FILENAME
        tools_path = profile_dir / TOOLS_FILENAME
        voice_path = profile_dir / VOICE_FILENAME
        if not instructions_path.is_file():
            raise FileNotFoundError(f"Profile {profile_name!r} has no {INSTRUCTIONS_FILENAME}")
        instructions = instructions_path.read_text(encoding="utf-8").strip()
        if not instructions:
            raise ValueError(f"Profile {profile_name!r} has empty {INSTRUCTIONS_FILENAME}")
        tools = self.parse_tools_text(tools_path.read_text(encoding="utf-8") if tools_path.exists() else "")
        voice = voice_path.read_text(encoding="utf-8").strip() if voice_path.exists() else ""
        return ProductionProfile(
            name=profile_name,
            instructions=instructions,
            tools=tuple(tools),
            voice=voice or None,
        )

    def save_new(
        self,
        name: str,
        *,
        instructions: str,
        tools_text: str,
        voice: str | None = None,
    ) -> ProductionProfile:
        """Create a new production profile."""
        profile_name = self.sanitize_name(name)
        if not profile_name:
            raise ValueError("invalid_profile_name")
        profile_dir = self.profile_dir(profile_name)
        if profile_dir.exists():
            raise FileExistsError(f"Profile {profile_name!r} already exists")
        self._write(profile_name, instructions=instructions, tools_text=tools_text, voice=voice)
        return self.load(profile_name)

    def overwrite(
        self,
        name: str,
        *,
        instructions: str,
        tools_text: str,
        voice: str | None = None,
    ) -> ProductionProfile:
        """Overwrite an existing production profile."""
        profile_name = self.sanitize_name(name)
        if not profile_name:
            raise ValueError("invalid_profile_name")
        if not self.profile_dir(profile_name).is_dir():
            raise FileNotFoundError(f"Profile {profile_name!r} does not exist")
        self._write(profile_name, instructions=instructions, tools_text=tools_text, voice=voice)
        return self.load(profile_name)

    def profile_dir(self, name: str) -> Path:
        """Return a profile directory path after validating its name."""
        profile_name = self.sanitize_name(name)
        if not profile_name:
            raise ValueError("invalid_profile_name")
        return self.root / profile_name

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Return a filesystem-safe profile name."""
        normalized = re.sub(r"\s+", "_", name.strip())
        return re.sub(r"[^a-zA-Z0-9_-]", "", normalized)

    @staticmethod
    def parse_tools_text(text: str) -> list[str]:
        """Return enabled tool names from a tools file body."""
        tools: list[str] = []
        for line in text.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                tools.append(candidate)
        return tools

    def _write(
        self,
        name: str,
        *,
        instructions: str,
        tools_text: str,
        voice: str | None,
    ) -> None:
        """Persist a profile directory."""
        clean_instructions = instructions.strip()
        if not clean_instructions:
            raise ValueError("instructions_required")
        profile_dir = self.profile_dir(name)
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / INSTRUCTIONS_FILENAME).write_text(clean_instructions + "\n", encoding="utf-8")
        (profile_dir / TOOLS_FILENAME).write_text("\n".join(self.parse_tools_text(tools_text)) + "\n", encoding="utf-8")
        voice_value = (voice or "").strip()
        voice_path = profile_dir / VOICE_FILENAME
        if voice_value:
            voice_path.write_text(voice_value + "\n", encoding="utf-8")
        else:
            try:
                voice_path.unlink()
            except FileNotFoundError:
                pass
