"""Tests for face identity tools."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs
from reachy_mini_conversation_app.tools.who_is_here import WhoIsHere
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.tools.remember_person import RememberPerson


class _FakeDB:
    def __init__(self) -> None:
        self.saved: list[tuple[str, np.ndarray]] = []

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.saved.append((name, embedding))

    def exemplar_count(self, name: str) -> int:
        return sum(1 for saved_name, _ in self.saved if saved_name == name)


class _FakeIdentityWorker:
    def __init__(self, identified: list[IdentifiedTarget]) -> None:
        self.identified = identified
        self.identifier = SimpleNamespace(db=_FakeDB())

    def identify(self, frame: np.ndarray, targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
        assert targets
        return self.identified


def _target(x_offset: float, area: float = 0.09) -> HeadTrackerTarget:
    side = area**0.5
    return HeadTrackerTarget(
        x_offset=x_offset,
        y_offset=0.0,
        confidence=0.9,
        bbox=(0.2, 0.2, side, side),
        frame_size=(640, 480),
    )


def _deps(identity_worker: object | None) -> ToolDependencies:
    head_tracker = MagicMock()
    head_tracker.get_head_targets.return_value = [_target(0.0)]
    camera_worker = MagicMock()
    camera_worker.head_tracker = head_tracker
    camera_worker.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        face_identity_worker=identity_worker,
    )


@pytest.mark.asyncio
async def test_who_is_here_lists_identified_people() -> None:
    """who_is_here should report current identities with compact offsets."""
    identity_worker = _FakeIdentityWorker(
        [
            IdentifiedTarget(
                target=_target(0.25),
                name="Alice",
                similarity=0.81234,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]
    )

    result = await WhoIsHere()(_deps(identity_worker))

    assert result["people"] == [
        {
            "name": "Alice",
            "x_offset": 0.25,
            "y_offset": 0.0,
            "similarity": 0.812,
            "seconds_in_view": 0.0,
        }
    ]


@pytest.mark.asyncio
async def test_remember_person_saves_largest_unknown_face() -> None:
    """remember_person should save the largest unknown visible face."""
    small_known = IdentifiedTarget(
        target=_target(-0.4, area=0.04),
        name="Alice",
        similarity=0.9,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    large_unknown = IdentifiedTarget(
        target=_target(0.4, area=0.16),
        name=None,
        similarity=0.2,
        embedding=np.array([0.0, 1.0], dtype=np.float32),
    )
    identity_worker = _FakeIdentityWorker([small_known, large_unknown])

    result = await RememberPerson()(_deps(identity_worker), name="Bob")

    assert result["status"] == "remembered"
    assert result["name"] == "Bob"
    assert result["exemplar_count"] == 1
    assert identity_worker.identifier.db.saved[0][0] == "Bob"


def test_face_tools_only_active_when_identity_worker_is_wired() -> None:
    """Face identity tools should be hidden until the identity worker is available."""
    inactive_names = {spec["name"] for spec in get_active_tool_specs(_deps(None))}
    active_names = {spec["name"] for spec in get_active_tool_specs(_deps(object()))}

    assert "who_is_here" not in inactive_names
    assert "remember_person" not in inactive_names
    assert "who_is_here" in active_names
    assert "remember_person" in active_names
