"""Tests for the face identity worker."""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np

from reachy_mini_conversation_app.face_identity_worker import FaceIdentifierWorker
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget


class _FakeIdentifier:
    def __init__(self, identified: list[IdentifiedTarget]) -> None:
        self.identified = identified

    def identify(self, frame: np.ndarray, targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
        return self.identified


def _target(x_offset: float = 0.0) -> HeadTrackerTarget:
    return HeadTrackerTarget(
        x_offset=x_offset,
        y_offset=0.0,
        confidence=0.9,
        bbox=(0.25 + x_offset * 0.1, 0.25, 0.30, 0.30),
        frame_size=(640, 480),
    )


def _identified(name: str | None, x_offset: float = 0.0) -> IdentifiedTarget:
    return IdentifiedTarget(
        target=_target(x_offset),
        name=name,
        similarity=0.8 if name else 0.2,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )


def _camera(targets: list[HeadTrackerTarget]) -> object:
    head_tracker = SimpleNamespace(get_head_targets=lambda _frame: targets)
    return SimpleNamespace(
        head_tracker=head_tracker,
        get_latest_frame=lambda: np.zeros((480, 640, 3), dtype=np.uint8),
    )


def test_face_identity_worker_processes_camera_targets() -> None:
    """The worker should identify camera targets and emit an entered event."""
    identified = _identified("Alice")
    worker = FaceIdentifierWorker(_camera([identified.target]), _FakeIdentifier([identified]))

    worker._process_once(10.0)

    snapshot = worker.snapshot()
    events = worker.drain_events()
    assert snapshot.visible[0].name == "Alice"
    assert snapshot.visible[0].first_seen_at == 10.0
    assert events[0].kind == "entered"
    assert events[0].name == "Alice"


def test_face_identity_worker_emits_named_and_left_events() -> None:
    """A stable unknown track should emit named and left transitions."""
    unknown = _identified(None, x_offset=0.0)
    named = _identified("Bob", x_offset=0.02)
    worker = FaceIdentifierWorker(_camera([unknown.target]), _FakeIdentifier([]))

    worker._update_state([unknown], 10.0)
    assert worker.drain_events()[0].kind == "entered"

    worker._update_state([named], 11.0)
    named_events = worker.drain_events()
    assert [(event.kind, event.name) for event in named_events] == [("named", "Bob")]

    worker._update_state([], 15.0)
    left_events = worker.drain_events()
    assert [(event.kind, event.name) for event in left_events] == [("left", "Bob")]
    assert left_events[0].last_seen_at == 11.0
    assert worker.snapshot().last_seen["Bob"] == 11.0
