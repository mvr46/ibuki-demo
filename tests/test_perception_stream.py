"""Tests for perception stream formatting and injection."""

from __future__ import annotations
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.perception_stream import (
    _format_event,
    _format_snapshot,
    run_perception_stream,
)
from reachy_mini_conversation_app.vision.face_identity_worker import VisionEvent, PerceptionSnapshot


class _FakeHandler:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def inject_environment_message(self, text: str, *, trigger_response: bool = False) -> None:
        assert trigger_response is False
        self.messages.append(text)


class _FakeWorker:
    def __init__(self, *, assistant_speaking: bool = False) -> None:
        self.camera_worker = SimpleNamespace(_assistant_speaking=assistant_speaking)
        self.events = [VisionEvent("entered", "Alice", "center", 10.0)]

    def drain_events(self) -> list[VisionEvent]:
        events = self.events
        self.events = []
        return events

    def snapshot(self) -> PerceptionSnapshot:
        return PerceptionSnapshot()


def _target(name: str | None, x_offset: float) -> IdentifiedTarget:
    return IdentifiedTarget(
        target=HeadTrackerTarget(
            x_offset=x_offset,
            y_offset=0.0,
            confidence=0.8,
            bbox=(0.2, 0.2, 0.2, 0.2),
            frame_size=(640, 480),
        ),
        name=name,
        similarity=0.8,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )


def test_perception_stream_formats_events_and_snapshots() -> None:
    """Vision messages should use the bracketed environment-message format."""
    entered = VisionEvent("entered", "Alice", "center", 10.0)
    left = VisionEvent("left", "Bob", "left", 70.0, last_seen_at=10.0)
    snapshot = PerceptionSnapshot(
        visible=(_target("Alice", 0.0), _target(None, 0.6)),
        last_seen={"Bob": 10.0},
        last_positions={"Bob": "left"},
    )

    assert _format_event(entered) == "[Vision: Alice entered the frame (center)]"
    assert _format_event(left) == "[Vision: Bob left, last seen 1 min ago]"
    assert _format_snapshot(snapshot, now=250.0) == (
        "[Vision: Visible now - Alice (center), unknown (right). Last seen recently: Bob (left, left 4 min ago)]"
    )


@pytest.mark.asyncio
async def test_perception_stream_injects_events() -> None:
    """The perception stream should inject queued events through the handler hook."""
    worker = _FakeWorker()
    handler = _FakeHandler()
    task = asyncio.create_task(run_perception_stream(worker, handler, snapshot_interval_s=999.0, event_debounce_s=0.0))

    await asyncio.sleep(0.6)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler.messages == ["[Vision: Alice entered the frame (center)]"]


@pytest.mark.asyncio
async def test_perception_stream_suppresses_while_assistant_speaks() -> None:
    """The perception stream should not inject events while assistant audio is active."""
    worker = _FakeWorker(assistant_speaking=True)
    handler = _FakeHandler()
    task = asyncio.create_task(run_perception_stream(worker, handler, snapshot_interval_s=0.0, event_debounce_s=0.0))

    await asyncio.sleep(0.6)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler.messages == []
    assert worker.events
