"""Tests for the static dashboard APIs and log buffer."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.runtime.console import LocalStream
from reachy_mini_conversation_app.runtime.dashboard import DashboardLogBuffer, sse_event, classify_log
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.face_identity_worker import PerceptionSnapshot
from reachy_mini_conversation_app.vision.face_recognition_lib import Person


def _target(track_id: int | None = 7) -> IdentifiedTarget:
    return IdentifiedTarget(
        target=HeadTrackerTarget(
            x_offset=0.25,
            y_offset=-0.1,
            confidence=0.88,
            bbox=(0.2, 0.15, 0.25, 0.35),
            frame_size=(640, 480),
        ),
        name=None,
        similarity=0.21,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        first_seen_at=10.0,
        last_seen_at=12.0,
        track_id=track_id,
        observed=False,
        held=True,
        stability=0.5,
        can_remember=True,
        last_observed_at=11.5,
    )


class _FakeDB:
    path = "/tmp/faces.db"

    def __init__(self) -> None:
        self.people = [Person("Alice", (np.array([1.0, 0.0], dtype=np.float32),))]

    def persons(self) -> list[Person]:
        return self.people


class _FakeFaceWorker:
    def __init__(self, visible: list[IdentifiedTarget]) -> None:
        self.visible = visible
        self.identifier = SimpleNamespace(db=_FakeDB())
        self.remember_calls: list[tuple[int, str]] = []

    def snapshot(self) -> PerceptionSnapshot:
        return PerceptionSnapshot(visible=tuple(self.visible))

    def remember_visible(self, track_id: int, name: str) -> dict[str, object]:
        self.remember_calls.append((track_id, name))
        if track_id != 7:
            raise KeyError(track_id)
        return {"status": "remembered", "track_id": track_id, "name": name, "exemplar_count": 1}


def _client(*, camera_worker: object | None, face_worker: object | None) -> TestClient:
    handler = MagicMock()
    handler.deps = SimpleNamespace(camera_worker=camera_worker, face_identity_worker=face_worker)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    app = FastAPI()
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    return TestClient(app)


def test_dashboard_status_reports_unavailable_camera_and_face_worker() -> None:
    """Dashboard status should be explicit when camera/face recognition are absent."""
    client = _client(camera_worker=None, face_worker=None)

    response = client.get("/api/dashboard/status")

    assert response.status_code == 200
    data = response.json()
    assert data["camera"]["available"] is False
    assert data["camera"]["frame_available"] is False
    assert data["face_recognition"]["available"] is False
    assert data["face_recognition"]["people"] == []


def test_dashboard_face_routes_return_frame_and_visible_faces() -> None:
    """Dashboard face APIs should expose latest frame and annotated face state."""
    camera_worker = MagicMock()
    camera_worker.head_tracker = object()
    camera_worker.get_latest_frame.return_value = np.zeros((32, 48, 3), dtype=np.uint8)
    camera_worker.get_speaker_focus_name.return_value = None
    face_worker = _FakeFaceWorker([_target()])
    client = _client(camera_worker=camera_worker, face_worker=face_worker)

    frame_response = client.get("/api/face/frame.jpg")
    state_response = client.get("/api/face/state")

    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"] == "image/jpeg"
    assert state_response.status_code == 200
    face = state_response.json()["faces"][0]
    assert face["track_id"] == 7
    assert face["bbox"] == {"x": 0.2, "y": 0.15, "width": 0.25, "height": 0.35}
    assert face["label"] == "unknown"
    assert face["observed"] is False
    assert face["held"] is True
    assert face["stability"] == 0.5
    assert face["can_remember"] is True
    assert face["last_observed_at"] == 11.5


def test_dashboard_remember_route_saves_selected_face() -> None:
    """Dashboard remember route should delegate to the visible track saver."""
    face_worker = _FakeFaceWorker([_target()])
    client = _client(camera_worker=MagicMock(), face_worker=face_worker)

    response = client.post("/api/face/remember", json={"face_id": 7, "name": "Alice"})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["name"] == "Alice"
    assert face_worker.remember_calls == [(7, "Alice")]


def test_dashboard_remember_route_rejects_stale_face() -> None:
    """Dashboard remember route should report stale selected faces as not visible."""
    face_worker = _FakeFaceWorker([_target()])
    client = _client(camera_worker=MagicMock(), face_worker=face_worker)

    response = client.post("/api/face/remember", json={"face_id": 99, "name": "Alice"})

    assert response.status_code == 404
    assert response.json()["error"] == "face_not_visible"


def test_dashboard_remember_route_rejects_empty_names() -> None:
    """Dashboard remember route should validate names before saving."""
    face_worker = _FakeFaceWorker([_target()])
    client = _client(camera_worker=MagicMock(), face_worker=face_worker)

    response = client.post("/api/face/remember", json={"face_id": 7, "name": "  "})

    assert response.status_code == 400
    assert response.json()["error"] == "name_required"
    assert face_worker.remember_calls == []


def test_dashboard_log_buffer_limits_and_formats_sse() -> None:
    """Dashboard log buffer should classify, bound, and SSE-format events."""
    logs = DashboardLogBuffer(capacity=2)
    first = logs.add("Camera started", category="VISION")
    second = logs.add("Tool call: who_is_here")
    third = logs.add("Realtime session updated")

    snapshot = logs.snapshot()

    assert first not in snapshot
    assert [event.id for event in snapshot] == [second.id, third.id]
    assert classify_log("reachy_mini_conversation_app.vision.camera_worker", "frame ready") == "VISION"
    payload = sse_event(third)
    assert f"id: {third.id}" in payload
    assert "event: log" in payload
    assert '"category": "LLM"' in payload
