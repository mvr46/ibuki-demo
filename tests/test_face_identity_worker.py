"""Tests for the face identity worker."""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np

from reachy_mini_conversation_app.face_identity_worker import FaceIdentifierWorker
from reachy_mini_conversation_app.vision.face_identity import FaceObservation, IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget


class _FakeIdentifier:
    def __init__(self, identified: list[IdentifiedTarget]) -> None:
        self.identified = identified
        self.db = _FakeDB()

    def identify(self, frame: np.ndarray, targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
        return self.identified


class _DetectionOnlyIdentifier:
    recognition_available = False

    def observe(self, frame: np.ndarray, targets: list[HeadTrackerTarget]) -> list[FaceObservation]:
        return [FaceObservation(target=target, name=None, similarity=0.0, embedding=None) for target in targets]


class _FakeDB:
    def __init__(self) -> None:
        self.saved: list[tuple[str, np.ndarray]] = []

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.saved.append((name, embedding.copy()))

    def exemplar_count(self, name: str) -> int:
        return sum(1 for saved_name, _ in self.saved if saved_name == name)


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
    assert worker.snapshot().visible == ()
    assert worker.drain_events() == []
    worker._process_once(10.4)

    snapshot = worker.snapshot()
    events = worker.drain_events()
    assert snapshot.visible[0].name == "Alice"
    assert snapshot.visible[0].track_id == 0
    assert snapshot.visible[0].first_seen_at == 10.0
    assert snapshot.visible[0].last_observed_at == 10.4
    assert events[0].kind == "entered"
    assert events[0].name == "Alice"


def test_face_identity_worker_can_expose_detection_only_targets() -> None:
    """Detector-only fallback should still expose stable face boxes."""
    target = _target()
    worker = FaceIdentifierWorker(
        _camera([target]),
        _DetectionOnlyIdentifier(),
        require_embedding_to_confirm=False,
    )

    worker._process_once(10.0)
    assert worker.snapshot().visible == ()
    worker._process_once(10.4)

    visible = worker.snapshot().visible
    events = worker.drain_events()
    assert len(visible) == 1
    assert visible[0].name is None
    assert visible[0].embedding is None
    assert visible[0].can_remember is False
    assert worker.recognition_available is False
    assert events[0].kind == "entered"
    assert events[0].name is None


def test_face_identity_worker_emits_named_and_left_events() -> None:
    """A stable unknown track should emit named and left transitions."""
    unknown = _identified(None, x_offset=0.0)
    named = _identified("Bob", x_offset=0.02)
    worker = FaceIdentifierWorker(_camera([unknown.target]), _FakeIdentifier([]))

    worker._update_state([unknown], 10.0)
    assert worker.drain_events() == []
    worker._update_state([unknown], 10.4)
    assert worker.drain_events()[0].kind == "entered"

    worker._update_state([named], 11.0)
    assert worker.drain_events() == []
    worker._update_state([named], 11.2)
    named_events = worker.drain_events()
    assert [(event.kind, event.name) for event in named_events] == [("named", "Bob")]

    worker._update_state([], 12.0)
    assert worker.snapshot().visible[0].name == "Bob"
    assert worker.snapshot().visible[0].held is True
    assert worker.drain_events() == []

    worker._update_state([], 13.0)
    left_events = worker.drain_events()
    assert [(event.kind, event.name) for event in left_events] == [("left", "Bob")]
    assert left_events[0].last_seen_at == 11.2
    assert worker.snapshot().last_seen["Bob"] == 11.2


def test_face_identity_worker_remembers_visible_track() -> None:
    """remember_visible should save the selected visible embedding and update state."""
    unknown = _identified(None)
    identifier = _FakeIdentifier([])
    worker = FaceIdentifierWorker(_camera([unknown.target]), identifier)
    worker._update_state([unknown], 10.0)
    worker._update_state([unknown], 10.4)
    track_id = worker.snapshot().visible[0].track_id

    assert track_id is not None
    result = worker.remember_visible(track_id, "Alice")

    snapshot = worker.snapshot()
    events = worker.drain_events()
    assert result["status"] == "remembered"
    assert result["name"] == "Alice"
    assert result["exemplar_count"] == 1
    assert snapshot.visible[0].name == "Alice"
    assert snapshot.visible[0].track_id == track_id
    assert identifier.db.saved[0][0] == "Alice"
    assert np.array_equal(identifier.db.saved[0][1], unknown.embedding)
    assert events[-1].kind == "named"
    assert events[-1].name == "Alice"


def test_face_identity_worker_records_visual_history() -> None:
    """Visible tracks should be queryable as bearing-aware observations."""
    named = _identified("Matteo", x_offset=-0.5)
    unknown = _identified(None, x_offset=0.4)
    worker = FaceIdentifierWorker(
        _camera([named.target, unknown.target]),
        _FakeIdentifier([]),
        camera_horizontal_fov_deg=60.0,
    )

    worker._update_state([named, unknown], 10.0)
    worker._update_state([named, unknown], 10.4)

    observations = worker.visual_window(9.0, 11.0)
    assert len(observations) == 2
    assert observations[0].track_id == 0
    assert observations[0].name == "Matteo"
    assert observations[0].visual_bearing_deg == -15.0
    assert observations[1].track_id == 1
    assert observations[1].name is None
    assert observations[1].visual_bearing_deg == 12.0


def test_face_identity_worker_rejects_stale_visible_track() -> None:
    """remember_visible should fail cleanly for absent/stale tracks."""
    worker = FaceIdentifierWorker(_camera([]), _FakeIdentifier([]))

    try:
        worker.remember_visible(99, "Alice")
    except KeyError as exc:
        assert "track_id=99" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("remember_visible should reject stale track IDs")


def test_face_identity_worker_holds_known_identity_through_dirty_frames() -> None:
    """A dirty frame should not erase a confirmed best-guess identity."""
    alice = _identified("Alice", x_offset=0.0)
    dirty = FaceObservation(target=_target(0.02), name=None, similarity=0.0, embedding=None)
    worker = FaceIdentifierWorker(_camera([alice.target]), _FakeIdentifier([]))

    worker._update_state([alice], 10.0)
    worker._update_state([alice], 10.4)
    worker.drain_events()
    worker._update_state([dirty], 10.8)

    visible = worker.snapshot().visible
    assert len(visible) == 1
    assert visible[0].name == "Alice"
    assert visible[0].observed is True
    assert visible[0].held is False
    assert visible[0].last_observed_at == 10.8
    assert worker.drain_events() == []


def test_face_identity_worker_holds_known_identity_when_target_is_missed() -> None:
    """A missed frame should hold the best guess until the missing window expires."""
    alice = _identified("Alice")
    worker = FaceIdentifierWorker(_camera([alice.target]), _FakeIdentifier([]))

    worker._update_state([alice], 10.0)
    worker._update_state([alice], 10.4)
    worker.drain_events()
    worker._update_state([], 11.0)

    visible = worker.snapshot().visible
    assert len(visible) == 1
    assert visible[0].name == "Alice"
    assert visible[0].observed is False
    assert visible[0].held is True
    assert 0.0 < visible[0].stability < 1.0
    assert worker.drain_events() == []


def test_face_identity_worker_hides_one_frame_false_detection() -> None:
    """One-frame detector noise should never become visible or emit entered."""
    unknown = _identified(None)
    worker = FaceIdentifierWorker(_camera([unknown.target]), _FakeIdentifier([]))

    worker._update_state([unknown], 10.0)
    worker._update_state([], 12.0)

    assert worker.snapshot().visible == ()
    assert worker.drain_events() == []


def test_face_identity_worker_suppresses_name_flicker_until_repeated() -> None:
    """A single competing identity should not replace the stable best guess."""
    alice = _identified("Alice", x_offset=0.0)
    bob = _identified("Bob", x_offset=0.02)
    worker = FaceIdentifierWorker(_camera([alice.target]), _FakeIdentifier([]))

    worker._update_state([alice], 10.0)
    worker._update_state([alice], 10.4)
    worker.drain_events()
    worker._update_state([bob], 10.8)

    assert worker.snapshot().visible[0].name == "Alice"
    assert worker.drain_events() == []

    worker._update_state([bob], 11.2)
    events = worker.drain_events()
    assert worker.snapshot().visible[0].name == "Bob"
    assert [(event.kind, event.name) for event in events] == [("named", "Bob")]


def test_face_identity_worker_remembers_held_track_with_last_embedding() -> None:
    """remember_visible should use the last good embedding for a held track."""
    unknown = _identified(None)
    identifier = _FakeIdentifier([])
    worker = FaceIdentifierWorker(_camera([unknown.target]), identifier)
    worker._update_state([unknown], 10.0)
    worker._update_state([unknown], 10.4)
    worker.drain_events()
    track_id = worker.snapshot().visible[0].track_id
    assert track_id is not None

    worker._update_state([], 11.0)
    assert worker.snapshot().visible[0].held is True
    result = worker.remember_visible(track_id, "Alice")

    assert result["status"] == "remembered"
    assert identifier.db.saved[0][0] == "Alice"
    assert np.array_equal(identifier.db.saved[0][1], unknown.embedding)


def test_face_identity_worker_rejects_visible_track_without_embedding() -> None:
    """remember_visible should reject tracks that have no usable embedding."""
    worker = FaceIdentifierWorker(_camera([]), _FakeIdentifier([]))
    with worker._lock:
        worker._state.visible = [
            IdentifiedTarget(
                target=_target(),
                name=None,
                similarity=0.0,
                embedding=None,
                track_id=7,
                can_remember=False,
            )
        ]

    try:
        worker.remember_visible(7, "Alice")
    except ValueError as exc:
        assert "no usable face embedding" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("remember_visible should reject tracks without embeddings")
