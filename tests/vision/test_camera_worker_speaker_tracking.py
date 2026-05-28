"""Tests for CameraWorker visual-only speaker tracking."""

from __future__ import annotations
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.head_tracking.speaker import target_from_doa


class _FakeTracker:
    def __init__(self, targets: list[HeadTrackerTarget]) -> None:
        self.targets = targets
        self.closed = False

    def get_head_targets(self, frame: np.ndarray) -> list[HeadTrackerTarget]:
        return self.targets

    def get_head_position(self, frame: np.ndarray) -> tuple[None, None]:
        return None, None

    def close(self) -> None:
        self.closed = True


class _FakePoller:
    def __init__(self, target: object | None, target_at: float | None) -> None:
        self.target = target
        self.target_at = target_at
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_latest(self) -> tuple[object | None, float | None]:
        return self.target, self.target_at


class _FakeRobot:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=8000)
        self.media = SimpleNamespace(get_frame=lambda: None)
        self.look_at_calls: list[tuple[float, float]] = []

    def look_at_image(self, x: float, y: float, **kwargs: object) -> np.ndarray:
        self.look_at_calls.append((float(x), float(y)))
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(x) / 1000.0
        pose[1, 3] = float(y) / 1000.0
        return pose


def _target(x_offset: float, confidence: float) -> HeadTrackerTarget:
    return HeadTrackerTarget(
        x_offset=x_offset,
        y_offset=0.0,
        confidence=confidence,
        bbox=(0.4 + x_offset * 0.1, 0.3, 0.2, 0.2),
        frame_size=(640, 480),
    )


def test_camera_worker_does_not_auto_build_doa_poller() -> None:
    """Runtime DoA polling should stay disabled even with a target-list tracker."""
    worker = CameraWorker(_FakeRobot(), _FakeTracker([_target(0.0, 0.9)]))

    assert worker.spatial_audio_source is None
    assert worker._build_doa_poller() is None


def test_camera_worker_ignores_explicit_doa_poller_for_visual_selection() -> None:
    """Deprecated DoA inputs should not bias face selection or body yaw."""
    now = time.monotonic()
    robot = _FakeRobot()
    poller = _FakePoller(target_from_doa(0.0, speech_detected=True), now)
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.6, 0.95), _target(0.6, 0.40)]),
        doa_poller=poller,
    )
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert worker.spatial_audio_source is None
    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] < 320
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)
    assert not poller.started


def test_camera_worker_start_stop_does_not_start_doa_source() -> None:
    """Camera lifecycle should not start or stop deprecated DoA pollers."""
    now = time.monotonic()
    tracker = _FakeTracker([_target(0.0, 0.9)])
    poller = _FakePoller(target_from_doa(np.pi, speech_detected=True), now)
    worker = CameraWorker(_FakeRobot(), tracker, doa_poller=poller)

    worker.start()
    worker.stop()

    assert not poller.started
    assert not poller.stopped
    assert tracker.closed


def test_camera_worker_prefers_named_focus_from_identity_snapshot() -> None:
    """Visual-only selection should still bias toward a requested visible name."""
    now = time.monotonic()
    robot = _FakeRobot()
    unknown = _target(-0.5, 0.98)
    alice = _target(0.5, 0.55)
    worker = CameraWorker(robot, _FakeTracker([unknown, alice]))
    worker.set_speaker_focus_name("Alice")
    worker.set_face_identity_worker(
        SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                visible=(
                    IdentifiedTarget(
                        target=alice,
                        name="Alice",
                        similarity=0.86,
                        embedding=np.array([1.0, 0.0], dtype=np.float32),
                    ),
                )
            )
        )
    )

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] > 320
    assert worker._speaker_selection_state.last_x_offset == pytest.approx(0.5)


def test_camera_worker_returns_head_and_body_offsets_to_neutral_after_loss() -> None:
    """Lost visual speaker focus should interpolate head and body offsets to neutral."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(robot, _FakeTracker([_target(0.6, 0.9)]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker._update_tracking_from_frame(frame, now)

    assert any(abs(value) > 0 for value in worker.get_face_tracking_offsets())
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)

    neutral_pose = np.eye(4, dtype=np.float32)
    worker._update_neutral_interpolation(now + worker.face_lost_delay, neutral_pose)
    worker._update_neutral_interpolation(now + worker.face_lost_delay + worker.interpolation_duration, neutral_pose)

    assert worker.get_face_tracking_offsets() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)
