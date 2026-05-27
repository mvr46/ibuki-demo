"""Tests for CameraWorker spatial speaker tracking."""

from __future__ import annotations
import time
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.head_tracking.speaker import SoundOrientationController, target_from_doa


class _FakeTracker:
    def __init__(self, targets: list[HeadTrackerTarget]) -> None:
        self.targets = targets

    def get_head_targets(self, frame: np.ndarray) -> list[HeadTrackerTarget]:
        return self.targets

    def get_head_position(self, frame: np.ndarray) -> tuple[None, None]:
        return None, None


class _FakePoller:
    def __init__(self, target: object | None, target_at: float | None) -> None:
        self.target = target
        self.target_at = target_at

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


def test_camera_worker_selects_face_matching_fresh_spatial_audio() -> None:
    """Camera worker should select the face whose x offset matches fresh DoA."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.7, 0.95), _target(0.6, 0.70)]),
        doa_poller=_FakePoller(target_from_doa(0.4 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] > 320
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)


def test_camera_worker_switches_visible_speaker_on_fresh_off_front_audio() -> None:
    """Fresh off-front DoA should break continuity and switch to the audio-matched face."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.45, 0.98), _target(0.45, 0.65)]),
        doa_poller=_FakePoller(target_from_doa(0.55 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._speaker_selection_state.last_x_offset = -0.45
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] > 320
    assert worker._speaker_selection_state.last_x_offset == pytest.approx(0.45)
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)


def test_camera_worker_keeps_locked_face_when_sound_is_front() -> None:
    """Front DoA should not pull focus away from the current visible speaker."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.6, 0.8), _target(0.0, 0.8)]),
        doa_poller=_FakePoller(target_from_doa(np.pi / 2.0, speech_detected=True), now),
    )
    worker._speaker_selection_state.last_x_offset = -0.6
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] < 320
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)


def test_camera_worker_ignores_unmatched_face_during_off_front_audio_search() -> None:
    """An off-camera speaker should make the robot search instead of staying on the old face."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.5, 0.95)]),
        doa_poller=_FakePoller(target_from_doa(0.3 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls == []
    assert worker.get_face_tracking_offsets() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert worker.get_tracking_body_yaw_offset() > 0.0


def test_camera_worker_searches_instead_of_relocking_center_face_for_side_audio() -> None:
    """A locked centered face should not absorb a fresh side-speaker cue."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(0.0, 0.99)]),
        doa_poller=_FakePoller(target_from_doa(0.45 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._speaker_selection_state.last_x_offset = 0.0
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls == []
    assert worker.get_tracking_body_yaw_offset() > 0.0


def test_camera_worker_logs_off_front_sound_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Off-front sound cues should be visible in debug logs."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(0.0, 0.99)]),
        doa_poller=_FakePoller(target_from_doa(0.45 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()
    caplog.set_level(logging.DEBUG, logger="reachy_mini_conversation_app.camera_worker")

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    debug_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "Spatial audio: sound not in front heard" in debug_output
    assert "event=starting_search" in debug_output
    assert "direction=right" in debug_output
    assert "visible_faces=1" in debug_output


def test_camera_worker_searches_left_for_left_side_audio() -> None:
    """Left-side audio search should mirror the right-side body-yaw sign."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(0.5, 0.95)]),
        doa_poller=_FakePoller(target_from_doa(0.7 * np.pi, speech_detected=True), now),
    )
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls == []
    assert worker.get_tracking_body_yaw_offset() < 0.0


def test_camera_worker_continues_audio_search_after_brief_sound() -> None:
    """A short off-front cue should keep turning briefly without constant sound."""
    now = time.monotonic()
    robot = _FakeRobot()
    poller = _FakePoller(target_from_doa(0.2 * np.pi / 2.0, speech_detected=True), now)
    worker = CameraWorker(robot, _FakeTracker([]), doa_poller=poller)
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    worker._update_tracking_from_frame(frame, now)
    first_yaw = worker.get_tracking_body_yaw_offset()
    poller.target_at = now - 10.0
    worker._update_tracking_from_frame(frame, now + 1.2)

    assert first_yaw > 0.0
    assert worker.get_tracking_body_yaw_offset() > first_yaw
    assert robot.look_at_calls == []


def test_camera_worker_keeps_searching_when_only_old_center_face_is_visible() -> None:
    """A stale search should not reacquire a centered face from the previous lock."""
    now = time.monotonic()
    robot = _FakeRobot()
    poller = _FakePoller(target_from_doa(0.2 * np.pi / 2.0, speech_detected=True), now)
    worker = CameraWorker(robot, _FakeTracker([_target(0.0, 0.99)]), doa_poller=poller)
    worker._speaker_selection_state.last_x_offset = 0.0
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    worker._update_tracking_from_frame(frame, now)
    first_yaw = worker.get_tracking_body_yaw_offset()
    poller.target_at = now - 10.0
    worker._update_tracking_from_frame(frame, now + 0.5)

    assert robot.look_at_calls == []
    assert worker.get_face_tracking_offsets() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert worker.get_tracking_body_yaw_offset() > first_yaw


def test_camera_worker_locks_face_found_after_audio_search() -> None:
    """A held sound search should accept a newly visible face in that direction."""
    now = time.monotonic()
    robot = _FakeRobot()
    tracker = _FakeTracker([])
    poller = _FakePoller(target_from_doa(0.2 * np.pi / 2.0, speech_detected=True), now)
    worker = CameraWorker(robot, tracker, doa_poller=poller)
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    worker._update_tracking_from_frame(frame, now)
    poller.target_at = now - 10.0
    tracker.targets = [_target(0.25, 0.8)]
    worker._update_tracking_from_frame(frame, now + 0.5)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] > 320.0
    assert worker.get_tracking_body_yaw_offset() > 0.0


def test_camera_worker_pauses_spatial_audio_while_assistant_speaks() -> None:
    """Assistant audio should gate off DoA bias and body-yaw updates."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.7, 0.95), _target(0.6, 0.70)]),
        doa_poller=_FakePoller(target_from_doa(0.4 * np.pi / 2.0, speech_detected=True), now),
    )
    worker.notify_user_speech_started()
    worker.notify_assistant_audio_started()

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] < 320
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)


def test_camera_worker_falls_back_to_visual_selection_without_doa() -> None:
    """Visual confidence should drive tracking when DoA is unavailable."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(-0.5, 0.9), _target(0.5, 0.4)]),
        doa_poller=_FakePoller(None, None),
    )

    worker._update_tracking_from_frame(np.zeros((480, 640, 3), dtype=np.uint8), now)

    assert robot.look_at_calls
    assert robot.look_at_calls[-1][0] < 320
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)


def test_camera_worker_returns_head_and_body_offsets_to_neutral_after_loss() -> None:
    """Lost speaker focus should interpolate head and body offsets to neutral."""
    now = time.monotonic()
    robot = _FakeRobot()
    worker = CameraWorker(
        robot,
        _FakeTracker([_target(0.6, 0.9)]),
        doa_poller=_FakePoller(target_from_doa(0.4 * np.pi / 2.0, speech_detected=True), now),
    )
    worker._sound_controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    worker.notify_user_speech_started()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker._update_tracking_from_frame(frame, now)

    assert any(abs(value) > 0 for value in worker.get_face_tracking_offsets())
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)

    neutral_pose = np.eye(4, dtype=np.float32)
    worker._update_neutral_interpolation(now + worker.face_lost_delay, neutral_pose)
    worker._update_neutral_interpolation(now + worker.face_lost_delay + worker.interpolation_duration, neutral_pose)

    assert worker.get_face_tracking_offsets() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert worker.get_tracking_body_yaw_offset() == pytest.approx(0.0)
