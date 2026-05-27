"""Tests for spatial-audio speaker tracking primitives."""

from __future__ import annotations
import math

import pytest

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.head_tracking.speaker import (
    SpeakerSelectionState,
    SoundOrientationController,
    select_speaker,
    target_from_doa,
    target_from_doa_response,
)


def _target(x_offset: float, confidence: float = 0.8) -> HeadTrackerTarget:
    return HeadTrackerTarget(
        x_offset=x_offset,
        y_offset=0.0,
        confidence=confidence,
        bbox=(0.4 + x_offset * 0.1, 0.3, 0.2, 0.2),
        frame_size=(640, 480),
    )


def test_target_from_doa_maps_front_left_and_right() -> None:
    """DoA angles should map to normalized front/left/right offsets."""
    assert target_from_doa(math.pi / 2.0, speech_detected=True).x_offset == pytest.approx(0.0)
    assert target_from_doa(0.0).x_offset == pytest.approx(1.0)
    assert target_from_doa(math.pi).x_offset == pytest.approx(-1.0)
    assert target_from_doa_response(None) is None
    assert target_from_doa_response({"angle": None}) is None


def test_sound_orientation_requires_stable_off_center_samples() -> None:
    """Sound orientation should wait for repeated off-center samples."""
    controller = SoundOrientationController(
        deadband=0.08,
        recenter_threshold=0.14,
        stable_samples=2,
        smoothing=1.0,
        max_yaw_degrees=45.0,
        max_step_degrees=30.0,
    )
    target = target_from_doa(0.0, speech_detected=True)

    assert controller.update(target) is None
    command = controller.update(target)

    assert command is not None
    assert command.body_yaw == pytest.approx(math.radians(30.0))
    assert command.yaw_correction == pytest.approx(math.radians(30.0))


def test_sound_orientation_turns_left_for_left_side_doa() -> None:
    """Sound orientation should mirror body-yaw signs for left/right DoA."""
    controller = SoundOrientationController(stable_samples=1, smoothing=1.0)
    command = controller.update(target_from_doa(math.pi, speech_detected=True))

    assert command is not None
    assert command.body_yaw < 0.0
    assert command.yaw_correction < 0.0


def test_sound_orientation_clamps_total_body_yaw() -> None:
    """Sound orientation should clamp accumulated body yaw."""
    controller = SoundOrientationController(stable_samples=1, smoothing=1.0, max_yaw_degrees=45.0)
    target = target_from_doa(0.0, speech_detected=True)

    for _ in range(5):
        command = controller.update(target)

    assert command is not None
    assert command.body_yaw == pytest.approx(math.radians(45.0))


def test_select_speaker_uses_audio_agreement_over_visual_confidence() -> None:
    """Speaker selection should prefer audio-aligned faces over larger faces."""
    selected = select_speaker(
        [_target(-0.8, confidence=0.95), _target(0.35, confidence=0.7)],
        audio_x_offset=0.4,
    )

    assert selected.target is not None
    assert selected.target.x_offset == pytest.approx(0.35)
    assert selected.audio_agreement > 0.9


def test_select_speaker_uses_visual_confidence_without_audio() -> None:
    """Speaker selection should fall back to visual confidence without audio."""
    selected = select_speaker(
        [_target(-0.2, confidence=0.4), _target(0.3, confidence=0.9)],
        audio_x_offset=None,
    )

    assert selected.target is not None
    assert selected.target.x_offset == pytest.approx(0.3)


def test_select_speaker_uses_continuity_to_stabilize_ties() -> None:
    """Speaker selection should use continuity when candidates tie."""
    state = SpeakerSelectionState(last_x_offset=-0.2)

    selected = select_speaker(
        [_target(-0.2, confidence=0.6), _target(0.2, confidence=0.6)],
        audio_x_offset=0.0,
        state=state,
    )

    assert selected.target is not None
    assert selected.target.x_offset == pytest.approx(-0.2)
