"""Tests for robot-noise-resistant local turn detection."""

from __future__ import annotations
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.backends.local_llm import LocalLLMResponse
from reachy_mini_conversation_app.backends.local_conversation import LocalConversationHandler
from reachy_mini_conversation_app.backends.local_turn_detector import (
    LocalFrameStats,
    LocalTurnDetector,
    LocalCompletedTurn,
    LocalTurnDetectorConfig,
    _BufferedFrame,
)


SR = 16000


def _speech_like(duration_s: float = 0.6, *, amplitude: float = 0.25) -> np.ndarray:
    """Return a deterministic speech-shaped harmonic fixture."""
    t = np.arange(int(SR * duration_s)) / SR
    wave = (
        0.45 * np.sin(2 * math.pi * 450 * t)
        + 0.3 * np.sin(2 * math.pi * 900 * t)
        + 0.2 * np.sin(2 * math.pi * 1700 * t)
    )
    envelope = 0.55 + 0.45 * np.sin(2 * math.pi * 4 * t) ** 2
    noise = 0.004 * np.random.default_rng(7).normal(size=t.size)
    return np.clip((amplitude * envelope * wave + noise) * 32767, -32768, 32767).astype(np.int16)


def _servo_tone(duration_s: float = 0.8) -> np.ndarray:
    """Return narrowband mechanical tone fixture."""
    t = np.arange(int(SR * duration_s)) / SR
    return np.clip(0.25 * np.sin(2 * math.pi * 180 * t) * 32767, -32768, 32767).astype(np.int16)


def _broadband_motor(duration_s: float = 0.8) -> np.ndarray:
    """Return broadband mechanical noise fixture."""
    noise = 0.22 * np.random.default_rng(11).normal(size=int(SR * duration_s))
    return np.clip(noise * 32767, -32768, 32767).astype(np.int16)


def _detector() -> LocalTurnDetector:
    return LocalTurnDetector(
        LocalTurnDetectorConfig(
            silence_seconds=0.1,
            min_speech_seconds=0.2,
            min_frame_rms=40.0,
        )
    )


def test_default_latency_endpoint_maps_to_23_frames() -> None:
    """The 0.45s endpoint default should map to 23 20ms VAD frames."""
    assert LocalTurnDetectorConfig(silence_seconds=0.45).silence_frames == 23


def _stats(*, speech_like: bool, snr_db: float = 28.2) -> LocalFrameStats:
    """Return deterministic frame stats for direct segment-threshold tests."""
    return LocalFrameStats(
        rms=1200.0,
        noise_floor_rms=45.0,
        snr_db=snr_db,
        speech_band_ratio=0.72,
        spectral_flatness=0.22,
        spectral_centroid_hz=1100.0,
        zero_crossing_rate=0.11,
        peak_dominance=0.18,
        speech_like=speech_like,
        robot_activity=True,
        noise_class="speech_like",
    )


def test_mechanical_noise_fixtures_do_not_complete_turns() -> None:
    """Servo and motor noise should not produce turns for STT."""
    silence = np.zeros(int(SR * 0.2), dtype=np.int16)
    for fixture in (_servo_tone(), _broadband_motor()):
        detector = _detector()
        detector.process(fixture)
        update = detector.process(silence)

        assert update.completed_turns == []


def test_real_speech_fixture_completes_with_pre_roll() -> None:
    """Speech-like audio should complete and keep pre-roll samples."""
    detector = _detector()
    pre_roll = np.zeros(int(SR * 0.25), dtype=np.int16)
    detector.process(pre_roll)
    detector.process(_speech_like())
    update = detector.process(np.zeros(int(SR * 0.2), dtype=np.int16))

    assert len(update.completed_turns) == 1
    completed = update.completed_turns[0]
    assert completed.audio.size > _speech_like().size
    assert completed.speech_ratio >= 0.45


def test_robot_activity_requires_stronger_speech_to_start_turn() -> None:
    """Robot activity should suppress weak self-audio while preserving loud barge-in."""
    silence = np.zeros(int(SR * 0.2), dtype=np.int16)
    weak_detector = _detector()
    weak_detector.process(_speech_like(amplitude=0.03), robot_activity=True)
    weak_update = weak_detector.process(silence, robot_activity=True)

    strong_detector = _detector()
    strong_detector.process(_speech_like(amplitude=0.12), robot_activity=True)
    strong_update = strong_detector.process(silence, robot_activity=True)

    assert weak_update.completed_turns == []
    assert len(strong_update.completed_turns) == 1


def test_robot_activity_allows_high_snr_barge_in_with_moderate_speech_ratio() -> None:
    """Clear human speech during robot activity should not be dropped solely below the 0.75 ratio."""
    detector = _detector()
    speech_indexes = {*range(60), 99}
    frame_audio = np.zeros(detector.config.frame_samples, dtype=np.int16)
    detector._segment = [
        _BufferedFrame(frame_audio, _stats(speech_like=index in speech_indexes)) for index in range(100)
    ]
    detector._speech_frame_indexes = sorted(speech_indexes)

    result = detector._finish_segment()

    assert isinstance(result, LocalCompletedTurn)
    assert result.speech_ratio == pytest.approx(0.61)
    assert result.avg_snr_db == pytest.approx(28.2)
    assert result.robot_activity is True


def test_robot_activity_allows_sustained_barge_in_with_moderate_snr() -> None:
    """Sustained human-like speech should survive the robot-activity gate around observed live values."""
    detector = _detector()
    speech_indexes = {*range(51), 93}
    frame_audio = np.zeros(detector.config.frame_samples, dtype=np.int16)
    detector._segment = [
        _BufferedFrame(frame_audio, _stats(speech_like=index in speech_indexes, snr_db=21.8))
        for index in range(94)
    ]
    detector._speech_frame_indexes = sorted(speech_indexes)

    result = detector._finish_segment()

    assert isinstance(result, LocalCompletedTurn)
    assert result.duration_s == pytest.approx(1.88)
    assert round(result.speech_ratio, 2) == 0.55
    assert result.avg_snr_db == pytest.approx(21.8)


def test_robot_activity_allows_high_snr_barge_in_with_choppy_ratio() -> None:
    """Very high-SNR speech should pass even when robot activity makes frame ratio choppy."""
    detector = _detector()
    speech_indexes = {*range(39), 79}
    frame_audio = np.zeros(detector.config.frame_samples, dtype=np.int16)
    detector._segment = [
        _BufferedFrame(frame_audio, _stats(speech_like=index in speech_indexes, snr_db=27.3))
        for index in range(80)
    ]
    detector._speech_frame_indexes = sorted(speech_indexes)

    result = detector._finish_segment()

    assert isinstance(result, LocalCompletedTurn)
    assert result.duration_s == pytest.approx(1.60)
    assert round(result.speech_ratio, 2) == 0.50
    assert result.avg_snr_db == pytest.approx(27.3)


def test_low_confidence_speech_segment_goes_to_stt() -> None:
    """Low-ratio, low-SNR speech-bearing segments should reach STT instead of being dropped."""
    detector = _detector()
    speech_indexes = {*range(20), 88}
    frame_audio = np.zeros(detector.config.frame_samples, dtype=np.int16)
    detector._segment = [
        _BufferedFrame(frame_audio, _stats(speech_like=index in speech_indexes, snr_db=5.6))
        for index in range(89)
    ]
    detector._speech_frame_indexes = sorted(speech_indexes)

    result = detector._finish_segment()

    assert isinstance(result, LocalCompletedTurn)
    assert result.duration_s == pytest.approx(1.78)
    assert round(result.speech_ratio, 2) == 0.24
    assert result.avg_snr_db == pytest.approx(5.6)
    assert result.robot_activity is True


@pytest.mark.asyncio
async def test_mechanical_noise_receive_does_not_call_stt() -> None:
    """Noise-only mic frames should never reach the local STT adapter."""

    class FakeSTT:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
            self.calls += 1
            return "noise"

    class FakeLLM:
        async def chat(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]],
        ) -> LocalLLMResponse:
            return LocalLLMResponse(content="hello", tool_calls=[])

    class FakeTTS:
        output_sample_rate = 24000

        async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
            return 24000, np.zeros(0, dtype=np.int16)

    stt = FakeSTT()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalConversationHandler(
        deps,
        stt_adapter=stt,
        llm_adapter=FakeLLM(),
        tts_adapter=FakeTTS(),
        turn_detector=_detector(),
    )

    await handler.receive((SR, _servo_tone()))
    await handler.receive((SR, np.zeros(int(SR * 0.2), dtype=np.int16)))
    await asyncio_sleep()

    assert stt.calls == 0


async def asyncio_sleep() -> None:
    """Yield once to allow accidental background STT tasks to run."""
    import asyncio

    await asyncio.sleep(0)
