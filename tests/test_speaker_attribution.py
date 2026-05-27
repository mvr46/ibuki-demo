"""Tests for multimodal speaker attribution fusion."""

from __future__ import annotations

import pytest

from reachy_mini_conversation_app.face_identity_worker import VisibleTrackObservation
from reachy_mini_conversation_app.speaker_attribution import SpeakerAttributionWorker, format_attributed_speech
from reachy_mini_conversation_app.vision.head_tracking.speaker import SpatialAudioSample


class _AudioSource:
    def __init__(self, samples: list[SpatialAudioSample]) -> None:
        self.samples = samples

    def get_latest(self):
        if not self.samples:
            return None, None
        return None, self.samples[-1].timestamp

    def window(self, start_s: float, end_s: float) -> tuple[SpatialAudioSample, ...]:
        return tuple(sample for sample in self.samples if start_s <= sample.timestamp <= end_s)


class _FaceWorker:
    def __init__(self, observations: list[VisibleTrackObservation]) -> None:
        self.observations = observations

    def visual_window(self, start_s: float, end_s: float) -> tuple[VisibleTrackObservation, ...]:
        return tuple(item for item in self.observations if start_s <= item.timestamp <= end_s)


def _audio(timestamp: float, azimuth_deg: float) -> SpatialAudioSample:
    return SpatialAudioSample(
        timestamp=timestamp,
        angle=0.0,
        x_offset=azimuth_deg / 90.0,
        azimuth_deg=azimuth_deg,
        speech_detected=True,
    )


def _obs(track_id: int, name: str | None, timestamp: float, bearing_deg: float) -> VisibleTrackObservation:
    return VisibleTrackObservation(
        track_id=track_id,
        name=name,
        x_offset=bearing_deg / 30.0,
        visual_bearing_deg=bearing_deg,
        bbox=(0.2, 0.2, 0.3, 0.3),
        confidence=0.9,
        timestamp=timestamp,
    )


def test_speaker_attribution_selects_named_audio_visual_match() -> None:
    """Audio/visual agreement should attribute speech to the matching named track."""
    worker = SpeakerAttributionWorker(
        spatial_audio_source=_AudioSource([_audio(10.2, -28.0), _audio(11.8, -30.0)]),
        face_identity_worker=_FaceWorker([
            _obs(7, "Matteo", 10.1, -31.0),
            _obs(7, "Matteo", 11.9, -30.0),
        ]),
        time_origin_s=10.0,
    )

    worker.notify_user_speech_started(10.0)
    segment = worker.notify_user_transcript("Can you bring me the tray?", 12.0)

    assert segment is not None
    assert segment.person_track_id == 7
    assert segment.person_name == "Matteo"
    assert segment.speaker_label == "Matteo/person_7"
    assert segment.audio_azimuth_deg == pytest.approx(-29.0)
    assert segment.visual_bearing_deg == pytest.approx(-30.5)
    assert segment.confidence > 0.75
    assert not segment.off_camera
    assert "Matteo/person_7 spoke from 0.00s to 2.00s" in format_attributed_speech(segment)


def test_speaker_attribution_uses_off_camera_fallback() -> None:
    """Audio without a winning visible candidate should be attributed off camera."""
    worker = SpeakerAttributionWorker(
        spatial_audio_source=_AudioSource([_audio(21.0, 40.0)]),
        face_identity_worker=_FaceWorker([]),
        time_origin_s=20.0,
    )

    worker.notify_user_speech_started(20.0)
    segment = worker.notify_user_transcript("Hello from the side", 22.0)

    assert segment is not None
    assert segment.off_camera
    assert segment.speaker_label == "off_camera_speaker"
    assert segment.person_track_id is None


def test_speaker_attribution_continuity_breaks_visual_ties() -> None:
    """The previous speaker track should win an otherwise ambiguous visual-only turn."""
    worker = SpeakerAttributionWorker(
        spatial_audio_source=_AudioSource([_audio(30.5, -20.0)]),
        face_identity_worker=_FaceWorker([
            _obs(1, "Alice", 30.1, -20.0),
            _obs(1, "Alice", 31.0, -20.0),
        ]),
        time_origin_s=30.0,
    )

    worker.notify_user_speech_started(30.0)
    first = worker.notify_user_transcript("First turn", 31.0)
    assert first is not None
    assert first.person_track_id == 1

    worker.spatial_audio_source = _AudioSource([])
    worker.face_identity_worker = _FaceWorker([
        _obs(2, "Bob", 32.1, 20.0),
        _obs(1, "Alice", 32.1, -20.0),
        _obs(2, "Bob", 32.9, 20.0),
        _obs(1, "Alice", 32.9, -20.0),
    ])
    worker.notify_user_speech_started(32.0)
    second = worker.notify_user_transcript("Second turn", 33.0)

    assert second is not None
    assert second.person_track_id == 1


def test_speaker_attribution_disagreement_is_low_confidence() -> None:
    """Strong audio/visual disagreement should not confidently select the visible face."""
    worker = SpeakerAttributionWorker(
        spatial_audio_source=_AudioSource([_audio(40.5, 80.0)]),
        face_identity_worker=_FaceWorker([
            _obs(3, "Carol", 40.1, -30.0),
            _obs(3, "Carol", 40.9, -30.0),
        ]),
        time_origin_s=40.0,
    )

    worker.notify_user_speech_started(40.0)
    segment = worker.notify_user_transcript("That came from elsewhere", 41.0)

    assert segment is not None
    assert segment.confidence < 0.5
    assert segment.off_camera


def test_speaker_attribution_self_speech_does_not_poison_continuity() -> None:
    """Assistant-overlap segments should be flagged and excluded from future continuity."""
    worker = SpeakerAttributionWorker(
        spatial_audio_source=_AudioSource([_audio(50.5, -20.0)]),
        face_identity_worker=_FaceWorker([
            _obs(7, "Reachy", 50.1, -20.0),
            _obs(7, "Reachy", 50.9, -20.0),
        ]),
        time_origin_s=50.0,
    )

    worker.notify_assistant_audio_started(49.5)
    worker.notify_user_speech_started(50.0)
    suppressed = worker.notify_user_transcript("assistant echo", 51.0)
    worker.notify_assistant_audio_done(51.2)

    assert suppressed is not None
    assert suppressed.self_speech_suppressed
    assert suppressed.confidence <= 0.2

    worker.spatial_audio_source = _AudioSource([])
    worker.face_identity_worker = _FaceWorker([
        _obs(8, "Dana", 52.1, 20.0),
        _obs(7, "Reachy", 52.1, -20.0),
        _obs(8, "Dana", 52.9, 20.0),
        _obs(7, "Reachy", 52.9, -20.0),
    ])
    worker.notify_user_speech_started(52.0)
    next_segment = worker.notify_user_transcript("real user", 53.0)

    assert next_segment is not None
    assert next_segment.person_track_id == 8


def test_speaker_attribution_ignores_empty_transcripts() -> None:
    """Blank backend transcripts should not create attribution events."""
    worker = SpeakerAttributionWorker(time_origin_s=0.0)

    worker.notify_user_speech_started(1.0)
    assert worker.notify_user_transcript("  ", 2.0) is None
    assert worker.snapshot() == ()
    assert worker.drain_events() == []
