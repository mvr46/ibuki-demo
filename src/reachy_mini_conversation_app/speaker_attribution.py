"""Multimodal speaker-attribution worker for ambient conversation context."""

from __future__ import annotations

import json
import time
import logging
import threading
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping

from reachy_mini_conversation_app.vision.head_tracking.speaker import SpatialAudioSample, SpatialAudioSource


logger = logging.getLogger(__name__)

EvidenceValue = float | str | bool | None
VOICE_CLUSTER_STATUS_UNAVAILABLE = "unavailable"
VISUAL_OBSERVATION_HOLD_SECONDS = 0.75
MIN_AUDIO_VISUAL_WIN_SCORE = 0.45


@dataclass(frozen=True)
class AttributedSpeechSegment:
    """One finalized speech segment with V1 multimodal speaker attribution."""

    segment_id: str
    start_s: float
    end_s: float
    transcript: str
    person_track_id: int | None
    person_name: str | None
    speaker_label: str
    audio_azimuth_deg: float | None
    visual_bearing_deg: float | None
    visual_activity_score: float
    active_speaker_score: float | None
    diarization_cluster: str | None
    voice_cluster_status: str
    confidence: float
    overlap: float
    off_camera: bool
    self_speech_suppressed: bool
    evidence: Mapping[str, EvidenceValue]


@dataclass
class _OpenSpeechSegment:
    segment_id: str
    start_abs_s: float
    end_abs_s: float | None = None
    partial_text: str = ""


@dataclass(frozen=True)
class _VisualCandidate:
    track_id: int
    name: str | None
    visual_bearing_deg: float
    visual_activity_score: float
    overlap: float
    observation_count: int


class SpeakerAttributionWorker:
    """Own speech segments, V1 fusion scoring, and attributed speech events."""

    def __init__(
        self,
        *,
        spatial_audio_source: SpatialAudioSource | None = None,
        face_identity_worker: object | None = None,
        assistant_state_source: object | None = None,
        time_origin_s: float | None = None,
    ) -> None:
        """Initialize the worker."""
        self.spatial_audio_source = spatial_audio_source
        self.face_identity_worker = face_identity_worker
        self.assistant_state_source = assistant_state_source
        self._time_origin_s = time_origin_s
        self._lock = threading.Lock()
        self._current: _OpenSpeechSegment | None = None
        self._segments: list[AttributedSpeechSegment] = []
        self._events: list[AttributedSpeechSegment] = []
        self._next_segment_id = 1
        self._last_speaker_track_id: int | None = None
        self._assistant_active_started_abs_s: float | None = None
        self._assistant_intervals_abs_s: list[tuple[float, float]] = []

    def notify_user_speech_started(self, now: float | None = None) -> None:
        """Begin a user speech segment."""
        timestamp = _now(now)
        with self._lock:
            self._ensure_origin(timestamp)
            self._current = _OpenSpeechSegment(
                segment_id=self._allocate_segment_id_locked(),
                start_abs_s=timestamp,
            )

    def notify_user_speech_stopped(self, now: float | None = None) -> None:
        """Mark the end of the current user speech segment."""
        timestamp = _now(now)
        with self._lock:
            if self._current is not None:
                self._current.end_abs_s = timestamp

    def notify_user_partial(self, text: str, now: float | None = None) -> None:
        """Record partial transcript text and lazily start a segment if needed."""
        timestamp = _now(now)
        cleaned = str(text or "")
        with self._lock:
            self._ensure_origin(timestamp)
            if self._current is None:
                self._current = _OpenSpeechSegment(
                    segment_id=self._allocate_segment_id_locked(),
                    start_abs_s=timestamp,
                )
            self._current.partial_text = cleaned

    def notify_user_transcript(self, transcript: str, now: float | None = None) -> AttributedSpeechSegment | None:
        """Finalize the current speech segment from a backend ASR transcript."""
        timestamp = _now(now)
        cleaned = str(transcript or "").strip()
        with self._lock:
            self._ensure_origin(timestamp)
            current = self._current
            self._current = None
            if not cleaned:
                return None
            if current is None:
                current = _OpenSpeechSegment(
                    segment_id=self._allocate_segment_id_locked(),
                    start_abs_s=timestamp,
                )
            end_abs_s = current.end_abs_s if current.end_abs_s is not None else timestamp

            segment = self._build_segment_locked(
                segment_id=current.segment_id,
                start_abs_s=current.start_abs_s,
                end_abs_s=end_abs_s,
                transcript=cleaned,
            )
            self._segments.append(segment)
            self._events.append(segment)
            if not segment.off_camera and not segment.self_speech_suppressed:
                self._last_speaker_track_id = segment.person_track_id
            return segment

    def notify_assistant_audio_started(self, now: float | None = None) -> None:
        """Mark assistant playback as active for self-speech suppression."""
        timestamp = _now(now)
        with self._lock:
            if self._assistant_active_started_abs_s is None:
                self._assistant_active_started_abs_s = timestamp

    def notify_assistant_audio_done(self, now: float | None = None) -> None:
        """Close an assistant playback interval."""
        timestamp = _now(now)
        with self._lock:
            if self._assistant_active_started_abs_s is not None:
                self._assistant_intervals_abs_s.append((self._assistant_active_started_abs_s, timestamp))
                self._assistant_active_started_abs_s = None
                self._assistant_intervals_abs_s = self._assistant_intervals_abs_s[-20:]

    def snapshot(self) -> tuple[AttributedSpeechSegment, ...]:
        """Return finalized attributed speech records."""
        with self._lock:
            return tuple(self._segments)

    def drain_events(self) -> list[AttributedSpeechSegment]:
        """Drain finalized attributed speech events waiting for context injection."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _allocate_segment_id_locked(self) -> str:
        segment_id = f"speech_{self._next_segment_id:06d}"
        self._next_segment_id += 1
        return segment_id

    def _ensure_origin(self, timestamp: float) -> None:
        if self._time_origin_s is None:
            self._time_origin_s = timestamp

    def _relative_time(self, timestamp: float) -> float:
        origin = timestamp if self._time_origin_s is None else self._time_origin_s
        return max(0.0, timestamp - origin)

    def _build_segment_locked(
        self,
        *,
        segment_id: str,
        start_abs_s: float,
        end_abs_s: float,
        transcript: str,
    ) -> AttributedSpeechSegment:
        if end_abs_s < start_abs_s:
            start_abs_s, end_abs_s = end_abs_s, start_abs_s

        audio_samples = _audio_window(self.spatial_audio_source, start_abs_s, end_abs_s)
        audio_azimuth_deg = _mean_audio_azimuth(audio_samples)
        visual_candidates = _visual_candidates(self.face_identity_worker, start_abs_s, end_abs_s)
        self_speech_suppressed = self._assistant_overlaps_locked(start_abs_s, end_abs_s)

        selected, selected_score, angular_match, continuity = self._select_candidate(
            visual_candidates,
            audio_azimuth_deg,
        )
        has_audio = audio_azimuth_deg is not None
        off_camera = False
        if selected is None:
            off_camera = has_audio
        elif has_audio and selected_score < MIN_AUDIO_VISUAL_WIN_SCORE:
            off_camera = True
            selected = None

        if selected is None:
            visual_bearing_deg = None
            visual_activity_score = 0.0
            overlap = 0.0
            person_track_id = None
            person_name = None
            speaker_label = "off_camera_speaker" if off_camera else "unknown_speaker"
            confidence = 0.48 if off_camera else 0.20
        else:
            visual_bearing_deg = selected.visual_bearing_deg
            visual_activity_score = selected.visual_activity_score
            overlap = selected.overlap
            person_track_id = selected.track_id
            person_name = selected.name
            speaker_label = _speaker_label(selected.name, selected.track_id)
            confidence = selected_score

        if self_speech_suppressed:
            confidence = min(confidence, 0.20)

        evidence: dict[str, EvidenceValue] = {
            "absolute_start_s": start_abs_s,
            "absolute_end_s": end_abs_s,
            "audio_sample_count": float(len(audio_samples)),
            "visual_candidate_count": float(len(visual_candidates)),
            "angular_match": angular_match,
            "continuity": continuity,
            "selected_score": selected_score,
            "assistant_overlap": self_speech_suppressed,
            "active_speaker_status": VOICE_CLUSTER_STATUS_UNAVAILABLE,
            "voice_cluster_status": VOICE_CLUSTER_STATUS_UNAVAILABLE,
        }

        return AttributedSpeechSegment(
            segment_id=segment_id,
            start_s=self._relative_time(start_abs_s),
            end_s=self._relative_time(end_abs_s),
            transcript=transcript,
            person_track_id=person_track_id,
            person_name=person_name,
            speaker_label=speaker_label,
            audio_azimuth_deg=audio_azimuth_deg,
            visual_bearing_deg=visual_bearing_deg,
            visual_activity_score=_clamp01(visual_activity_score),
            active_speaker_score=None,
            diarization_cluster=None,
            voice_cluster_status=VOICE_CLUSTER_STATUS_UNAVAILABLE,
            confidence=_clamp01(confidence),
            overlap=_clamp01(overlap),
            off_camera=off_camera,
            self_speech_suppressed=self_speech_suppressed,
            evidence=evidence,
        )

    def _select_candidate(
        self,
        candidates: tuple[_VisualCandidate, ...],
        audio_azimuth_deg: float | None,
    ) -> tuple[_VisualCandidate | None, float, float, float]:
        best: tuple[float, _VisualCandidate, float, float] | None = None
        for candidate in candidates:
            angular_match = _angular_match(audio_azimuth_deg, candidate.visual_bearing_deg)
            continuity = 1.0 if candidate.track_id == self._last_speaker_track_id else 0.0
            score = angular_match * 0.55 + candidate.visual_activity_score * 0.30 + continuity * 0.15
            if best is None or (score, candidate.observation_count) > (best[0], best[1].observation_count):
                best = (score, candidate, angular_match, continuity)

        if best is None:
            return None, 0.0, 0.0, 0.0
        return best[1], _clamp01(best[0]), _clamp01(best[2]), _clamp01(best[3])

    def _assistant_overlaps_locked(self, start_abs_s: float, end_abs_s: float) -> bool:
        if self._assistant_currently_speaking_locked():
            return True
        for assistant_start, assistant_end in self._assistant_intervals_abs_s:
            if max(start_abs_s, assistant_start) <= min(end_abs_s, assistant_end):
                return True
        return False

    def _assistant_currently_speaking_locked(self) -> bool:
        if self._assistant_active_started_abs_s is not None:
            return True
        source = self.assistant_state_source
        if source is None:
            return False
        lock = getattr(source, "_speech_state_lock", None)
        if lock is None:
            return bool(getattr(source, "_assistant_speaking", False))
        with lock:
            return bool(getattr(source, "_assistant_speaking", False))


def format_attributed_speech(segment: AttributedSpeechSegment) -> str:
    """Format an attributed segment as an ambient conversation-context message."""
    visual_state = "visually active" if segment.visual_activity_score >= 0.35 else "visual activity low"
    if segment.visual_activity_score <= 0.0:
        visual_state = "no visual activity"

    parts = [
        f"{segment.speaker_label} spoke from {segment.start_s:.2f}s to {segment.end_s:.2f}s",
        _format_angle("audio", segment.audio_azimuth_deg),
        _format_angle("visual", segment.visual_bearing_deg),
        visual_state,
        f"voice cluster {segment.voice_cluster_status}",
    ]
    if segment.self_speech_suppressed:
        parts.append("assistant-overlap suppressed")
    parts.extend(
        [
            f"confidence {segment.confidence:.2f}",
            f"transcript={json.dumps(segment.transcript)}",
        ]
    )
    return f"[Speech attribution: {', '.join(parts)}]"


def _audio_window(
    source: SpatialAudioSource | None,
    start_abs_s: float,
    end_abs_s: float,
) -> tuple[SpatialAudioSample, ...]:
    if source is None:
        return ()
    window = getattr(source, "window", None)
    if not callable(window):
        return ()
    try:
        return tuple(window(start_abs_s, end_abs_s))
    except Exception as exc:
        logger.debug("Spatial audio window unavailable for speaker attribution: %s", exc)
        return ()


def _mean_audio_azimuth(samples: tuple[SpatialAudioSample, ...]) -> float | None:
    if not samples:
        return None
    speech_samples = tuple(sample for sample in samples if sample.speech_detected)
    selected = speech_samples or samples
    return sum(sample.azimuth_deg for sample in selected) / len(selected)


def _visual_candidates(
    face_identity_worker: object | None,
    start_abs_s: float,
    end_abs_s: float,
) -> tuple[_VisualCandidate, ...]:
    observations = _visual_window(face_identity_worker, start_abs_s, end_abs_s)
    by_track: dict[int, list[object]] = defaultdict(list)
    for observation in observations:
        track_id = getattr(observation, "track_id", None)
        if track_id is None:
            continue
        by_track[int(track_id)].append(observation)

    duration = max(0.01, end_abs_s - start_abs_s)
    candidates: list[_VisualCandidate] = []
    for track_id, items in by_track.items():
        if not items:
            continue
        bearing = _mean(float(getattr(item, "visual_bearing_deg", 0.0)) for item in items)
        confidence = _mean(float(getattr(item, "confidence", 0.0)) for item in items)
        timestamps = [float(getattr(item, "timestamp", start_abs_s)) for item in items]
        overlap = _estimate_visual_overlap(timestamps, start_abs_s, end_abs_s, duration)
        latest = max(items, key=lambda item: float(getattr(item, "timestamp", start_abs_s)))
        candidates.append(
            _VisualCandidate(
                track_id=track_id,
                name=getattr(latest, "name", None),
                visual_bearing_deg=bearing,
                visual_activity_score=_clamp01(confidence * max(overlap, 0.20)),
                overlap=overlap,
                observation_count=len(items),
            )
        )
    return tuple(candidates)


def _visual_window(
    face_identity_worker: object | None,
    start_abs_s: float,
    end_abs_s: float,
) -> tuple[object, ...]:
    if face_identity_worker is None:
        return ()
    visual_window = getattr(face_identity_worker, "visual_window", None)
    if callable(visual_window):
        try:
            return tuple(visual_window(start_abs_s, end_abs_s))
        except Exception as exc:
            logger.debug("Visual history unavailable for speaker attribution: %s", exc)
            return ()
    return ()


def _estimate_visual_overlap(
    timestamps: list[float],
    start_abs_s: float,
    end_abs_s: float,
    duration: float,
) -> float:
    if not timestamps:
        return 0.0
    observed_start = max(start_abs_s, min(timestamps) - VISUAL_OBSERVATION_HOLD_SECONDS / 2.0)
    observed_end = min(end_abs_s, max(timestamps) + VISUAL_OBSERVATION_HOLD_SECONDS / 2.0)
    return _clamp01((observed_end - observed_start) / duration)


def _angular_match(audio_azimuth_deg: float | None, visual_bearing_deg: float | None) -> float:
    if audio_azimuth_deg is None or visual_bearing_deg is None:
        return 0.5
    return _clamp01(1.0 - abs(audio_azimuth_deg - visual_bearing_deg) / 90.0)


def _format_angle(label: str, value: float | None) -> str:
    if value is None:
        return f"{label} unavailable"
    magnitude = abs(value)
    if magnitude < 1.0:
        direction = "front"
    else:
        direction = "right" if value > 0.0 else "left"
    return f"{label} {magnitude:.0f}deg {direction}"


def _speaker_label(name: str | None, track_id: int) -> str:
    track_label = f"person_{track_id}"
    return f"{name}/{track_label}" if name else track_label


def _mean(values: object) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _now(now: float | None) -> float:
    return time.monotonic() if now is None else float(now)
