"""Runtime performance diagnostics for the local Reachy Mini conversation loop."""

from __future__ import annotations
import time
import threading
from dataclasses import field, dataclass


@dataclass
class PerformanceSnapshot:
    """JSON-friendly performance counters and recent latency samples."""

    daemon_rtt_ms: float | None = None
    daemon_state: str | None = None
    media_state: dict[str, object] = field(default_factory=dict)
    transport: dict[str, object] = field(default_factory=dict)
    health_checks: dict[str, object] = field(default_factory=dict)
    local_model: dict[str, object] = field(default_factory=dict)
    local_tts: dict[str, object] = field(default_factory=dict)
    voice_activity: dict[str, object] = field(default_factory=dict)
    camera_frame_age_ms: float | None = None
    camera_fps: float | None = None
    audio_input_frames: int = 0
    audio_output_frames: int = 0
    dropped_audio_frames: int = 0
    audio_queue_depth_s: float | None = None
    stt_ms: float | None = None
    llm_first_token_ms: float | None = None
    llm_total_ms: float | None = None
    tts_ms: float | None = None
    first_audio_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dashboard payload."""
        return {
            "daemon_rtt_ms": self.daemon_rtt_ms,
            "daemon_state": self.daemon_state,
            "media_state": self.media_state,
            "transport": self.transport,
            "health_checks": self.health_checks,
            "local_model": self.local_model,
            "local_tts": self.local_tts,
            "voice_activity": self.voice_activity,
            "camera_frame_age_ms": self.camera_frame_age_ms,
            "camera_fps": self.camera_fps,
            "audio_input_frames": self.audio_input_frames,
            "audio_output_frames": self.audio_output_frames,
            "dropped_audio_frames": self.dropped_audio_frames,
            "audio_queue_depth_s": self.audio_queue_depth_s,
            "stt_ms": self.stt_ms,
            "llm_first_token_ms": self.llm_first_token_ms,
            "llm_total_ms": self.llm_total_ms,
            "tts_ms": self.tts_ms,
            "first_audio_ms": self.first_audio_ms,
        }


class PerformanceDiagnostics:
    """Small thread-safe diagnostics collector for dashboard and logs."""

    def __init__(self) -> None:
        """Initialize counters."""
        self._lock = threading.Lock()
        self._snapshot = PerformanceSnapshot()
        self._camera_frame_times: list[float] = []

    def set_transport(self, **payload: object) -> None:
        """Record the active transport selection."""
        with self._lock:
            self._snapshot.transport = dict(payload)

    def set_daemon(self, *, rtt_ms: float | None = None, state: str | None = None) -> None:
        """Record daemon health."""
        with self._lock:
            self._snapshot.daemon_rtt_ms = rtt_ms
            self._snapshot.daemon_state = state

    def set_media_state(self, state: dict[str, object]) -> None:
        """Record robot media health."""
        with self._lock:
            self._snapshot.media_state = dict(state)

    def set_health_checks(self, checks: dict[str, object]) -> None:
        """Record robot health-check results."""
        with self._lock:
            self._snapshot.health_checks = dict(checks)

    def set_local_model(self, **payload: object) -> None:
        """Record local model capability diagnostics."""
        with self._lock:
            merged = dict(self._snapshot.local_model)
            merged.update(payload)
            self._snapshot.local_model = merged

    def set_local_tts(self, **payload: object) -> None:
        """Record local Piper TTS readiness diagnostics."""
        with self._lock:
            merged = dict(self._snapshot.local_tts)
            merged.update(payload)
            self._snapshot.local_tts = merged

    def set_voice_activity(self, **payload: object) -> None:
        """Record local VAD and robot-noise suppression diagnostics."""
        with self._lock:
            merged = dict(self._snapshot.voice_activity)
            merged.update(payload)
            self._snapshot.voice_activity = merged

    def record_rejected_segment(self, *, reason: str, source: str = "vad", **payload: object) -> None:
        """Count and annotate a rejected local voice segment."""
        with self._lock:
            merged = dict(self._snapshot.voice_activity)
            previous_count = merged.get("rejected_segment_count") or 0
            try:
                count = (
                    int(previous_count)
                    if isinstance(previous_count, (str, bytes, bytearray, int, float))
                    else 0
                )
            except (TypeError, ValueError):
                count = 0
            merged["rejected_segment_count"] = count + 1
            merged["last_reject_reason"] = reason
            if source == "stt":
                merged["last_stt_reject_reason"] = reason
            else:
                merged["last_vad_reject_reason"] = reason
            merged.update(payload)
            self._snapshot.voice_activity = merged

    def record_camera_frame(self) -> None:
        """Record that a camera frame was received."""
        now = time.monotonic()
        with self._lock:
            self._camera_frame_times.append(now)
            cutoff = now - 5.0
            self._camera_frame_times = [item for item in self._camera_frame_times if item >= cutoff]
            self._snapshot.camera_frame_age_ms = 0.0
            if len(self._camera_frame_times) >= 2:
                elapsed = self._camera_frame_times[-1] - self._camera_frame_times[0]
                self._snapshot.camera_fps = (len(self._camera_frame_times) - 1) / elapsed if elapsed > 0 else None

    def record_audio_input_frame(self) -> None:
        """Record an input audio frame forwarded to the backend."""
        with self._lock:
            self._snapshot.audio_input_frames += 1

    def record_audio_output_frame(self, *, queue_depth_s: float | None = None) -> None:
        """Record an output audio frame pushed to the robot."""
        with self._lock:
            self._snapshot.audio_output_frames += 1
            self._snapshot.audio_queue_depth_s = queue_depth_s

    def record_dropped_audio_frame(self) -> None:
        """Record an input audio frame dropped because no backend was ready."""
        with self._lock:
            self._snapshot.dropped_audio_frames += 1

    def record_turn_metrics(
        self,
        *,
        stt_ms: float | None = None,
        llm_first_token_ms: float | None = None,
        llm_total_ms: float | None = None,
        tts_ms: float | None = None,
        first_audio_ms: float | None = None,
    ) -> None:
        """Record the latest local voice turn timings."""
        with self._lock:
            if stt_ms is not None:
                self._snapshot.stt_ms = stt_ms
            if llm_first_token_ms is not None:
                self._snapshot.llm_first_token_ms = llm_first_token_ms
            if llm_total_ms is not None:
                self._snapshot.llm_total_ms = llm_total_ms
            if tts_ms is not None:
                self._snapshot.tts_ms = tts_ms
            if first_audio_ms is not None:
                self._snapshot.first_audio_ms = first_audio_ms

    def snapshot(self) -> dict[str, object]:
        """Return the latest diagnostics payload."""
        with self._lock:
            snap = PerformanceSnapshot(**self._snapshot.__dict__)
            if self._camera_frame_times:
                snap.camera_frame_age_ms = (time.monotonic() - self._camera_frame_times[-1]) * 1000
            return snap.to_dict()
