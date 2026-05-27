"""Spatial-audio speaker selection for head tracking."""

from __future__ import annotations
import json
import math
import time
import logging
import threading
from typing import Any, Callable, Protocol
from collections import deque
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.face_recognition_lib import IOU_SAME_FACE, iou


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoundTarget:
    """One normalized direction-of-arrival target from the robot daemon."""

    angle: float
    x_offset: float
    speech_detected: bool


@dataclass(frozen=True)
class SpatialAudioSample:
    """One timestamped direction-of-arrival sample."""

    timestamp: float
    angle: float
    x_offset: float
    azimuth_deg: float
    speech_detected: bool


class SpatialAudioSource(Protocol):
    """Shared source of recent robot spatial-audio samples."""

    def get_latest(self) -> tuple[SoundTarget | None, float | None]:
        """Return the latest target and monotonic timestamp."""
        ...

    def window(self, start_s: float, end_s: float) -> tuple[SpatialAudioSample, ...]:
        """Return samples whose timestamps fall inside a monotonic-time window."""
        ...


@dataclass(frozen=True)
class SoundOrientationCommand:
    """One bounded body-yaw correction from a sound target."""

    body_yaw: float
    yaw_correction: float
    drift: float
    target: SoundTarget


@dataclass
class SoundOrientationController:
    """Convert DoA x offsets into stable, bounded body-yaw offsets."""

    control_interval_seconds: float = 0.20
    deadband: float = 0.08
    recenter_threshold: float = 0.14
    stable_samples: int = 2
    max_yaw_degrees: float = 45.0
    max_step_degrees: float = 30.0
    # Body yaw follows Reachy hardware direction; it is not the head-yaw sign used by move_head.
    body_yaw_gain: float = 1.0
    smoothing: float = 0.35

    def __post_init__(self) -> None:
        """Initialize controller state."""
        self.body_yaw = 0.0
        self._smoothed_x: float | None = None
        self._stable_count = 0

    def reset_observation(self) -> None:
        """Forget observed sound drift without changing the held body yaw."""
        self._smoothed_x = None
        self._stable_count = 0

    def reset(self) -> None:
        """Return the controller to neutral."""
        self.body_yaw = 0.0
        self.reset_observation()

    def update(self, target: SoundTarget | None) -> SoundOrientationCommand | None:
        """Return a body-yaw command once sound drift is stable enough."""
        if target is None:
            self.reset_observation()
            return None

        smoothing = _clamp_score(self.smoothing)
        if self._smoothed_x is None:
            self._smoothed_x = target.x_offset
        else:
            self._smoothed_x = (1.0 - smoothing) * self._smoothed_x + smoothing * target.x_offset

        drift = abs(self._smoothed_x)
        if drift < self.deadband:
            self.reset_observation()
            return None
        if drift < self.recenter_threshold:
            return None

        self._stable_count += 1
        if self._stable_count < max(1, self.stable_samples):
            return None

        max_yaw = math.radians(self.max_yaw_degrees)
        max_step = math.radians(self.max_step_degrees)
        yaw_correction = min(max(self._smoothed_x * max_yaw * self.body_yaw_gain, -max_step), max_step)
        self.body_yaw = min(max(self.body_yaw + yaw_correction, -max_yaw), max_yaw)
        command = SoundOrientationCommand(
            body_yaw=self.body_yaw,
            yaw_correction=yaw_correction,
            drift=drift,
            target=target,
        )
        self.reset_observation()
        return command


def target_from_doa(angle: float, speech_detected: bool = False) -> SoundTarget:
    """Map Reachy DoA radians to a normalized left/right audio target."""
    x_offset = ((math.pi / 2.0) - float(angle)) / (math.pi / 2.0)
    return SoundTarget(
        angle=float(angle),
        x_offset=_clamp_unit(x_offset),
        speech_detected=bool(speech_detected),
    )


def target_from_doa_response(raw_doa: dict[str, Any] | None) -> SoundTarget | None:
    """Return a sound target from a daemon DoA response."""
    if raw_doa is None or raw_doa.get("angle") is None:
        return None
    return target_from_doa(
        angle=float(raw_doa["angle"]),
        speech_detected=bool(raw_doa.get("speech_detected", False)),
    )


@dataclass(frozen=True)
class SpeakerSelectionConfig:
    """Weights for one audio/visual speaker-selection pass."""

    audio_weight: float = 0.55
    visual_weight: float = 0.30
    continuity_weight: float = 0.15
    name_match_bonus: float = 0.75


@dataclass
class SpeakerSelectionState:
    """Temporal state that stabilizes active-speaker focus."""

    last_x_offset: float | None = None

    def remember(self, target: HeadTrackerTarget) -> None:
        """Remember the selected target position."""
        self.last_x_offset = target.x_offset


@dataclass(frozen=True)
class SpeakerSelectionResult:
    """Selected speaker target and the score components that selected it."""

    target: HeadTrackerTarget | None
    audio_agreement: float
    visual_confidence: float
    temporal_continuity: float
    name_match: bool = False


def select_speaker(
    targets: list[HeadTrackerTarget],
    *,
    audio_x_offset: float | None,
    state: SpeakerSelectionState | None = None,
    config: SpeakerSelectionConfig = SpeakerSelectionConfig(),
    prefer_name: str | None = None,
    identity_targets: list[object] | None = None,
) -> SpeakerSelectionResult:
    """Select the likely active speaker from visible face targets and audio."""
    candidates: list[tuple[float, int, HeadTrackerTarget, float, float, float, bool]] = []
    for index, target in enumerate(targets):
        audio_agreement = _audio_agreement(target.x_offset, audio_x_offset)
        visual_confidence = _clamp_score(target.confidence)
        continuity = _temporal_continuity(target.x_offset, state)
        name_match = _target_matches_name(target, prefer_name, identity_targets)
        score = (
            audio_agreement * config.audio_weight
            + visual_confidence * config.visual_weight
            + continuity * config.continuity_weight
            + (config.name_match_bonus if name_match else 0.0)
        )
        candidates.append((score, index, target, audio_agreement, visual_confidence, continuity, name_match))

    if not candidates:
        return SpeakerSelectionResult(None, 0.0, 0.0, 0.0)

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    _, _, selected, audio_agreement, visual_confidence, continuity, name_match = candidates[0]
    if state is not None:
        state.remember(selected)
    return SpeakerSelectionResult(selected, audio_agreement, visual_confidence, continuity, name_match)


class DaemonDoAPoller:
    """Poll `/api/state/doa` without blocking camera processing."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        interval_seconds: float = 0.20,
        timeout_seconds: float = 0.15,
        history_seconds: float = 30.0,
        history_maxlen: int = 600,
        reader: Callable[[str, int, float], dict[str, Any] | None] | None = None,
    ) -> None:
        """Initialize the daemon DoA poller."""
        self.host = host
        self.port = port
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.history_seconds = max(1.0, float(history_seconds))
        self._reader = reader or read_daemon_doa
        self._lock = threading.Lock()
        self._latest: SoundTarget | None = None
        self._latest_at: float | None = None
        self._history: deque[SpatialAudioSample] = deque(maxlen=max(1, int(history_maxlen)))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure_count = 0

    def start(self) -> None:
        """Start polling in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reachy-doa-poller")
        self._thread.start()

    def stop(self) -> None:
        """Stop polling."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self) -> tuple[SoundTarget | None, float | None]:
        """Return the latest target and its monotonic timestamp."""
        with self._lock:
            return self._latest, self._latest_at

    def window(self, start_s: float, end_s: float) -> tuple[SpatialAudioSample, ...]:
        """Return recent spatial-audio samples inside ``[start_s, end_s]``."""
        start = min(float(start_s), float(end_s))
        end = max(float(start_s), float(end_s))
        with self._lock:
            return tuple(sample for sample in self._history if start <= sample.timestamp <= end)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                target = target_from_doa_response(self._reader(self.host, self.port, self.timeout_seconds))
            except Exception as exc:
                self._failure_count += 1
                if self._failure_count == 1:
                    logger.debug("Robot DoA polling unavailable at %s:%s: %s", self.host, self.port, exc)
            else:
                self._failure_count = 0
                if target is not None:
                    self._record_target(target, time.monotonic())
            self._stop_event.wait(self.interval_seconds)

    def _record_target(self, target: SoundTarget, timestamp: float) -> None:
        """Record a target in latest-sample state and the bounded history."""
        sample = SpatialAudioSample(
            timestamp=float(timestamp),
            angle=target.angle,
            x_offset=target.x_offset,
            azimuth_deg=target.x_offset * 90.0,
            speech_detected=target.speech_detected,
        )
        with self._lock:
            self._latest = target
            self._latest_at = sample.timestamp
            self._history.append(sample)
            cutoff = sample.timestamp - self.history_seconds
            while self._history and self._history[0].timestamp < cutoff:
                self._history.popleft()


def read_daemon_doa(host: str, port: int, timeout_seconds: float) -> dict[str, Any] | None:
    """Read robot-side DoA from the Reachy daemon HTTP endpoint."""
    with urlopen(f"http://{host}:{port}/api/state/doa", timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    if not body or body == "null":
        return None
    value = json.loads(body)
    return value if isinstance(value, dict) else None


def build_daemon_spatial_audio_source(reachy_mini: object) -> DaemonDoAPoller | None:
    """Build the shared daemon-backed spatial audio source when host/port are available."""
    client = getattr(reachy_mini, "client", None)
    host = getattr(client, "host", None) or getattr(reachy_mini, "host", None)
    port = getattr(client, "port", None) or getattr(reachy_mini, "port", None)
    if not host or port is None:
        return None
    try:
        return DaemonDoAPoller(str(host), int(port))
    except Exception as exc:
        logger.debug("Skipping robot spatial-audio source setup: %s", exc)
        return None


def _audio_agreement(x_offset: float, audio_x_offset: float | None) -> float:
    if audio_x_offset is None:
        return 0.5
    return _clamp_score(1.0 - abs(float(x_offset) - float(audio_x_offset)) / 2.0)


def _temporal_continuity(x_offset: float, state: SpeakerSelectionState | None) -> float:
    if state is None or state.last_x_offset is None:
        return 0.0
    return _clamp_score(1.0 - abs(float(x_offset) - state.last_x_offset) / 2.0)


def _target_matches_name(
    target: HeadTrackerTarget,
    prefer_name: str | None,
    identity_targets: list[object] | None,
) -> bool:
    if not prefer_name or not identity_targets:
        return False
    preferred = prefer_name.strip().casefold()
    if not preferred:
        return False

    target_box = _target_xyxy(target)
    for identity_target in identity_targets:
        if str(getattr(identity_target, "name", "") or "").strip().casefold() != preferred:
            continue
        visible_target = getattr(identity_target, "target", None)
        if not isinstance(visible_target, HeadTrackerTarget):
            continue
        if iou(target_box, _target_xyxy(visible_target)) >= IOU_SAME_FACE:
            return True
    return False


def _target_xyxy(target: HeadTrackerTarget) -> NDArray[np.float32]:
    x, y, width, height = target.bbox
    frame_width, frame_height = target.frame_size
    return np.array(
        [
            x * frame_width,
            y * frame_height,
            (x + width) * frame_width,
            (y + height) * frame_height,
        ],
        dtype=np.float32,
    )


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
