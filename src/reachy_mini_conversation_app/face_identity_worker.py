"""Background face identity worker and perception state."""

from __future__ import annotations
import time
import logging
import threading
from typing import Any, Literal, Protocol, Sequence
from collections import deque
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.vision.face_identity import (
    FaceObservation,
    IdentifiedTarget,
    target_to_bbox_xyxy,
    get_head_targets_from_camera,
)
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.face_recognition_lib import iou


logger = logging.getLogger(__name__)

VisionEventKind = Literal["entered", "left", "named"]
DEFAULT_TRACK_IOU_THRESHOLD = 0.30
DEFAULT_CONFIRMATION_OBSERVATIONS = 2
DEFAULT_MISSING_HOLD_SECONDS = 1.6
DEFAULT_NAME_CONFIRMATION_OBSERVATIONS = 2


@dataclass(frozen=True)
class VisionEvent:
    """One face-identity perception event."""

    kind: VisionEventKind
    name: str | None
    position: str
    timestamp: float
    last_seen_at: float | None = None


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Thread-safe copy of current perception state."""

    visible: tuple[IdentifiedTarget, ...] = ()
    last_seen: dict[str, float] = field(default_factory=dict)
    last_positions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibleTrackObservation:
    """One timestamped visible-track observation for multimodal attribution."""

    track_id: int
    name: str | None
    x_offset: float
    visual_bearing_deg: float
    bbox: tuple[float, float, float, float]
    confidence: float
    timestamp: float


@dataclass
class PerceptionState:
    """Mutable face perception state guarded by a worker lock."""

    visible: list[IdentifiedTarget] = field(default_factory=list)
    last_seen: dict[str, float] = field(default_factory=dict)
    last_positions: dict[str, str] = field(default_factory=dict)
    events: deque[VisionEvent] = field(default_factory=deque)
    visual_history: deque[VisibleTrackObservation] = field(default_factory=deque)


@dataclass
class _StableFaceTrack:
    """Temporal face track used to smooth noisy recognition frames."""

    track_id: int
    target: HeadTrackerTarget
    first_seen_at: float
    last_observed_at: float
    observations: int = 0
    confirmed: bool = False
    stable_name: str | None = None
    stable_similarity: float = 0.0
    embedding: NDArray[np.float32] | None = None
    candidate_name: str | None = None
    candidate_count: int = 0
    candidate_similarity: float = 0.0
    observed_this_pass: bool = False


class FaceIdentityIdentifier(Protocol):
    """Face identifier surface used by the background worker."""

    db: Any
    recognition_available: bool

    def identify(
        self,
        frame_bgr: NDArray[np.uint8],
        targets: list[HeadTrackerTarget],
    ) -> list[IdentifiedTarget]:
        """Identify detector targets in a camera frame."""
        ...


class FaceIdentifierWorker:
    """Poll camera frames and update thread-safe face identity perception state."""

    def __init__(
        self,
        camera_worker: object,
        identifier: FaceIdentityIdentifier,
        *,
        rate_hz: float = 2.5,
        camera_horizontal_fov_deg: float | None = None,
        visual_history_seconds: float = 30.0,
        visual_history_maxlen: int = 600,
        tracking_iou_threshold: float = DEFAULT_TRACK_IOU_THRESHOLD,
        confirmation_observations: int = DEFAULT_CONFIRMATION_OBSERVATIONS,
        missing_hold_seconds: float = DEFAULT_MISSING_HOLD_SECONDS,
        name_confirmation_observations: int = DEFAULT_NAME_CONFIRMATION_OBSERVATIONS,
        require_embedding_to_confirm: bool = True,
    ) -> None:
        """Initialize the worker."""
        self.camera_worker = camera_worker
        self.identifier = identifier
        self.recognition_available = bool(getattr(identifier, "recognition_available", True))
        self.require_embedding_to_confirm = bool(require_embedding_to_confirm)
        self.rate_hz = max(0.1, float(rate_hz))
        self.camera_horizontal_fov_deg = (
            float(camera_horizontal_fov_deg)
            if camera_horizontal_fov_deg is not None
            else float(config.REACHY_CAMERA_HORIZONTAL_FOV_DEG)
        )
        self.visual_history_seconds = max(1.0, float(visual_history_seconds))
        self.tracking_iou_threshold = max(0.0, min(1.0, float(tracking_iou_threshold)))
        self.confirmation_observations = max(1, int(confirmation_observations))
        self.missing_hold_seconds = max(0.0, float(missing_hold_seconds))
        self.name_confirmation_observations = max(1, int(name_confirmation_observations))
        self._state = PerceptionState()
        self._state.visual_history = deque(maxlen=max(1, int(visual_history_maxlen)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracks: dict[int, _StableFaceTrack] = {}
        self._next_track_id = 0
        self._track_names: dict[int, str | None] = {}
        self._track_first_seen: dict[int, float] = {}
        self._track_last_seen: dict[int, float] = {}
        self._track_positions: dict[int, str] = {}

    def start(self) -> None:
        """Start polling in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="face-identity-worker")
        self._thread.start()

    def stop(self) -> None:
        """Stop polling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def identify(self, frame_bgr: NDArray[np.uint8], targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
        """Identify targets synchronously using the owned identifier."""
        return self.identifier.identify(frame_bgr, targets)

    def snapshot(self) -> PerceptionSnapshot:
        """Return a thread-safe snapshot of visible people and last-seen metadata."""
        with self._lock:
            return PerceptionSnapshot(
                visible=tuple(_copy_identified(target) for target in self._state.visible),
                last_seen=dict(self._state.last_seen),
                last_positions=dict(self._state.last_positions),
            )

    def drain_events(self) -> list[VisionEvent]:
        """Drain and return queued perception events."""
        with self._lock:
            events = list(self._state.events)
            self._state.events.clear()
        return events

    def visual_window(self, start_s: float, end_s: float) -> tuple[VisibleTrackObservation, ...]:
        """Return visible-track observations inside ``[start_s, end_s]``."""
        start = min(float(start_s), float(end_s))
        end = max(float(start_s), float(end_s))
        with self._lock:
            return tuple(item for item in self._state.visual_history if start <= item.timestamp <= end)

    def remember_visible(self, track_id: int, name: str) -> dict[str, object]:
        """Save a currently visible tracked face under ``name`` and update state."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must be a non-empty string")

        with self._lock:
            match_index = next(
                (index for index, item in enumerate(self._state.visible) if item.track_id == track_id),
                None,
            )
            if match_index is None:
                raise KeyError(f"visible face track_id={track_id} not found")

            current = self._state.visible[match_index]
            if current.embedding is None or not current.can_remember:
                raise ValueError(f"visible face track_id={track_id} has no usable face embedding yet")
            self.identifier.db.add(clean_name, current.embedding)
            exemplar_count = self.identifier.db.exemplar_count(clean_name)
            now = time.monotonic()
            position = position_label(current.target.x_offset)
            renamed = IdentifiedTarget(
                target=current.target,
                name=clean_name,
                similarity=1.0,
                embedding=current.embedding,
                first_seen_at=current.first_seen_at,
                last_seen_at=now,
                track_id=current.track_id,
                observed=current.observed,
                held=current.held,
                stability=current.stability,
                can_remember=True,
                last_observed_at=current.last_observed_at,
            )
            self._state.visible[match_index] = renamed
            track = self._tracks.get(track_id)
            if track is not None:
                track.stable_name = clean_name
                track.stable_similarity = 1.0
                track.embedding = current.embedding.copy()
                track.candidate_name = None
                track.candidate_count = 0
                track.candidate_similarity = 0.0
                track.confirmed = True
            self._track_names[track_id] = clean_name
            self._track_last_seen[track_id] = now
            self._track_positions[track_id] = position
            self._state.last_seen[clean_name] = now
            self._state.last_positions[clean_name] = position
            self._state.events.append(VisionEvent("named", clean_name, position, now))

        logger.info("Remembered visible face track_id=%s as %s (%d exemplar(s))", track_id, clean_name, exemplar_count)
        return {
            "status": "remembered",
            "track_id": track_id,
            "name": clean_name,
            "exemplar_count": exemplar_count,
            "x_offset": round(float(current.target.x_offset), 3),
            "y_offset": round(float(current.target.y_offset), 3),
        }

    def _loop(self) -> None:
        interval = 1.0 / self.rate_hz
        while not self._stop.is_set():
            self._process_once(time.monotonic())
            self._stop.wait(interval)

    def _process_once(self, current_time: float) -> None:
        frame = self.camera_worker.get_latest_frame()  # type: ignore[attr-defined]
        if frame is None:
            self._update_state([], current_time)
            return
        targets = get_head_targets_from_camera(self.camera_worker, frame)
        observations = self._observe(frame, targets) if targets else []
        self._update_state(observations, current_time)

    def _observe(self, frame: NDArray[np.uint8], targets: list[HeadTrackerTarget]) -> list[FaceObservation]:
        observe = getattr(self.identifier, "observe", None)
        if callable(observe):
            return list(observe(frame, targets))
        return [_observation_from_identified(item) for item in self.identifier.identify(frame, targets)]

    def _update_state(
        self,
        observations: Sequence[FaceObservation | IdentifiedTarget],
        current_time: float,
    ) -> None:
        normalized = [_as_observation(item) for item in observations]
        for track in self._tracks.values():
            track.observed_this_pass = False

        events: list[VisionEvent] = []
        matched_track_ids: set[int] = set()
        for observation in normalized:
            matched_track = self._match_track(observation, matched_track_ids)
            track = matched_track if matched_track is not None else self._create_track(observation, current_time)
            matched_track_ids.add(track.track_id)
            named = self._record_observation(track, observation, current_time)
            if named is not None:
                events.append(VisionEvent("named", named, position_label(track.target.x_offset), current_time))

        visible: list[IdentifiedTarget] = []
        expired_track_ids: list[int] = []
        for track_id, track in sorted(self._tracks.items()):
            if not track.confirmed and self._should_confirm(track):
                self._confirm_track(track)
                events.append(
                    VisionEvent("entered", track.stable_name, position_label(track.target.x_offset), current_time)
                )

            if track.confirmed:
                if current_time - track.last_observed_at <= self.missing_hold_seconds:
                    visible.append(self._visible_target(track, current_time))
                    continue
                events.append(
                    VisionEvent(
                        "left",
                        track.stable_name,
                        position_label(track.target.x_offset),
                        current_time,
                        last_seen_at=track.last_observed_at,
                    )
                )
                expired_track_ids.append(track_id)
                continue

            if current_time - track.last_observed_at > self.missing_hold_seconds:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            self._tracks.pop(track_id, None)

        with self._lock:
            self._state.visible = visible
            self._record_visual_observations_locked(visible, current_time)
            for item in visible:
                if item.name is not None and item.observed:
                    seen_at = item.last_observed_at if item.last_observed_at is not None else current_time
                    self._state.last_seen[item.name] = seen_at
                    self._state.last_positions[item.name] = position_label(item.target.x_offset)
            for event in events:
                if event.kind == "left" and event.name is not None and event.last_seen_at is not None:
                    self._state.last_seen[event.name] = event.last_seen_at
                    self._state.last_positions[event.name] = event.position
                self._state.events.append(event)
            self._refresh_track_metadata_locked(visible)

    def _match_track(
        self,
        observation: FaceObservation,
        matched_track_ids: set[int],
    ) -> _StableFaceTrack | None:
        """Return the best unmatched track for an observation."""
        observation_box = _target_bbox(observation.target)
        best_track: _StableFaceTrack | None = None
        best_score = 0.0
        for track in self._tracks.values():
            if track.track_id in matched_track_ids:
                continue
            score = iou(observation_box, _target_bbox(track.target))
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is None or best_score < self.tracking_iou_threshold:
            return None
        return best_track

    def _create_track(self, observation: FaceObservation, current_time: float) -> _StableFaceTrack:
        """Create a new unconfirmed temporal track."""
        track = _StableFaceTrack(
            track_id=self._next_track_id,
            target=observation.target,
            first_seen_at=current_time,
            last_observed_at=current_time,
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def _record_observation(
        self,
        track: _StableFaceTrack,
        observation: FaceObservation,
        current_time: float,
    ) -> str | None:
        """Update a temporal track from one detector observation."""
        track.target = observation.target
        track.last_observed_at = current_time
        track.observations += 1
        track.observed_this_pass = True

        if observation.embedding is not None:
            track.embedding = observation.embedding.copy()
            if track.stable_name is None:
                track.stable_similarity = float(observation.similarity)

        if observation.name is None:
            return None
        return self._record_name_evidence(track, observation.name, float(observation.similarity))

    def _record_name_evidence(self, track: _StableFaceTrack, name: str, similarity: float) -> str | None:
        """Record repeated identity evidence and return a promoted name, if any."""
        if name == track.stable_name:
            track.stable_similarity = max(track.stable_similarity, similarity)
            track.candidate_name = None
            track.candidate_count = 0
            track.candidate_similarity = 0.0
            return None

        if track.candidate_name == name:
            track.candidate_count += 1
            track.candidate_similarity = max(track.candidate_similarity, similarity)
        else:
            track.candidate_name = name
            track.candidate_count = 1
            track.candidate_similarity = similarity

        if not track.confirmed or track.candidate_count < self.name_confirmation_observations:
            return None

        track.stable_name = name
        track.stable_similarity = track.candidate_similarity
        track.candidate_name = None
        track.candidate_count = 0
        track.candidate_similarity = 0.0
        return name

    def _should_confirm(self, track: _StableFaceTrack) -> bool:
        """Return whether a candidate track is stable enough to expose."""
        if track.observations < self.confirmation_observations:
            return False
        return not self.require_embedding_to_confirm or track.embedding is not None

    def _confirm_track(self, track: _StableFaceTrack) -> None:
        """Expose a track and seed its best initial identity guess."""
        track.confirmed = True
        if track.stable_name is None and track.candidate_name is not None:
            track.stable_name = track.candidate_name
            track.stable_similarity = track.candidate_similarity
            track.candidate_name = None
            track.candidate_count = 0
            track.candidate_similarity = 0.0

    def _visible_target(self, track: _StableFaceTrack, current_time: float) -> IdentifiedTarget:
        """Build a snapshot target from a stable track."""
        observed = track.observed_this_pass
        age = max(0.0, current_time - track.last_observed_at)
        if observed or self.missing_hold_seconds <= 0.0:
            stability = 1.0
        else:
            stability = max(0.0, 1.0 - age / self.missing_hold_seconds)
        return IdentifiedTarget(
            target=track.target,
            name=track.stable_name,
            similarity=track.stable_similarity,
            embedding=None if track.embedding is None else track.embedding.copy(),
            first_seen_at=track.first_seen_at,
            last_seen_at=current_time,
            track_id=track.track_id,
            observed=observed,
            held=not observed,
            stability=stability,
            can_remember=track.embedding is not None,
            last_observed_at=track.last_observed_at,
        )

    def _refresh_track_metadata_locked(self, visible: list[IdentifiedTarget]) -> None:
        """Update legacy track metadata dictionaries from the stabilized snapshot."""
        self._track_names = {int(item.track_id): item.name for item in visible if item.track_id is not None}
        self._track_first_seen = {
            int(item.track_id): item.first_seen_at
            for item in visible
            if item.track_id is not None and item.first_seen_at is not None
        }
        self._track_last_seen = {
            int(item.track_id): _last_seen_time(item)
            for item in visible
            if item.track_id is not None and (item.last_observed_at is not None or item.last_seen_at is not None)
        }
        self._track_positions = {
            int(item.track_id): position_label(item.target.x_offset) for item in visible if item.track_id is not None
        }

    def _record_visual_observations_locked(self, visible: list[IdentifiedTarget], current_time: float) -> None:
        """Append visible-track observations. Caller must hold ``self._lock``."""
        cutoff = current_time - self.visual_history_seconds
        while self._state.visual_history and self._state.visual_history[0].timestamp < cutoff:
            self._state.visual_history.popleft()

        for item in visible:
            if item.track_id is None or not item.observed:
                continue
            target = item.target
            x, y, width, height = target.bbox
            self._state.visual_history.append(
                VisibleTrackObservation(
                    track_id=int(item.track_id),
                    name=item.name,
                    x_offset=float(target.x_offset),
                    visual_bearing_deg=float(target.x_offset) * (self.camera_horizontal_fov_deg / 2.0),
                    bbox=(float(x), float(y), float(width), float(height)),
                    confidence=float(target.confidence),
                    timestamp=float(current_time),
                )
            )


def position_label(x_offset: float) -> str:
    """Return a compact left/center/right label for an image-space x offset."""
    if x_offset <= -0.33:
        return "left"
    if x_offset >= 0.33:
        return "right"
    return "center"


def _as_observation(item: FaceObservation | IdentifiedTarget) -> FaceObservation:
    if isinstance(item, FaceObservation):
        return item
    return _observation_from_identified(item)


def _observation_from_identified(item: IdentifiedTarget) -> FaceObservation:
    return FaceObservation(
        target=item.target,
        name=item.name,
        similarity=item.similarity,
        embedding=None if item.embedding is None else item.embedding.copy(),
    )


def _target_bbox(target: HeadTrackerTarget) -> NDArray[np.float32]:
    return target_to_bbox_xyxy(target, target.frame_size[::-1] + (3,))


def _last_seen_time(item: IdentifiedTarget) -> float:
    if item.last_observed_at is not None:
        return item.last_observed_at
    if item.last_seen_at is not None:
        return item.last_seen_at
    return 0.0


def _copy_identified(target: IdentifiedTarget) -> IdentifiedTarget:
    return IdentifiedTarget(
        target=target.target,
        name=target.name,
        similarity=target.similarity,
        embedding=None if target.embedding is None else target.embedding.copy(),
        first_seen_at=target.first_seen_at,
        last_seen_at=target.last_seen_at,
        track_id=target.track_id,
        observed=target.observed,
        held=target.held,
        stability=target.stability,
        can_remember=target.can_remember,
        last_observed_at=target.last_observed_at,
    )
