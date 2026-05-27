"""Background face identity worker and perception state."""

from __future__ import annotations
import time
import logging
import threading
from typing import Literal
from collections import deque
from dataclasses import field, dataclass

import numpy as np

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.vision.face_identity import (
    FaceIdentifier,
    IdentifiedTarget,
    with_seen_times,
    target_to_bbox_xyxy,
    get_head_targets_from_camera,
)
from reachy_mini_conversation_app.vision.face_recognition_lib import Tracker


logger = logging.getLogger(__name__)

VisionEventKind = Literal["entered", "left", "named"]


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


class FaceIdentifierWorker:
    """Poll camera frames and update thread-safe face identity perception state."""

    def __init__(
        self,
        camera_worker: object,
        identifier: FaceIdentifier,
        *,
        rate_hz: float = 2.5,
        camera_horizontal_fov_deg: float | None = None,
        visual_history_seconds: float = 30.0,
        visual_history_maxlen: int = 600,
    ) -> None:
        """Initialize the worker."""
        self.camera_worker = camera_worker
        self.identifier = identifier
        self.rate_hz = max(0.1, float(rate_hz))
        self.camera_horizontal_fov_deg = (
            float(camera_horizontal_fov_deg)
            if camera_horizontal_fov_deg is not None
            else float(config.REACHY_CAMERA_HORIZONTAL_FOV_DEG)
        )
        self.visual_history_seconds = max(1.0, float(visual_history_seconds))
        self._state = PerceptionState()
        self._state.visual_history = deque(maxlen=max(1, int(visual_history_maxlen)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker = Tracker()
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

    def identify(self, frame_bgr: np.ndarray, targets: list[object]) -> list[IdentifiedTarget]:
        """Identify targets synchronously using the owned identifier."""
        return self.identifier.identify(frame_bgr, targets)  # type: ignore[arg-type]

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
            )
            self._state.visible[match_index] = renamed
            self._track_names[track_id] = clean_name
            self._track_last_seen[track_id] = now
            self._track_positions[track_id] = position
            self._state.last_seen[clean_name] = now
            self._state.last_positions[clean_name] = position
            self._state.events.append(VisionEvent("named", clean_name, position, now))

            for track in self._tracker.tracks:
                if int(track.get("id", -1)) == track_id:
                    track["name"] = clean_name
                    track["unknown_streak"] = 0
                    break

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
        identified = self.identifier.identify(frame, targets) if targets else []
        self._update_state(identified, current_time)

    def _update_state(self, identified: list[IdentifiedTarget], current_time: float) -> None:
        previous_ids = set(self._track_names)
        previous_names = dict(self._track_names)
        previous_last_seen = dict(self._track_last_seen)
        previous_positions = dict(self._track_positions)
        detections = [
            (target_to_bbox_xyxy(item.target, item.target.frame_size[::-1] + (3,)), item.name) for item in identified
        ]
        tracks = self._tracker.step(detections)
        current_ids = {int(track["id"]) for track in tracks}

        events: list[VisionEvent] = []
        visible: list[IdentifiedTarget] = []
        next_names: dict[int, str | None] = {}
        next_first_seen: dict[int, float] = {}
        next_last_seen: dict[int, float] = {}
        next_positions: dict[int, str] = {}

        for item, track in zip(identified, tracks):
            track_id = int(track["id"])
            current_name = track.get("name")
            previous_name = previous_names.get(track_id)
            position = position_label(item.target.x_offset)
            first_seen_at = self._track_first_seen.get(track_id, current_time)

            if track_id not in previous_ids:
                events.append(VisionEvent("entered", current_name, position, current_time))
            elif previous_name is None and current_name is not None:
                events.append(VisionEvent("named", current_name, position, current_time))

            next_names[track_id] = current_name
            next_first_seen[track_id] = first_seen_at
            next_last_seen[track_id] = current_time
            next_positions[track_id] = position
            if current_name is not None:
                track["name"] = current_name
            visible.append(
                with_seen_times(
                    item,
                    first_seen_at=first_seen_at,
                    last_seen_at=current_time,
                    track_id=track_id,
                )
            )

        for track_id in sorted(previous_ids - current_ids):
            name = previous_names.get(track_id)
            last_seen_at = previous_last_seen.get(track_id, current_time)
            position = previous_positions.get(track_id, "center")
            events.append(VisionEvent("left", name, position, current_time, last_seen_at=last_seen_at))

        with self._lock:
            self._track_names = next_names
            self._track_first_seen = next_first_seen
            self._track_last_seen = next_last_seen
            self._track_positions = next_positions
            self._state.visible = visible
            self._record_visual_observations_locked(visible, current_time)
            for item in visible:
                if item.name is not None:
                    self._state.last_seen[item.name] = current_time
                    self._state.last_positions[item.name] = position_label(item.target.x_offset)
            for event in events:
                if event.kind == "left" and event.name is not None and event.last_seen_at is not None:
                    self._state.last_seen[event.name] = event.last_seen_at
                    self._state.last_positions[event.name] = event.position
                self._state.events.append(event)

    def _record_visual_observations_locked(self, visible: list[IdentifiedTarget], current_time: float) -> None:
        """Append visible-track observations. Caller must hold ``self._lock``."""
        cutoff = current_time - self.visual_history_seconds
        while self._state.visual_history and self._state.visual_history[0].timestamp < cutoff:
            self._state.visual_history.popleft()

        for item in visible:
            if item.track_id is None:
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


def _copy_identified(target: IdentifiedTarget) -> IdentifiedTarget:
    return IdentifiedTarget(
        target=target.target,
        name=target.name,
        similarity=target.similarity,
        embedding=target.embedding.copy(),
        first_seen_at=target.first_seen_at,
        last_seen_at=target.last_seen_at,
        track_id=target.track_id,
    )
