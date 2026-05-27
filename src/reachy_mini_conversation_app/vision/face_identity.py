"""Adapters between YOLO face targets and persistent face identities."""

from __future__ import annotations
import time
import logging
from typing import Any
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget
from reachy_mini_conversation_app.vision.face_recognition_lib import (
    DEFAULT_THRESHOLD,
    FaceDB,
    FaceRecognizer,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentifiedTarget:
    """One visible face target enriched with identity information."""

    target: HeadTrackerTarget
    name: str | None
    similarity: float
    embedding: NDArray[np.float32]
    first_seen_at: float | None = None
    last_seen_at: float | None = None


class FaceIdentifier:
    """Identify YOLO face targets using a recognizer and persistent FaceDB."""

    def __init__(self, recognizer: FaceRecognizer, db: FaceDB, threshold: float = DEFAULT_THRESHOLD):
        """Initialize the identifier."""
        self.recognizer = recognizer
        self.db = db
        self.threshold = threshold

    def identify(
        self,
        frame_bgr: NDArray[np.uint8],
        targets: list[HeadTrackerTarget],
    ) -> list[IdentifiedTarget]:
        """Return identity-enriched targets for all recognizer-readable faces."""
        identified: list[IdentifiedTarget] = []
        for target in targets:
            bbox_xyxy = target_to_bbox_xyxy(target, frame_bgr.shape)
            embedding = self.recognizer.embed(frame_bgr, bbox_xyxy)
            if embedding is None:
                continue
            name, similarity = self.db.match(embedding, self.threshold)
            identified.append(
                IdentifiedTarget(
                    target=target,
                    name=name,
                    similarity=similarity,
                    embedding=np.asarray(embedding, dtype=np.float32),
                )
            )
        return identified


class FaceIdentityService:
    """Synchronous face-identity service used before the background worker is started."""

    def __init__(self, identifier: FaceIdentifier):
        """Initialize the service."""
        self.identifier = identifier

    def identify(self, frame_bgr: NDArray[np.uint8], targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
        """Identify a set of targets in one frame."""
        now = time.monotonic()
        return [
            with_seen_times(target, first_seen_at=now, last_seen_at=now)
            for target in self.identifier.identify(frame_bgr, targets)
        ]

    def start(self) -> None:
        """No-op lifecycle method for parity with the background worker."""

    def stop(self) -> None:
        """No-op lifecycle method for parity with the background worker."""


def build_default_face_identity_service() -> FaceIdentityService:
    """Build the default face-identity service using the per-user FaceDB path."""
    db = FaceDB()
    recognizer = FaceRecognizer()
    return FaceIdentityService(FaceIdentifier(recognizer=recognizer, db=db))


def get_head_targets_from_camera(camera_worker: Any, frame_bgr: NDArray[np.uint8]) -> list[HeadTrackerTarget]:
    """Return all current head targets from the camera worker's tracker, if supported."""
    head_tracker = getattr(camera_worker, "head_tracker", None)
    get_targets = getattr(head_tracker, "get_head_targets", None)
    if not callable(get_targets):
        return []
    try:
        targets = get_targets(frame_bgr)
    except Exception as exc:
        logger.error("Face identity target detection failed: %s", exc)
        return []
    return [target for target in targets if isinstance(target, HeadTrackerTarget)]


def identify_from_camera(camera_worker: Any, identity_worker: Any) -> list[IdentifiedTarget]:
    """Identify visible camera targets synchronously with a service or worker."""
    frame = camera_worker.get_latest_frame()
    if frame is None:
        return []

    targets = get_head_targets_from_camera(camera_worker, frame)
    if not targets:
        return []

    identify = getattr(identity_worker, "identify", None)
    if callable(identify):
        return list(identify(frame, targets))

    identifier = getattr(identity_worker, "identifier", None)
    if identifier is None:
        return []
    return list(identifier.identify(frame, targets))


def target_to_bbox_xyxy(target: HeadTrackerTarget, frame_shape: tuple[int, ...]) -> NDArray[np.float32]:
    """Convert a normalized ``HeadTrackerTarget`` bbox into clipped pixel ``xyxy``."""
    frame_h, frame_w = frame_shape[:2]
    target_w, target_h = target.frame_size
    width = max(1, int(target_w or frame_w))
    height = max(1, int(target_h or frame_h))
    scale_x = frame_w / width
    scale_y = frame_h / height

    x, y, w, h = target.bbox
    x1 = max(0.0, min(float(frame_w - 1), float(x) * width * scale_x))
    y1 = max(0.0, min(float(frame_h - 1), float(y) * height * scale_y))
    x2 = max(0.0, min(float(frame_w), float(x + w) * width * scale_x))
    y2 = max(0.0, min(float(frame_h), float(y + h) * height * scale_y))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def with_seen_times(
    identified: IdentifiedTarget,
    *,
    first_seen_at: float | None,
    last_seen_at: float | None,
) -> IdentifiedTarget:
    """Return an identified target with presence timestamps attached."""
    return IdentifiedTarget(
        target=identified.target,
        name=identified.name,
        similarity=identified.similarity,
        embedding=identified.embedding,
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
    )
