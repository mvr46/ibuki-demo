"""Tests for face identity adapters."""

from __future__ import annotations

import numpy as np

from reachy_mini_conversation_app.vision.face_identity import FaceIdentifier, target_to_bbox_xyxy
from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerTarget


class _FakeRecognizer:
    def __init__(self) -> None:
        self.bboxes: list[np.ndarray] = []

    def embed(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        self.bboxes.append(bbox_xyxy)
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeDB:
    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        assert threshold == 0.4
        return ("Alice", 0.83)


def _target() -> HeadTrackerTarget:
    return HeadTrackerTarget(
        x_offset=0.0,
        y_offset=0.0,
        confidence=0.9,
        bbox=(0.25, 0.20, 0.50, 0.50),
        frame_size=(640, 480),
    )


def test_target_to_bbox_xyxy_uses_frame_pixels() -> None:
    """Normalized target bboxes should convert to pixel-space xyxy boxes."""
    bbox = target_to_bbox_xyxy(_target(), (480, 640, 3))

    assert bbox.tolist() == [160.0, 96.0, 480.0, 336.0]


def test_face_identifier_identifies_targets() -> None:
    """FaceIdentifier should embed target crops and attach FaceDB matches."""
    recognizer = _FakeRecognizer()
    identifier = FaceIdentifier(recognizer=recognizer, db=_FakeDB())

    identified = identifier.identify(np.zeros((480, 640, 3), dtype=np.uint8), [_target()])

    assert len(identified) == 1
    assert identified[0].name == "Alice"
    assert identified[0].similarity == 0.83
    assert recognizer.bboxes[0].tolist() == [160.0, 96.0, 480.0, 336.0]
