from __future__ import annotations
import logging
from typing import Protocol, cast
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerResult, HeadTrackerTarget


try:
    from supervision import Detections
    from ultralytics import YOLO  # type: ignore[attr-defined]
except ImportError as e:
    raise ImportError(
        "To use YOLO head tracker, please install the extra dependencies: pip install '.[yolo_vision]'",
    ) from e
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


class _YoloModel(Protocol):
    """Minimal YOLO model interface used by the head tracker."""

    def __call__(self, source: NDArray[np.uint8], **kwargs: object) -> Sequence[object]: ...

    def to(self, device: str) -> _YoloModel: ...


class YoloHeadTracker:
    """Lightweight head tracker using YOLO for face detection."""

    def __init__(
        self,
        model_repo: str = "AdamCodd/YOLOv11n-face-detection",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLO-based head tracker."""
        self.confidence_threshold = confidence_threshold

        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = cast(_YoloModel, YOLO(model_path).to(device))
            logger.info("YOLO face detection model loaded from %s", model_repo)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

    def _select_best_target(self, targets: list[HeadTrackerTarget]) -> HeadTrackerTarget | None:
        """Select the best target based on confidence and area."""
        if not targets:
            return None

        max_area = max(target.area for target in targets) or 1.0
        return max(targets, key=lambda target: target.confidence * 0.7 + (target.area / max_area) * 0.3)

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """Convert bounding box center to MediaPipe-style coordinates [-1, 1]."""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0

        return np.array([norm_x, norm_y], dtype=np.float32)

    def _bbox_to_target(
        self,
        bbox: NDArray[np.float32],
        confidence: float,
        w: int,
        h: int,
    ) -> HeadTrackerTarget:
        """Convert a pixel-space YOLO box to a normalized speaker target."""
        center = self._bbox_to_mp_coords(bbox, w, h)
        x1 = max(0.0, min(1.0, float(bbox[0]) / max(1, w)))
        y1 = max(0.0, min(1.0, float(bbox[1]) / max(1, h)))
        x2 = max(0.0, min(1.0, float(bbox[2]) / max(1, w)))
        y2 = max(0.0, min(1.0, float(bbox[3]) / max(1, h)))
        return HeadTrackerTarget(
            x_offset=float(center[0]),
            y_offset=float(center[1]),
            confidence=float(confidence),
            bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
            frame_size=(w, h),
        )

    def get_head_targets(self, img: NDArray[np.uint8]) -> list[HeadTrackerTarget]:
        """Return all detected face targets above the confidence threshold."""
        h, w = img.shape[:2]

        try:
            results = self.model(img, verbose=False)
            detections = Detections.from_ultralytics(results[0])
            if detections.xyxy.shape[0] == 0 or detections.confidence is None:
                logger.debug("No face targets detected")
                return []

            targets: list[HeadTrackerTarget] = []
            for bbox, confidence in zip(detections.xyxy, detections.confidence):
                if confidence < self.confidence_threshold:
                    continue
                targets.append(self._bbox_to_target(bbox, float(confidence), w, h))
            return targets
        except Exception as e:
            logger.error("Error in head target detection: %s", e)
            return []

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Get head position from face detection."""
        try:
            target = self._select_best_target(self.get_head_targets(img))
            if target is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            logger.debug("Face detected with confidence: %.2f", target.confidence)
            return target.center, 0.0

        except Exception as e:
            logger.error("Error in head position detection: %s", e)
            return None, None
