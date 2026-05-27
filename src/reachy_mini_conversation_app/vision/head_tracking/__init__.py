"""Head-tracking backends and process helpers."""

from typing import Protocol, TypeAlias, SupportsFloat
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


HeadTrackerResult: TypeAlias = tuple[NDArray[np.float32] | None, SupportsFloat | None]


@dataclass(frozen=True)
class HeadTrackerTarget:
    """One normalized head or face target detected in a camera frame."""

    x_offset: float
    y_offset: float
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_size: tuple[int, int]

    @property
    def center(self) -> NDArray[np.float32]:
        """Return MediaPipe-style target coordinates in [-1, 1]."""
        return np.array([self.x_offset, self.y_offset], dtype=np.float32)

    @property
    def area(self) -> float:
        """Return normalized target area."""
        return max(0.0, float(self.bbox[2])) * max(0.0, float(self.bbox[3]))


class HeadTracker(Protocol):
    """Shared interface for optional head-tracking backends."""

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""


class HeadTargetProvider(Protocol):
    """Optional interface for trackers that can return all visible targets."""

    def get_head_targets(self, img: NDArray[np.uint8]) -> list[HeadTrackerTarget]:
        """Return all detected head targets for a frame."""
