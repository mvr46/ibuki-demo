"""Face recognition primitives ported from the reference CLI."""

from __future__ import annotations
import pickle
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".reachy-mini" / "faces.db"
DEFAULT_THRESHOLD = 0.4
IOU_SAME_FACE = 0.5


@dataclass(frozen=True)
class Person:
    """One named person and their stored exemplar embeddings."""

    name: str
    embeddings: tuple[NDArray[np.float32], ...]


class FaceDB:
    """Pickled ``dict[name, list[normed_embedding]]`` with multi-exemplar match."""

    def __init__(self, path: Path | str | None = None):
        """Initialize a persistent face database."""
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self.people: dict[str, list[NDArray[np.float32]]] = {}
        self.load()

    def load(self) -> None:
        """Load the database from disk, or start empty when absent."""
        if self.path.exists():
            with self.path.open("rb") as f:
                raw_people = pickle.load(f)
            self.people = {
                str(name): [np.asarray(embedding, dtype=np.float32) for embedding in exemplars]
                for name, exemplars in raw_people.items()
            }
            n_emb = sum(len(v) for v in self.people.values())
            logger.info("Loaded %d people (%d embeddings) from %s", len(self.people), n_emb, self.path)
            return

        self.people = {}
        logger.info("No face DB at %s; starting empty.", self.path)

    def save(self) -> None:
        """Atomically save the database to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.people, f)
        tmp.replace(self.path)

    def add(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Add one exemplar embedding for a person."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must be non-empty")
        self.people.setdefault(clean_name, []).append(_normalize_embedding(embedding))
        self.save()

    def match(self, embedding: NDArray[np.float32], threshold: float) -> tuple[Optional[str], float]:
        """Return ``(name, similarity)`` if best match >= threshold, else ``(None, best_sim)``."""
        if not self.people:
            return None, 0.0

        query = _normalize_embedding(embedding)
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, exemplars in self.people.items():
            sims = np.stack(exemplars) @ query
            sim = float(sims.max())
            if sim > best_sim:
                best_name, best_sim = name, sim

        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim

    def exemplar_count(self, name: str) -> int:
        """Return the number of saved exemplars for a person."""
        return len(self.people.get(name, []))

    def persons(self) -> list[Person]:
        """Return a structured view of the database contents."""
        return [Person(name=name, embeddings=tuple(exemplars)) for name, exemplars in self.people.items()]


def iou(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


class Tracker:
    """Greedy IoU tracker — each detection inherits the most-overlapping prior track."""

    def __init__(self) -> None:
        """Initialize empty tracking state."""
        self.tracks: list[dict] = []
        self._next_id = 0

    def step(self, detections: list[tuple[NDArray[np.float32], Optional[str]]]) -> list[dict]:
        """Return tracks aligned 1:1 with ``detections``."""
        for track in self.tracks:
            track["_matched"] = False

        result: list[dict] = []
        for det_bbox, det_name in detections:
            best_i, best_v = -1, 0.0
            for index, track in enumerate(self.tracks):
                if track["_matched"]:
                    continue
                value = iou(det_bbox, track["bbox"])
                if value > best_v:
                    best_v, best_i = value, index

            if best_v >= IOU_SAME_FACE:
                track = self.tracks[best_i]
                track["_matched"] = True
                track["bbox"] = det_bbox
                if det_name is not None:
                    track["name"] = det_name
                    track["unknown_streak"] = 0
                elif track["name"] is None and not track["skipped"]:
                    track["unknown_streak"] += 1
                result.append(track)
            else:
                track = {
                    "id": self._next_id,
                    "bbox": det_bbox,
                    "name": det_name,
                    "unknown_streak": 0 if det_name else 1,
                    "skipped": False,
                    "_matched": True,
                }
                self._next_id += 1
                result.append(track)

        self.tracks = list(result)
        return result


class FaceRecognizer:
    """Recognition arm only: landmark, align, embed. No detection."""

    def __init__(self, name: str = "buffalo_l"):
        """Load InsightFace landmark and embedding models from the local model cache."""
        from insightface import model_zoo

        models_dir = Path.home() / ".insightface" / "models" / name
        landmarker_path = models_dir / "2d106det.onnx"
        embedder_path = models_dir / "w600k_r50.onnx"
        if not landmarker_path.exists() or not embedder_path.exists():
            raise RuntimeError(
                "InsightFace buffalo_l models are not installed. Run "
                "`insightface-cli model.download buffalo_l` or run `examples/face_recognition/run.sh` once."
            )

        self._landmarker = model_zoo.get_model(str(landmarker_path))
        self._embedder = model_zoo.get_model(str(embedder_path))
        self._landmarker.prepare(ctx_id=-1)
        self._embedder.prepare(ctx_id=-1)

    def embed(self, frame_bgr: NDArray[np.uint8], bbox_xyxy: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Crop, align to 112x112 via landmarks, return a 512-D normalized embedding."""
        if frame_bgr is None or frame_bgr.ndim < 2:
            return None

        clipped_bbox = _clip_bbox_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), frame_bgr.shape)
        if clipped_bbox is None:
            return None

        landmarks = self._landmarks(frame_bgr, clipped_bbox)
        if landmarks is None:
            return None

        aligned = self._norm_crop(frame_bgr, landmarks)
        raw_embedding = self._embed(aligned)
        if raw_embedding is None:
            return None
        return _normalize_embedding(raw_embedding)

    def _landmarks(self, frame_bgr: NDArray[np.uint8], bbox_xyxy: NDArray[np.float32]) -> NDArray[np.float32] | None:
        face = _FaceContainer(bbox=bbox_xyxy)
        try:
            returned = self._landmarker.get(frame_bgr, face)
        except Exception as exc:
            logger.debug("InsightFace landmarking failed: %s", exc)
            return None

        landmarks = _first_available_landmarks(
            getattr(face, "kps", None),
            getattr(face, "landmark_2d_106", None),
            getattr(face, "landmark", None),
            returned,
        )
        if landmarks is None:
            return None
        landmarks_array = np.asarray(landmarks, dtype=np.float32)
        if landmarks_array.ndim != 2 or landmarks_array.shape[1] != 2:
            return None
        if landmarks_array.shape[0] == 5:
            return landmarks_array
        return _five_point_landmarks(landmarks_array, bbox_xyxy)

    def _norm_crop(self, frame_bgr: NDArray[np.uint8], kps: NDArray[np.float32]) -> NDArray[np.uint8]:
        from insightface.utils import face_align

        return face_align.norm_crop(frame_bgr, landmark=kps, image_size=112)

    def _embed(self, aligned_bgr: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        try:
            if hasattr(self._embedder, "get_feat"):
                embedding = self._embedder.get_feat(aligned_bgr)
            else:
                embedding = self._embedder.get(aligned_bgr)
        except Exception as exc:
            logger.debug("InsightFace embedding failed: %s", exc)
            return None

        embedding_array = np.asarray(embedding, dtype=np.float32)
        if embedding_array.ndim > 1:
            embedding_array = embedding_array.reshape(-1, embedding_array.shape[-1])[0]
        return embedding_array


def _clip_bbox_xyxy(bbox: NDArray[np.float32], frame_shape: tuple[int, ...]) -> NDArray[np.float32] | None:
    if bbox.shape != (4,):
        return None
    height, width = frame_shape[:2]
    x1 = float(np.clip(bbox[0], 0, width - 1))
    y1 = float(np.clip(bbox[1], 0, height - 1))
    x2 = float(np.clip(bbox[2], 0, width))
    y2 = float(np.clip(bbox[3], 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class _FaceContainer(dict):
    """Tiny mutable face object compatible with InsightFace landmark models."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        for key, value in kwargs.items():
            self[key] = value

    def __setattr__(self, name: str, value: object) -> None:
        dict.__setitem__(self, name, value)
        object.__setattr__(self, name, value)

    def __setitem__(self, name: str, value: object) -> None:
        dict.__setitem__(self, name, value)
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> object | None:
        return self.get(name)


def _first_available_landmarks(*candidates: object) -> object | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _five_point_landmarks(
    landmarks: NDArray[np.float32],
    bbox_xyxy: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Reduce dense landmarks to approximate InsightFace five-point landmarks."""
    x1, y1, x2, y2 = bbox_xyxy
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    x_mid = x1 + width * 0.5

    upper = landmarks[landmarks[:, 1] <= y1 + height * 0.58]
    lower = landmarks[landmarks[:, 1] > y1 + height * 0.45]

    left_eye_points = upper[upper[:, 0] < x_mid]
    right_eye_points = upper[upper[:, 0] >= x_mid]
    mouth_points = lower[lower[:, 1] >= y1 + height * 0.62]

    left_eye = _mean_or_default(left_eye_points, (x1 + width * 0.35, y1 + height * 0.40))
    right_eye = _mean_or_default(right_eye_points, (x1 + width * 0.65, y1 + height * 0.40))
    nose = landmarks[np.argmin(np.sum((landmarks - np.array([x_mid, y1 + height * 0.55])) ** 2, axis=1))]

    if len(mouth_points) >= 2:
        mouth_left = _mean_or_default(mouth_points[mouth_points[:, 0] < x_mid], (x1 + width * 0.40, y1 + height * 0.74))
        mouth_right = _mean_or_default(
            mouth_points[mouth_points[:, 0] >= x_mid],
            (x1 + width * 0.60, y1 + height * 0.74),
        )
    else:
        mouth_left = np.array([x1 + width * 0.40, y1 + height * 0.74], dtype=np.float32)
        mouth_right = np.array([x1 + width * 0.60, y1 + height * 0.74], dtype=np.float32)

    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right]).astype(np.float32)


def _mean_or_default(points: NDArray[np.float32], default: tuple[float, float]) -> NDArray[np.float32]:
    if len(points) == 0:
        return np.asarray(default, dtype=np.float32)
    return np.mean(points, axis=0).astype(np.float32)


def _normalize_embedding(embedding: NDArray[np.float32]) -> NDArray[np.float32]:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return (vector / norm).astype(np.float32)
