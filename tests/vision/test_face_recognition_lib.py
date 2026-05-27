"""Tests for face recognition primitives."""

from __future__ import annotations

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.face_recognition_lib import FaceDB, Tracker, FaceRecognizer


def test_face_db_matches_best_exemplar(tmp_path) -> None:
    """FaceDB should persist and match the best cosine-similarity exemplar."""
    db_path = tmp_path / "faces.db"
    db = FaceDB(db_path)
    db.add("Alice", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    db.add("Bob", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    reloaded = FaceDB(db_path)
    name, similarity = reloaded.match(np.array([0.9, 0.1, 0.0], dtype=np.float32), threshold=0.4)

    assert name == "Alice"
    assert similarity == pytest.approx(0.994, abs=0.01)
    assert reloaded.exemplar_count("Alice") == 1


def test_tracker_reuses_tracks_by_iou() -> None:
    """The IoU tracker should keep an ID for overlapping detections."""
    tracker = Tracker()
    first = tracker.step([(np.array([10, 10, 40, 40], dtype=np.float32), None)])
    second = tracker.step([(np.array([12, 10, 42, 40], dtype=np.float32), "Alice")])

    assert first[0]["id"] == second[0]["id"]
    assert second[0]["name"] == "Alice"
    assert second[0]["unknown_streak"] == 0


def test_face_recognizer_embed_aligns_and_normalizes() -> None:
    """FaceRecognizer.embed should landmark, align, and normalize embeddings."""

    class FakeLandmarker:
        def get(self, frame: np.ndarray, face: object) -> np.ndarray:
            return np.array(
                [
                    [18.0, 24.0],
                    [42.0, 24.0],
                    [30.0, 35.0],
                    [22.0, 48.0],
                    [40.0, 48.0],
                ],
                dtype=np.float32,
            )

    class FakeEmbedder:
        def get_feat(self, aligned: np.ndarray) -> np.ndarray:
            assert aligned.shape == (112, 112, 3)
            embedding = np.zeros((1, 512), dtype=np.float32)
            embedding[0, 0] = 3.0
            embedding[0, 1] = 4.0
            return embedding

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._landmarker = FakeLandmarker()
    recognizer._embedder = FakeEmbedder()
    recognizer._norm_crop = lambda _frame, _kps: np.zeros((112, 112, 3), dtype=np.uint8)

    embedding = recognizer.embed(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.array([10, 10, 50, 55], dtype=np.float32),
    )

    assert embedding is not None
    assert embedding.shape == (512,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert embedding[0] == pytest.approx(0.6)
    assert embedding[1] == pytest.approx(0.8)
