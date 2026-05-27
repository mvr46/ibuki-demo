"""Webcam face recognition with a persistent name database.

Pipeline per frame:
1. Capture frame from webcam, mirror it.
2. Run SCRFD detection + ResNet50 recognition via insightface.FaceAnalysis.
3. For each face, look up its 512-D normed embedding in the on-disk DB
   (multi-exemplar cosine match, threshold configurable).
4. Track faces across frames by bbox IoU so we don't re-prompt on flicker.
5. Once a face has been "unknown" for STABILITY frames in a row, pause and
   prompt for a name in the terminal; save the embedding under that name.

DB layout: a pickle file containing ``dict[str, list[np.ndarray]]``.
Each name maps to a list of exemplar embeddings; matching uses the max
similarity across all exemplars of a name. Naming a recognised-but-low-
confidence face again under the same name strengthens the model.

Controls inside the cv2 window:
    q   quit
    r   reload DB from disk
"""

from __future__ import annotations
import sys
import time
import pickle
import argparse
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


DEFAULT_DB = "faces.db"
DEFAULT_THRESHOLD = 0.4
DEFAULT_STABILITY = 5
IOU_SAME_FACE = 0.5


# ----------------------------------------------------------------------------
# Face database
# ----------------------------------------------------------------------------


class FaceDB:
    """Pickled ``dict[name, list[normed_embedding]]`` with multi-exemplar match."""

    def __init__(self, path: Path):
        """Load or create a persistent face database at ``path``."""
        self.path = path
        self.people: dict[str, list[np.ndarray]] = {}
        self.load()

    def load(self) -> None:
        """Load people and exemplar embeddings from disk if available."""
        if self.path.exists():
            with self.path.open("rb") as f:
                self.people = pickle.load(f)
            n_emb = sum(len(v) for v in self.people.values())
            print(f"Loaded {len(self.people)} people ({n_emb} embeddings) from {self.path}")
        else:
            self.people = {}
            print(f"No DB at {self.path} — starting empty.")

    def save(self) -> None:
        """Persist the database with an atomic file replacement."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.people, f)
        tmp.replace(self.path)  # atomic on POSIX

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Store a new exemplar embedding for ``name``."""
        self.people.setdefault(name, []).append(embedding.astype(np.float32))
        self.save()

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[Optional[str], float]:
        """Return ``(name, similarity)`` if best match >= threshold, else ``(None, best_sim)``."""
        if not self.people:
            return None, 0.0
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, exemplars in self.people.items():
            sims = np.stack(exemplars) @ embedding  # (k,)
            s = float(sims.max())
            if s > best_sim:
                best_name, best_sim = name, s
        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim


# ----------------------------------------------------------------------------
# Lightweight per-frame tracking by IoU
# ----------------------------------------------------------------------------


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for two bounding boxes."""
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
    return inter / (area_a + area_b - inter)


class Tracker:
    """Greedy IoU tracker — each detection inherits the most-overlapping prior track.

    Each track holds ``name`` (or ``None``), an ``unknown_streak`` counter that
    increments while ``name`` is still ``None``, and a ``skipped`` flag set when
    the user dismisses the prompt for this face this session.
    """

    def __init__(self) -> None:
        """Create an empty tracker."""
        self.tracks: list[dict] = []
        self._next_id = 0

    def step(self, detections: list[tuple[np.ndarray, Optional[str]]]) -> list[dict]:
        """Return tracks aligned 1:1 with ``detections``."""
        for t in self.tracks:
            t["_matched"] = False

        result: list[dict] = []
        for det_bbox, det_name in detections:
            best_i, best_v = -1, 0.0
            for i, t in enumerate(self.tracks):
                if t["_matched"]:
                    continue
                v = iou(det_bbox, t["bbox"])
                if v > best_v:
                    best_v, best_i = v, i

            if best_v >= IOU_SAME_FACE:
                t = self.tracks[best_i]
                t["_matched"] = True
                t["bbox"] = det_bbox
                if det_name is not None:
                    t["name"] = det_name
                    t["unknown_streak"] = 0
                elif t["name"] is None and not t["skipped"]:
                    t["unknown_streak"] += 1
                result.append(t)
            else:
                t = {
                    "id": self._next_id,
                    "bbox": det_bbox,
                    "name": det_name,
                    "unknown_streak": 0 if det_name else 1,
                    "skipped": False,
                    "_matched": True,
                }
                self._next_id += 1
                result.append(t)

        # Drop tracks that didn't survive this frame.
        self.tracks = list(result)
        return result


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------


def draw_face(frame: np.ndarray, track: dict, sim: float, stability: int) -> None:
    """Draw a bounding box and label for a tracked face."""
    x1, y1, x2, y2 = (int(v) for v in track["bbox"])
    name = track["name"]
    if name is not None:
        label = f"{name}  sim={sim:.2f}"
        color = (0, 200, 0)  # green BGR
    elif track["skipped"]:
        label = "skipped"
        color = (128, 128, 128)
    else:
        label = f"? ({track['unknown_streak']}/{stability})"
        color = (0, 165, 255)  # orange BGR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_paused_banner(frame: np.ndarray) -> None:
    """Draw the prompt banner shown while waiting for terminal input."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 36), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "PAUSED — enter name in terminal (blank to skip)",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def main() -> int:
    """Run webcam face recognition with optional enrollment prompts."""
    parser = argparse.ArgumentParser(description="Webcam face recognition with a name DB.")
    parser.add_argument("--camera", type=int, default=0, help="cv2 VideoCapture index")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="cosine-similarity threshold to call a match"
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="path to the face database pickle")
    parser.add_argument(
        "--stability", type=int, default=DEFAULT_STABILITY, help="consecutive unknown detections before prompting"
    )
    parser.add_argument("--no-mirror", action="store_true", help="don't horizontally flip the preview")
    args = parser.parse_args()

    print(f"insightface {insightface.__version__}")
    print("Loading FaceAnalysis (CPU)...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  ready")

    db = FaceDB(Path(args.db))
    tracker = Tracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            "error: could not open camera. Grant Terminal camera access "
            "in System Settings → Privacy & Security → Camera.",
            file=sys.stderr,
        )
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window = "InsightFace Recognition  (q=quit, r=reload DB)"
    cv2.namedWindow(window)
    print("Webcam open. Press q in the video window to quit, r to reload DB.\n")

    fps_t0 = time.time()
    fps_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("error: failed to read frame", file=sys.stderr)
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        faces = app.get(frame)

        detections = [(f.bbox.astype(int), db.match(f.normed_embedding, args.threshold)[0]) for f in faces]
        tracks = tracker.step(detections)

        # Render boxes + labels.
        for face, track in zip(faces, tracks):
            _, sim = db.match(face.normed_embedding, args.threshold)
            draw_face(frame, track, sim, args.stability)

        # FPS counter.
        fps_n += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_n / (time.time() - fps_t0)
            fps_n, fps_t0 = 0, time.time()
            cv2.setWindowTitle(window, f"InsightFace Recognition  ({fps:.1f} FPS)")

        # First stable unknown in this frame → pause and prompt.
        prompt_idx = None
        for i, track in enumerate(tracks):
            if track["name"] is None and not track["skipped"] and track["unknown_streak"] >= args.stability:
                prompt_idx = i
                break

        if prompt_idx is not None:
            paused = frame.copy()
            draw_paused_banner(paused)
            # Highlight the face being named in cyan.
            tx1, ty1, tx2, ty2 = (int(v) for v in tracks[prompt_idx]["bbox"])
            cv2.rectangle(paused, (tx1, ty1), (tx2, ty2), (255, 255, 0), 3)
            cv2.imshow(window, paused)
            cv2.waitKey(1)  # paint

            print(f"\n→ Unknown face at bbox {tracks[prompt_idx]['bbox'].tolist()}.")
            try:
                name = input("  Name (blank to skip this session): ").strip()
            except (EOFError, KeyboardInterrupt):
                name = ""
            if name:
                db.add(name, faces[prompt_idx].normed_embedding)
                tracks[prompt_idx]["name"] = name
                tracks[prompt_idx]["unknown_streak"] = 0
                print(f"  ✓ saved '{name}' → {args.db} ({len(db.people[name])} exemplar(s))")
            else:
                tracks[prompt_idx]["skipped"] = True
                print("  ✗ skipped for this session")
        else:
            cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            db.load()

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
