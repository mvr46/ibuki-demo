"""Smoke-test demo for InsightFace face analysis.

Loads the bundled `t1.jpg` (six faces), runs detection + landmarks + age/gender
recognition on CPU, writes an annotated `t1_output.jpg`, and prints the
all-to-all face similarity matrix.

First run downloads the `buffalo_l` model pack (~326 MB) into
``~/.insightface/models/`` automatically. Subsequent runs use the cache.
"""

from __future__ import annotations
import sys
import time

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def main() -> int:
    """Run the bundled InsightFace smoke test and write an annotated image."""
    print(f"insightface version: {insightface.__version__}")
    print("Loading FaceAnalysis (CPU). First run downloads ~326 MB...")

    t0 = time.time()
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)  # -1 = CPU; det_size defaults to Auto (128 + 640)
    print(f"  ready in {time.time() - t0:.1f}s")

    img = ins_get_image("t1")
    print(f"Loaded test image t1.jpg: shape={img.shape}")

    t0 = time.time()
    faces = app.get(img)
    print(f"Detected {len(faces)} faces in {time.time() - t0:.2f}s")

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        gender = "M" if face.sex == "M" else "F"
        print(
            f"  face {i}: bbox=({x1},{y1})-({x2},{y2})  "
            f"age={int(face.age)}  sex={gender}  det_score={face.det_score:.3f}"
        )

    annotated = app.draw_on(img, faces)
    out_path = "t1_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Wrote annotated image -> {out_path}")

    if len(faces) >= 2:
        feats = np.array([f.normed_embedding for f in faces], dtype=np.float32)
        sims = feats @ feats.T
        print("All-to-all cosine similarity:")
        with np.printoptions(precision=3, suppress=True):
            print(sims)

    return 0


if __name__ == "__main__":
    sys.exit(main())
