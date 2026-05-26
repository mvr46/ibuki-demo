# Face recognition reference CLI

Two standalone demos showing the InsightFace pipeline that the
[face-recognition integration](../../docs/FACE_RECOGNITION_INTEGRATION.md)
will adopt into the conversation app. Use this directory as the working
reference when implementing Phase 1 — the `FaceDB` class and recognition
loop in [recognize.py](recognize.py) are the code the spec asks you to
extract into `insightface/perception_lib.py` and `vision/face_identity.py`.

## What's here

| File | Purpose |
|---|---|
| `demo.py` | Static-image smoke test on the bundled `t1.jpg` (6 faces). Verifies detection + recognition + age/sex on CPU. |
| `recognize.py` | Live webcam loop with a multi-exemplar pickled face DB. Pauses and prompts for a name when an unknown face is stable for ≥ 1 second; saves the embedding under that name; recognises returning faces on subsequent frames. |
| `run.sh` | One-command runner for `demo.py`. Creates a venv, pip-installs `insightface` from PyPI, runs the demo, opens the annotated output. |
| `recognize.sh` | Same venv as `run.sh`. Runs the webcam recognizer. |

## Quick start (macOS)

```bash
./run.sh         # static-image smoke test
./recognize.sh   # live webcam, name faces as you go
```

The first run downloads the `buffalo_l` model pack (~326 MB) into
`~/.insightface/models/`. Subsequent runs are instant.

`recognize.sh` stores names in `./faces.db` (a pickled
`dict[name, list[normed_embedding]]`). Delete the file to start fresh.

## How the integration spec relates

The spec at [`docs/FACE_RECOGNITION_INTEGRATION.md`](../../docs/FACE_RECOGNITION_INTEGRATION.md)
re-uses three pieces of this code:

- `FaceDB` → `insightface/perception_lib.py` (verbatim, with one tweak for
  the new default DB path under `~/.reachy-mini/faces.db`).
- The cosine-similarity matching loop → `FaceRecognizer.embed()`, but
  operating on a YOLO-cropped face instead of running its own SCRFD
  detection.
- The IoU `Tracker` that gates "stable unknown" → the same logic, ported
  to `FaceIdentifierWorker`, drives the `[Vision: Alice entered the frame]`
  events streamed into the realtime LLM.
