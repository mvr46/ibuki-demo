#!/usr/bin/env bash
# One-command runner for the InsightFace face-analysis demo.
#
# Creates a Python 3.12 venv via uv, installs insightface from PyPI plus
# the CPU onnxruntime, then runs demo.py. The first run downloads the
# buffalo_l model pack (~326 MB) into ~/.insightface/models/.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found. Install with: brew install uv" >&2
  exit 1
fi

VENV=.venv
PY_VERSION=3.12

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv ($VENV) with Python $PY_VERSION"
  uv venv --python "$PY_VERSION" "$VENV"
fi

# Always re-resolve deps — cheap when already installed, correct after edits.
echo "==> Installing dependencies"
VIRTUAL_ENV="$PWD/$VENV" uv pip install --quiet \
  "insightface>=0.7" \
  onnxruntime \
  opencv-python

echo "==> Running demo"
"$VENV/bin/python" demo.py

OUT="$PWD/t1_output.jpg"
if [[ -f "$OUT" ]]; then
  echo "==> Opening $OUT"
  open "$OUT" 2>/dev/null || echo "  (open failed; view it at $OUT)"
fi
