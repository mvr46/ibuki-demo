#!/usr/bin/env bash
# One-command webcam face recognition.
#
# Reuses (or creates) the same .venv as run.sh. On first webcam access,
# macOS will prompt the Terminal for camera permission — if denied, the
# script exits with a hint to re-grant in System Settings.
set -euo pipefail

cd "$(dirname "$0")"

VENV=.venv
PY_VERSION=3.12

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found. Install with: brew install uv" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv ($VENV) with Python $PY_VERSION"
  uv venv --python "$PY_VERSION" "$VENV"
  echo "==> Installing dependencies"
  VIRTUAL_ENV="$PWD/$VENV" uv pip install --quiet \
    "insightface>=0.7" onnxruntime opencv-python
fi

exec "$VENV/bin/python" recognize.py "$@"
