#!/usr/bin/env bash
# Downloads the MediaPipe HolisticLandmarker model asset used by
# app/vision/landmarks.py. Official, publicly hosted MediaPipe Tasks model
# — not a custom or invented file. See docs/architecture.md for details.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/backend/models/mediapipe"
MODEL_PATH="$MODEL_DIR/holistic_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "MediaPipe model already present at $MODEL_PATH — skipping download."
else
    echo "Downloading MediaPipe HolisticLandmarker model..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
    echo "Saved to $MODEL_PATH"
fi
