# Downloads the MediaPipe HolisticLandmarker model asset used by
# app/vision/landmarks.py. This is Google's official, publicly hosted
# MediaPipe Tasks model — not a custom or invented file.
# See docs/architecture.md ("Why MediaPipe Tasks") for details.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelDir = Join-Path $repoRoot "backend\models\mediapipe"
$modelPath = Join-Path $modelDir "holistic_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

if (Test-Path $modelPath) {
    Write-Host "MediaPipe model already present at $modelPath — skipping download."
} else {
    Write-Host "Downloading MediaPipe HolisticLandmarker model..."
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
    Write-Host "Saved to $modelPath"
}
