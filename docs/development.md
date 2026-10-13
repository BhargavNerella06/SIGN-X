# SIGN-X — Development Guide

## Prerequisites

- Windows 11 (primary target), Python 3.10/3.11, Node.js 18+ (for the
  frontend), a webcam (for live testing — everything else works without one).
- Optional: an NVIDIA GPU + CUDA-enabled PyTorch build for faster training.
  The default `pip install torch` on Windows without a matching CUDA index
  URL installs the **CPU-only** build — see `docs/model.md` / the root
  README's "GPU setup" note if you have an RTX GPU and want CUDA training.

## First-time setup

```powershell
# from the repo root
./scripts/setup_windows.ps1
```

This creates `backend/.venv`, installs `backend/requirements.txt`,
downloads the MediaPipe HolisticLandmarker model into
`backend/models/mediapipe/`, and runs `npm install` in `frontend/`.

## Running in development

```powershell
./scripts/run_dev.ps1
```

or manually, in two terminals:

```powershell
# Terminal 1 — backend
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Then open the Vite dev server URL (typically http://localhost:5173).

## Project layout

See the root `README.md` for the full tree. The short version:
`backend/app/` is the runtime server, `backend/training/` is the offline
training pipeline, `frontend/` is the React dashboard.

## Testing

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m pytest -q
```

Tests never require a physical webcam — they use synthetic in-memory
frames and, where useful, monkeypatched settings/temp directories. See
`backend/tests/`.

## Configuration

All tunables live in `app/config.py` (`Settings`, a `pydantic-settings`
model) and can be overridden with `SIGNX_`-prefixed environment variables
or a `backend/.env` file, e.g.:

```
SIGNX_CAMERA_INDEX=1
SIGNX_CONFIDENCE_THRESHOLD=0.7
SIGNX_MODEL_TYPE=bilstm
```

Training-specific values additionally live in `training/config.yaml`.

## Coding conventions

- No hardcoded feature dimensions, class lists, or architecture
  hyperparameters — read them from `app.config.settings` or
  `training/config.yaml`.
- Vision/model/inference/language/speech stay in their own packages under
  `app/`; don't reach across layers except through the documented
  interfaces (e.g. `FeatureExtractor`, `SignClassifier`, `TTSService`).
- Never let a single bad frame, missing model, or disconnected camera
  crash the WebSocket loop — degrade to an `error`/`UNKNOWN` message
  instead (Section 25 of the original spec).

## Stage 2 notes (real-dataset integration)

- `backend/tests/test_dataset_adapter.py`, `test_landmark_dataset.py`,
  `test_class_balance.py`, `test_offline_inference_pipeline.py` were added
  alongside the real-dataset adapters — all use small synthetic fixtures
  (tiny fake parquet files, a handful of `.npy` arrays, a few frames of a
  generated `.avi`), never the actual multi-GB INCLUDE dataset.
- Writing `test_offline_inference_pipeline.py` surfaced a real bug:
  `training/dataset.py` previously derived its real-vs-padding mask by
  checking for all-zero rows, which incorrectly treated a genuine "nothing
  detected this frame" real frame the same as a padding row. Fixed by
  `app/vision/preprocessing.py::length_based_mask`, which masks based on
  the sequence's length *before* padding instead. If you're extending this
  codebase and see a similar all-zero-row heuristic, prefer this pattern.
- `training/datasets/include.py`'s label cleaner was also fixed after a
  test caught it missing INCLUDE's non-numeric `"Ex. <name>"` label prefix
  (e.g. `"Ex. Monsoon"`) alongside the usual `"<number>. <name>"` form.

## Known environment notes from this build

- Verified on Windows 11, Python 3.11.5, PyTorch 2.13 (CPU build — no CUDA
  GPU was available in the build/test environment, so CUDA code paths are
  implemented and auto-detected but were not exercised on real hardware
  here).
- MediaPipe 1.0.1 on this machine only ships the Tasks API (`mp.solutions`
  is gone) — see `docs/architecture.md` for why `HolisticLandmarker` is
  used instead.
- Node.js was not installed in the build/test environment, so the frontend
  could not be `npm install`ed / build-tested there. The backend (API,
  WebSocket protocol, models, training pipeline, tests) was fully
  installed and verified. Run `npm install && npm run build` once Node is
  available to verify the frontend compiles.
