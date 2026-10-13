# SIGN-X

**Real-Time Indian Sign Language (ISL) → Text → Voice Translator**

SIGN-X captures webcam video in the browser, extracts hand/pose/face
landmarks with MediaPipe, feeds a sliding window of landmark sequences into
a temporal deep learning model (a Transformer or a BiLSTM baseline), turns
the recognized sign into a growing sentence, and speaks it aloud with
text-to-speech — entirely on your own machine.

> **Honesty note**: no dataset or trained model ships with this repository.
> Without training a model, the backend runs an explicitly-flagged
> **untrained demo model** so the full pipeline (camera → landmarks →
> buffer → model → smoothing → UI) is runnable and inspectable end to end.
> See [Limitations](#limitations) and [`docs/model.md`](docs/model.md).

---

## Features

- Real-time webcam capture → MediaPipe HolisticLandmarker (hands + pose + face in one pass)
- Configurable landmark normalization, missing-landmark handling, temporal padding/truncation, optional velocity features
- Rolling temporal buffer (configurable window + stride) — the model always sees a sequence, never a single frame
- Temporal Transformer (primary) **and** BiLSTM (baseline) sign classifiers, trainable and directly comparable
- Confidence thresholding, majority-vote smoothing, debounce, and an explicit `UNKNOWN` state
- Rule-based gloss → sentence builder (phrase templates + a small SOV→SVO grammar heuristic)
- Local/offline text-to-speech (`pyttsx3`) behind a swappable `TTSService` interface
- FastAPI REST + WebSocket backend; React + TypeScript + Tailwind dashboard
- Full training pipeline: landmark extraction → dataset prep → train → evaluate, with checkpointing, early stopping, LR scheduling, and augmentation
- CPU/CUDA auto-detection; automatic degradation on missing camera/model/dataset instead of crashing
- 56 passing backend tests, none requiring a physical webcam

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagram and component breakdown. Short version:

```
Webcam (browser) --JPEG/WebSocket--> FastAPI --> MediaPipe HolisticLandmarker
  --> Feature Extractor --> Temporal Buffer --> Transformer/BiLSTM
  --> Prediction Smoother --> Sentence Builder --> Text --> TTS
```

Repository layout:

```
sign-x/ (this repo, at D:\aiproject)
├── backend/
│   ├── app/            # FastAPI server: api, vision, models, inference, language, speech, utils
│   ├── training/        # extract_landmarks.py, prepare_dataset.py, train.py, evaluate.py, config.yaml
│   ├── datasets/         # your dataset goes here (see docs/dataset.md) + a synthetic pipeline-test generator
│   ├── models/           # trained checkpoints + MediaPipe model asset
│   └── tests/            # pytest suite (no webcam required)
├── frontend/             # React + TypeScript + Tailwind + Vite dashboard
├── scripts/              # setup_windows.ps1, setup_linux.sh, run_dev.ps1, download_models.*
└── docs/                 # architecture.md, dataset.md, model.md, development.md
```

## System requirements

- Windows 11 (primary target; Linux/macOS scripts also provided), Python 3.10/3.11, Node.js 18+
- A webcam for live use (everything else — training, tests, API — works without one)
- Optional: an NVIDIA GPU with a CUDA-enabled PyTorch build for faster training (see [GPU setup](#gpu-setup))

## Installation

```powershell
git clone <this-repo> sign-x
cd sign-x
./scripts/setup_windows.ps1
```

This creates `backend/.venv`, installs `backend/requirements.txt`, downloads the MediaPipe model asset into `backend/models/mediapipe/`, and runs `npm install` in `frontend/`. Linux/macOS: `./scripts/setup_linux.sh`.

## Running it

```powershell
./scripts/run_dev.ps1
```

or manually:

```powershell
# Terminal 1
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

Open the Vite URL (typically http://localhost:5173). The backend API docs are at http://localhost:8000/docs.

### Webcam setup

The browser (not the backend process) opens the webcam via `getUserMedia` and streams JPEG frames to the backend over WebSocket — grant camera permission when the browser prompts. If access is denied or no camera exists, the dashboard shows a clear status message instead of a blank/broken video panel. `SIGNX_CAMERA_INDEX` and related settings in `app/config.py` are used by the *training-side* OpenCV video reading, not by the browser capture path.

## Dataset setup

No dataset is bundled. See [`docs/dataset.md`](docs/dataset.md) for recommended, real ISL datasets (INCLUDE by AI4Bharat, ISL-CSLTR) and how to lay them out under `backend/datasets/raw/<CLASS_NAME>/*.mp4`. To exercise the pipeline immediately without a real dataset:

```powershell
cd backend
python -m datasets.sample.generate_sample_dataset --dataset_name isl_demo --samples_per_class 20
python -m training.prepare_dataset --dataset_name isl_demo
```

(This generates **synthetic**, non-ISL data purely to prove the pipeline runs — see the warning printed by the script.)

## Training

```powershell
cd backend
python -m training.extract_landmarks --dataset_name isl_demo   # from a real dataset under datasets/raw/
python -m training.prepare_dataset --dataset_name isl_demo
python -m training.train --config training/config.yaml
```

Edit `training/config.yaml` to switch between `transformer`/`bilstm`, adjust hyperparameters, dataset path, batch size, epochs, etc. See [`docs/model.md`](docs/model.md) for the full architecture and checkpoint format.

## Evaluation

```powershell
python -m training.evaluate --config training/config.yaml
```

Writes `training/logs/evaluation_results_<model_type>.json` (accuracy, macro precision/recall/F1, top-1/top-k accuracy, per-class accuracy, confusion matrix, inference latency) and `confusion_matrix_<model_type>.png`.

## Model usage

`GET /api/model/info` reports the currently loaded model (type, trained/untrained, device, parameter count, class list, and — once trained — its evaluation metrics). The backend loads the model once at startup and reuses it for every request; it never reloads per-frame. If no checkpoint exists, it falls back to an untrained demo model rather than crashing, and reports `"is_trained": false` everywhere.

## API documentation

- `GET /api/health` — service status + selected device
- `GET /api/model/info` / `POST /api/model/reload` — model status / hot-swap between transformer/bilstm
- `GET /api/dataset/info` — what's actually prepared under `backend/datasets/`
- Interactive OpenAPI docs at `/docs` once the backend is running

## WebSocket protocol

`ws://localhost:8000/ws/predict`

**Client → Server**: binary frames (one JPEG image each) for prediction; text frames for control: JSON objects `{"type": "clear"}`, `{"type": "reset"}` (camera switched), `{"type": "speak"}`, `{"type": "stop_speech"}`, `{"type": "ping"}`. Anything else gets an `error` reply and the connection stays open. Every binary frame gets exactly one `prediction` or `error` reply.

**Server → Client** (JSON): `connection` (on connect, includes model info), `prediction` (`sign`, `confidence`, `is_unknown`, `committed`, `sentence`, `top_predictions`, `fps`, `latency_ms`, `landmarks` for overlay drawing, ...), `status`, `error`. See [`docs/architecture.md`](docs/architecture.md) for full details.

## GPU setup

`app/config.py` auto-detects CUDA via `torch.cuda.is_available()` and falls back to CPU. The default `pip install torch` on Windows installs a **CPU-only** build. To use an NVIDIA GPU (e.g. RTX 4050), install a CUDA-enabled build instead, e.g.:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(pick the CUDA version matching your driver — see pytorch.org). The selected device is shown in the dashboard's System Status panel and in `/api/health`.

## Testing

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m pytest -q
```

No physical webcam is required. Most tests use synthetic frames; `tests/test_model_integrity.py` checks the shipped checkpoints (pinned SHA-256, exact weights, 28-class mapping) and — when the git-ignored dataset and INCLUDE videos are present — re-scores the held-out test split in memory and compares it with the recorded results (skipped otherwise).

Frontend:

```powershell
cd frontend
npm test        # Node's built-in test runner (Node >= 22.18)
npm run lint
npm run build   # tsc -b + vite build
```

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for the GitHub + Render setup (`render.yaml`, `backend/Dockerfile`), measured resource needs and plan sizing. Camera capture and speech always run in the user's browser.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Dashboard shows "Camera unavailable" | No webcam, or another app is holding it. Grant browser camera permission; try a different browser. |
| `MediaPipe model unavailable` banner | Run `scripts/download_models.ps1` (or `.sh`) to fetch `holistic_landmarker.task` into `backend/models/mediapipe/`. |
| Model page shows "Untrained demo weights" | Expected until you run the training pipeline — see [Training](#training). |
| WebSocket keeps reconnecting | Backend isn't running, or a firewall is blocking port 8000. Check the backend terminal for errors. |
| `training.train` fails with "Split directory not found" | Run `training.extract_landmarks` then `training.prepare_dataset` first. |
| Frontend `npm install` fails | Ensure Node.js 18+ is installed and on PATH (`node --version`). |

## Limitations

- ISL grammar (topic-comment structure, spatial referencing, grammatically meaningful facial expressions) is far richer than this project's small template-based sentence builder — see `app/language/gloss_to_text.py`.
- The default ~20-word vocabulary is a starting point for validating the pipeline, not a claim of general ISL coverage.
- Recognition quality depends heavily on lighting, camera angle, and distance from the camera.
- Different signers sign the same word differently; generalization depends entirely on the training data you provide.
- Continuous sentence recognition (beyond isolated signs) is architecturally supported but substantially harder, and is provided as an extensible foundation rather than a finished continuous-ISL translator.
- No accuracy/F1 numbers are claimed anywhere in this repo unless produced by an actual `training/evaluate.py` run against a real trained checkpoint.

Full discussion in [`docs/architecture.md`](docs/architecture.md#limitations-read-before-drawing-conclusions-from-this-project) and [`docs/about` page in the app].

## Future extensions

Larger ISL vocabulary · continuous sign recognition · sequence-to-sequence gloss translation · facial-expression/grammar recognition · signer adaptation · multilingual (Hindi/Telugu/English) speech output · mobile/ONNX/TensorRT deployment · sign-generation/avatar output · personalized calibration.

## Privacy & security

Camera frames are processed in memory and are not persisted to disk by default. No frame or landmark data is sent to any third-party/external service — everything runs on your own backend process. WebSocket input is size-limited and validated; malformed frames/messages return an `error` message instead of crashing the connection.

## License

MIT — see [`LICENSE`](LICENSE). Third-party datasets and the MediaPipe model asset retain their own licenses (see `docs/dataset.md` / `docs/architecture.md`). The bundled trained checkpoints were trained on INCLUDE (CC BY 4.0) — see [`backend/models/ATTRIBUTION.md`](backend/models/ATTRIBUTION.md) for provenance and the required citation.
