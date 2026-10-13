# SIGN-X Backend

FastAPI server + training pipeline. See the root [`README.md`](../README.md) for full setup/usage and [`../docs/`](../docs/) for architecture, dataset, model, and development details.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
..\scripts\download_models.ps1     # fetches the MediaPipe model asset
uvicorn app.main:app --reload --port 8000
```

## Layout

- `app/` — the runtime server: `api/` (REST + WebSocket), `vision/` (camera, MediaPipe, preprocessing, feature extraction), `models/` (Transformer, BiLSTM, loader), `inference/` (temporal buffer, smoothing, predictor), `language/` (vocabulary, gloss→text, sentence builder), `speech/` (TTS interface), `utils/` (logging, metrics).
- `training/` — `extract_landmarks.py` → `prepare_dataset.py` → `train.py` → `evaluate.py`, driven by `config.yaml`.
- `datasets/` — your dataset goes here (not bundled); `datasets/sample/` generates a synthetic pipeline-test dataset.
- `models/` — trained checkpoints (`isl_transformer.pt`, `isl_bilstm.pt`, `classes.json`) + `models/mediapipe/` (downloaded model asset).
- `tests/` — pytest suite; no physical webcam required.

## Tests

```powershell
python -m pytest -q
```
