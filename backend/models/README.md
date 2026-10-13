# Trained model checkpoints

This directory holds trained model artifacts produced by `training/train.py`:

```
models/
├── isl_transformer.pt   # Transformer checkpoint (created by training/train.py)
├── isl_bilstm.pt        # BiLSTM checkpoint (created by training/train.py)
├── classes.json         # class list matching whichever checkpoint you trained last
└── mediapipe/
    └── holistic_landmarker.task   # MediaPipe model asset — see scripts/download_models.*
```

The two trained checkpoints and `classes.json` (28 INCLUDE classes, ~5.8 MB
total) are committed so the deployed backend can serve them; their SHA-256
hashes are pinned in `tests/test_model_integrity.py`. The MediaPipe asset is
not committed — download it with `scripts/download_models.*`.

If no `isl_transformer.pt` / `isl_bilstm.pt`
is present, the backend automatically falls back to an **untrained, randomly
initialized** model of the same architecture so the full app (camera →
landmarks → buffer → model → smoothing → UI) stays runnable for development
— but its predictions are meaningless until you actually train a model
(see `backend/training/` and `docs/model.md`).

`GET /api/model/info` always reports `"is_trained": true|false` so the
frontend/API never silently presents a demo model as a real one.
