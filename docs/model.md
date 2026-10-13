# SIGN-X — Model Documentation

## Architectures

### Temporal Transformer (primary) — `app/models/transformer.py`

```
Input (batch, seq_len, feature_dim)
  -> Linear + LayerNorm (input projection to embedding_dim)
  -> Sinusoidal Positional Encoding
  -> TransformerEncoder (num_layers x [self-attention + feed-forward])
  -> Learned attention pooling over time (mask-aware)
  -> Dropout
  -> Linear classification head -> (batch, num_classes) logits
```

Configurable (via `app/config.py` defaults or `training/config.yaml`):
`embedding_dim`, `transformer_layers`, `transformer_heads`,
`transformer_ff_dim`, `dropout`, `sequence_length` (used as the positional
encoding's max length).

### BiLSTM (baseline) — `app/models/bilstm.py`

```
Input (batch, seq_len, feature_dim)
  -> LayerNorm
  -> Bidirectional multi-layer LSTM (packs padded sequences using the mask)
  -> Concatenate final forward + backward hidden states
  -> Dropout
  -> Linear classification head -> (batch, num_classes) logits
```

Configurable: `bilstm_hidden_dim`, `bilstm_layers`, `dropout`.

Both models share the exact same `(batch, seq_len, feature_dim)` input
contract and an optional `(batch, seq_len)` boolean mask (`True` = real
frame, `False` = padding), so they're drop-in interchangeable —
`app/models/classifier.py::build_model()` is the single factory for both.

## Checkpoint format

`training/train.py` saves (and `app/models/model_loader.py` loads) a single
`torch.save`'d dict:

```python
{
    "model_type": "transformer" | "bilstm",
    "input_dim": int,
    "class_names": List[str],
    "state_dict": <model.state_dict()>,
    "config": {...architecture hyperparameters + sequence_length...},
    "metrics": {"val_accuracy": float, "val_loss": float, "epoch": int},
}
```

alongside a matching `classes.json`. Checkpoints go in `backend/models/`:
`isl_transformer.pt`, `isl_bilstm.pt`.

## Untrained "demo" fallback

If `backend/models/isl_transformer.pt` (or `isl_bilstm.pt`) doesn't exist,
or fails to load, `model_loader.py` builds a **randomly initialized** model
over the configured vocabulary instead of crashing the server. This is
reported honestly everywhere:

- `GET /api/model/info` returns `"is_trained": false`
- Startup logs a clear warning
- The frontend Model page must surface this flag, not hide it

**No accuracy, F1, or "% correct" numbers are fabricated anywhere.** Any
metric shown in the UI or docs comes from an actual `training/evaluate.py`
run (`training/logs/evaluation_results_<model_type>.json`) against a real
trained checkpoint and a real dataset split — never invented.

## Training the models

```powershell
# 1. Prepare a dataset (see docs/dataset.md), then:
python -m training.extract_landmarks --dataset_name isl_demo
python -m training.prepare_dataset --dataset_name isl_demo

# 2. Train (edit training/config.yaml to pick transformer vs bilstm, hyperparameters)
python -m training.train --config training/config.yaml

# 3. Evaluate on the held-out test split
python -m training.evaluate --config training/config.yaml
```

`train.py` implements: checkpoint saving (best validation accuracy only),
early stopping (`early_stopping_patience`), `ReduceLROnPlateau` LR
scheduling, and a per-epoch JSON training history under `training/logs/`.

`evaluate.py` reports: accuracy, macro precision/recall/F1, top-1 and
top-`k` accuracy, per-class accuracy, a confusion matrix (JSON + PNG), and
average per-sample inference latency — written to
`training/logs/evaluation_results_<model_type>.json` and
`confusion_matrix_<model_type>.png`.

## Comparing Transformer vs BiLSTM

Train both (point `training/config.yaml`'s `model.type` at each in turn,
or copy the config), then compare their
`training/logs/evaluation_results_*.json` files directly — same test
split, same metrics, side by side. Neither model is hardcoded as "the
better one" anywhere in the code; that's an empirical question for your
actual dataset.

## Device selection

`app/config.py::Settings.resolved_device` returns `"cuda"` when
`torch.cuda.is_available()`, else `"cpu"`, unless `SIGNX_DEVICE` is forced
to a specific value. The currently selected device is surfaced via
`GET /api/health` and `GET /api/model/info`, and shown in the dashboard's
System Status panel. All inference and evaluation code uses
`torch.inference_mode()` / `torch.no_grad()`.
