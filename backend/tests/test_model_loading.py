"""Tests for app.models.model_loader — demo fallback and trained-checkpoint
loading. No GPU or real dataset is required."""

from __future__ import annotations

import json

import pytest
import torch

from app.config import settings
from app.models.classifier import build_model
from app.models.model_loader import ModelBundle, load_model


def test_demo_fallback_when_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    bundle = load_model("transformer")
    assert isinstance(bundle, ModelBundle)
    assert bundle.is_trained is False
    assert bundle.checkpoint_path is None
    assert bundle.num_classes == len(settings.vocabulary) + 1  # + UNKNOWN
    assert bundle.classifier.num_parameters > 0


def test_demo_bundle_predict_runs_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    bundle = load_model("transformer")
    x = torch.randn(1, settings.sequence_length, bundle.input_dim)
    result = bundle.classifier.predict(x, top_k=3)
    assert len(result.top_labels) == 3


def test_load_trained_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    class_names = ["HELLO", "YES", "NO"]
    input_dim = 30

    model = build_model(
        "transformer", input_dim=input_dim, num_classes=len(class_names),
        embedding_dim=16, transformer_layers=1, transformer_heads=2, transformer_ff_dim=32,
        max_seq_len=16,  # must match the checkpoint's "sequence_length" below
    )
    checkpoint_path = tmp_path / "isl_transformer.pt"
    torch.save({
        "model_type": "transformer",
        "input_dim": input_dim,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "config": {
            "embedding_dim": 16, "transformer_layers": 1, "transformer_heads": 2,
            "transformer_ff_dim": 32, "dropout": 0.2, "sequence_length": 16,
        },
        "metrics": {"val_accuracy": 0.87},
    }, checkpoint_path)

    bundle = load_model("transformer")
    assert bundle.is_trained is True
    assert bundle.class_names == class_names
    assert bundle.metrics["val_accuracy"] == 0.87

    x = torch.randn(1, 16, input_dim)
    result = bundle.classifier.predict(x, top_k=2)
    assert len(result.top_labels) == 2


def test_corrupted_checkpoint_falls_back_to_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    bad_path = tmp_path / "isl_transformer.pt"
    bad_path.write_bytes(b"not a real checkpoint")

    bundle = load_model("transformer")
    assert bundle.is_trained is False  # gracefully degraded, did not raise


def test_build_model_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_model("not_a_real_model_type", input_dim=10, num_classes=5)


def test_server_limits_torch_threads_at_startup(monkeypatch):
    """Idle PyTorch intra-op threads busy-wait between the server's tiny
    per-window inferences (~400% CPU at 15 fps with 6 threads vs ~70% with 1)."""
    from fastapi.testclient import TestClient

    import app.main as main_module

    calls = []
    monkeypatch.setattr(main_module.torch, "set_num_threads", calls.append)
    with TestClient(main_module.app):
        pass
    assert calls == [settings.inference_threads] and settings.inference_threads == 1
