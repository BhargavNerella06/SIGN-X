"""Tests for the model backbones (Transformer + BiLSTM) — verifies shapes,
configurability, and that padding masks are honored without crashing."""

from __future__ import annotations

import torch

from app.models.bilstm import SignBiLSTM
from app.models.classifier import ClassificationResult, SignClassifier, build_model
from app.models.transformer import SignTransformer


def test_transformer_forward_pass_shape():
    model = SignTransformer(input_dim=32, num_classes=10, embedding_dim=16, num_layers=2, num_heads=2, ff_dim=32)
    x = torch.randn(4, 20, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_transformer_respects_mask():
    model = SignTransformer(input_dim=8, num_classes=3, embedding_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[False, False, True, True, True, True], [True] * 6])
    logits = model(x, mask=mask)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_transformer_configurable_hyperparameters():
    model = SignTransformer(
        input_dim=10, num_classes=5, embedding_dim=24, num_layers=3, num_heads=4, ff_dim=48, dropout=0.1,
    )
    assert model.embedding_dim == 24
    assert len(model.encoder.layers) == 3


def test_bilstm_forward_pass_shape():
    model = SignBiLSTM(input_dim=16, num_classes=7, hidden_dim=12, num_layers=2)
    x = torch.randn(3, 15, 16)
    logits = model(x)
    assert logits.shape == (3, 7)


def test_bilstm_forward_pass_with_mask():
    model = SignBiLSTM(input_dim=6, num_classes=4, hidden_dim=8, num_layers=1)
    x = torch.randn(2, 10, 6)
    mask = torch.tensor([[True] * 5 + [False] * 5, [True] * 10])
    logits = model(x, mask=mask)
    assert logits.shape == (2, 4)


def test_build_model_factory_transformer_and_bilstm():
    t_model = build_model("transformer", input_dim=10, num_classes=5)
    b_model = build_model("bilstm", input_dim=10, num_classes=5)
    assert isinstance(t_model, SignTransformer)
    assert isinstance(b_model, SignBiLSTM)


def test_sign_classifier_predict_returns_topk():
    model = build_model("transformer", input_dim=10, num_classes=6, embedding_dim=8, transformer_layers=1, transformer_heads=2)
    classifier = SignClassifier(model, class_names=[f"C{i}" for i in range(6)], device="cpu")
    x = torch.randn(1, 12, 10)
    result = classifier.predict(x, top_k=3)
    assert isinstance(result, ClassificationResult)
    assert len(result.top_labels) == 3
    assert 0.0 <= result.best_prob <= 1.0
    assert abs(sum(result.top_probs) - sum(sorted(result.top_probs, reverse=True))) < 1e-6
