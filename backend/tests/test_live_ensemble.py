"""Tests for the live Transformer+BiLSTM ensemble (EnsembleClassifier and
model_loader.with_live_ensemble)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.models.classifier import EnsembleClassifier, SignClassifier
from app.models.model_loader import load_model, with_live_ensemble

CLASSES = ["A", "B", "C"]


class _FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x, mask=None):
        return self.logits.expand(x.shape[0], -1)


def _member(logits, classes=CLASSES):
    return SignClassifier(_FixedLogits(logits), list(classes))


def test_ensemble_averages_member_probabilities():
    a, b = _member([4.0, 0.0, 0.0]), _member([0.0, 4.0, 0.0])
    x = torch.zeros(1, 8, 5)
    pa = torch.softmax(torch.tensor([4.0, 0.0, 0.0]), -1)
    pb = torch.softmax(torch.tensor([0.0, 4.0, 0.0]), -1)
    expected = (pa + pb) / 2

    result = EnsembleClassifier([a, b]).predict(x, top_k=3)

    assert sorted(result.top_labels[:2]) == ["A", "B"]
    assert result.top_probs[0] == pytest.approx(float(expected.max()), abs=1e-4)
    assert sum(result.top_probs) == pytest.approx(1.0, abs=1e-3)


def test_ensemble_can_disagree_with_each_member():
    # Both members lean to their own class, but agree on C as the runner-up:
    # the average must surface C, not just echo one member.
    a, b = _member([2.0, -5.0, 1.8]), _member([-5.0, 2.0, 1.8])
    assert EnsembleClassifier([a, b]).predict(torch.zeros(1, 8, 5)).best_label == "C"


def test_ensemble_rejects_mismatched_class_lists():
    with pytest.raises(ValueError):
        EnsembleClassifier([_member([0, 0, 0]), _member([0, 0, 0], classes=["A", "C", "B"])])


def test_with_live_ensemble_off_keeps_single_model(monkeypatch):
    monkeypatch.setattr(settings, "live_ensemble", False)
    bundle = with_live_ensemble(load_model("transformer"))
    assert isinstance(bundle.classifier, SignClassifier)
    assert bundle.ensemble_members == []


def test_with_live_ensemble_skips_untrained_demo_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "live_ensemble", True)
    bundle = with_live_ensemble(load_model("transformer"))
    assert bundle.is_trained is False
    assert bundle.ensemble_members == []


@pytest.mark.skipif(
    not ((settings.models_dir / settings.transformer_checkpoint).exists()
         and (settings.models_dir / settings.bilstm_checkpoint).exists()),
    reason="trained checkpoints not present",
)
def test_with_live_ensemble_combines_real_checkpoints(monkeypatch):
    monkeypatch.setattr(settings, "live_ensemble", True)
    bundle = with_live_ensemble(load_model("transformer"))

    assert bundle.is_trained and bundle.model_type == "transformer"
    assert bundle.ensemble_members == ["transformer", "bilstm"]
    assert isinstance(bundle.classifier, EnsembleClassifier)
    assert bundle.num_classes == 28
    info = bundle.info()
    assert info["ensemble_members"] == ["transformer", "bilstm"]
    assert info["num_classes"] == 28

    x = torch.zeros(1, settings.sequence_length, bundle.input_dim)
    mask = torch.ones(1, settings.sequence_length, dtype=torch.bool)
    result = bundle.classifier.predict(x, mask=mask, top_k=5)
    assert len(result.top_labels) == 5 and set(result.top_labels) <= set(bundle.class_names)
