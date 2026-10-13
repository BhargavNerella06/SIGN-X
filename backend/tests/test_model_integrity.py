"""Integrity checks for the trained checkpoints that ship with SIGN-X.

These run against the REAL files in backend/models/ (not synthetic ones) and
fail loudly if a checkpoint is missing, has been replaced, or silently
degraded to the untrained demo fallback. Checks that need the (git-ignored)
dataset or source videos are skipped when those are absent.

If you retrain on purpose, update ``_EXPECTED_SHA256`` and the recorded
evaluation JSON together, and note the change in docs/model.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.config import BACKEND_DIR, settings
from app.models import model_loader
from app.models.classifier import EnsembleClassifier

_MODELS = BACKEND_DIR / "models"
_DATASET = BACKEND_DIR / "datasets" / "isl_include_subset28"
_EVAL_DIR = BACKEND_DIR / "training" / "logs" / "isl_include_subset28"
_VIDEOS = BACKEND_DIR / "datasets" / "_include_zips" / "extracted"

# SHA-256 recorded at audit time (2026-10-13); identical to the
# best_model.pt of the training runs in training/logs/isl_include_subset28/.
_EXPECTED_SHA256 = {
    "isl_transformer.pt": "7c745bfb80b85f2198f3dce86fb677365eb23578f5ee2b3d64f52b158e822831",
    "isl_bilstm.pt": "639dcd1f0caa3aa9c4bf2b11090edd1ef7a2c547a8fc7eb558fdfd9ec418eeb9",
    "classes.json": "ca87b055bf445794339063bdd82f1f8fe9ec43cb2be0fb606e9487ad9ff27ebb",
}
_NUM_CLASSES = 28
_MODEL_TYPES = ("transformer", "bilstm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(model_type: str) -> dict:
    name = settings.transformer_checkpoint if model_type == "transformer" else settings.bilstm_checkpoint
    return torch.load(_MODELS / name, map_location="cpu", weights_only=False)


def _classes_json() -> list[str]:
    return json.loads((_MODELS / "classes.json").read_text(encoding="utf-8"))["classes"]


@pytest.mark.parametrize("filename", sorted(_EXPECTED_SHA256))
def test_shipped_model_files_are_unchanged(filename):
    path = _MODELS / filename
    assert path.exists(), f"{path} is missing"
    assert _sha256(path) == _EXPECTED_SHA256[filename], f"{filename} differs from the audited checkpoint"


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_checkpoint_metadata_matches_runtime_configuration(model_type):
    ckpt = _checkpoint(model_type)
    assert ckpt["model_type"] == model_type
    assert ckpt["input_dim"] == settings.feature_dim == 330
    assert ckpt["config"]["sequence_length"] == settings.sequence_length == 32
    assert ckpt["class_names"] == _classes_json()
    assert len(ckpt["class_names"]) == _NUM_CLASSES == len(set(ckpt["class_names"]))
    assert settings.unknown_label not in ckpt["class_names"]
    assert ckpt["dataset_name"] == "isl_include_subset28"


@pytest.mark.skipif(not (_DATASET / "class_mapping.json").exists(), reason="prepared dataset not present")
def test_class_order_matches_the_dataset_mapping():
    mapping = json.loads((_DATASET / "class_mapping.json").read_text(encoding="utf-8"))["classes"]
    assert mapping == _classes_json()


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_loader_uses_trained_weights_not_a_random_fallback(model_type):
    bundle = model_loader.load_model(model_type)
    assert bundle.is_trained, "fell back to the untrained demo model"
    assert bundle.checkpoint_path and Path(bundle.checkpoint_path).name.startswith("isl_")
    assert bundle.num_classes == _NUM_CLASSES and bundle.input_dim == 330

    # Every parameter must equal the checkpoint's tensor, bit for bit.
    loaded = bundle.classifier.model.state_dict()
    saved = _checkpoint(model_type)["state_dict"]
    assert loaded.keys() == saved.keys()
    for key, tensor in saved.items():
        assert torch.equal(loaded[key].cpu(), tensor.cpu()), key


def test_live_bundle_is_the_trained_ensemble(monkeypatch):
    monkeypatch.setattr(settings, "live_ensemble", True)
    bundle = model_loader.with_live_ensemble(model_loader.load_model("transformer"))
    assert bundle.is_trained
    assert isinstance(bundle.classifier, EnsembleClassifier)
    assert bundle.ensemble_members == ["transformer", "bilstm"]
    info = bundle.info()
    assert info["is_trained"] is True and info["num_classes"] == _NUM_CLASSES and info["input_dim"] == 330


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_trained_model_outputs_are_finite_probabilities(model_type):
    bundle = model_loader.load_model(model_type)
    model = bundle.classifier.model
    torch.manual_seed(0)
    x = torch.randn(4, settings.sequence_length, settings.feature_dim)
    mask = torch.ones(4, settings.sequence_length, dtype=torch.bool)
    mask[1, :10] = False  # a left-padded sequence, as the live buffer produces
    with torch.inference_mode():
        logits = model(x, mask=mask)
    assert logits.shape == (4, _NUM_CLASSES)
    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    result = bundle.classifier.predict(x[:1], mask=mask[:1], top_k=5)
    assert len(result.top_labels) == 5 and set(result.top_labels) <= set(bundle.class_names)
    assert all(0.0 <= p <= 1.0 for p in result.top_probs)
    assert result.top_probs == sorted(result.top_probs, reverse=True)


def test_inference_is_deterministic():
    bundle = model_loader.load_model("transformer")
    x = torch.randn(1, settings.sequence_length, settings.feature_dim)
    a = bundle.classifier.predict(x, top_k=3)
    b = bundle.classifier.predict(x, top_k=3)
    assert a == b


@pytest.mark.slow
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
@pytest.mark.skipif(not (_DATASET / "test").exists(), reason="held-out test split not present")
def test_recorded_test_split_results_are_reproduced(model_type):
    """Re-scores the held-out test split in memory with training/evaluate.py's
    own ``evaluate`` (without writing any file) and compares it with the
    results recorded after training. No tuning against the test set happens
    here — this only proves nothing upstream of the model changed."""
    recorded_path = _EVAL_DIR / f"evaluation_results_{model_type}.json"
    if not recorded_path.exists():
        pytest.skip("recorded evaluation results not present")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))

    from torch.utils.data import DataLoader

    from training.dataset import LandmarkDataset, collate_fn
    from training.evaluate import evaluate

    bundle = model_loader.load_model(model_type)
    test_ds = LandmarkDataset(_DATASET, "test", settings.sequence_length, class_names=bundle.class_names)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    results = evaluate(bundle.classifier.model, loader, "cpu", bundle.num_classes)

    assert results["num_test_samples"] == recorded["num_test_samples"] == 150
    assert results["confusion_matrix"] == recorded["confusion_matrix"]
    assert results["accuracy"] == pytest.approx(recorded["accuracy"], abs=1e-9)
    assert results["f1_macro"] == pytest.approx(recorded["f1_macro"], abs=1e-9)
    assert results["top_5_accuracy"] == pytest.approx(recorded["top_5_accuracy"], abs=1e-9)


_HELLO_VIDEO = _VIDEOS / "Greetings" / "48. Hello" / "MVI_0029.MOV"


@pytest.mark.slow
@pytest.mark.skipif(not _HELLO_VIDEO.exists(), reason="INCLUDE source video not present")
def test_live_feature_extraction_reproduces_the_training_features():
    """Training/inference consistency: today's FeatureExtractor (the code the
    live server runs) re-extracts a test-split video into exactly the
    features the model was evaluated on."""
    from training.extract_landmarks import extract_video_fresh

    stored = sorted((_DATASET / "test" / "HELLO").glob("Greetings_48_Hello_MVI_0029_MOV__*.npy"))
    if not stored:
        pytest.skip("matching .npy not present")
    expected = np.load(stored[0])
    actual = extract_video_fresh(_HELLO_VIDEO)
    assert actual is not None
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
