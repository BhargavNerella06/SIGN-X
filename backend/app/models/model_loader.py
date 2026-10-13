"""Model loading and lifecycle management.

Loads a trained checkpoint (``models/isl_transformer.pt`` or
``models/isl_bilstm.pt`` + ``models/classes.json``) exactly once and hands
out a shared ``SignClassifier`` instance — the model is never reloaded per
request/frame.

If no trained checkpoint exists yet (the common case before any training
run has been done), this module builds an **untrained, randomly
initialized** model over the configured vocabulary instead of crashing.
This keeps the full pipeline (camera -> landmarks -> buffer -> model ->
smoothing -> UI) runnable end-to-end for development and demoing the
architecture, while being explicit — via ``ModelBundle.is_trained`` and the
``/api/model/info`` endpoint — that its predictions are not meaningful
until a real checkpoint is trained and placed in ``backend/models/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from app.config import settings
from app.models.classifier import EnsembleClassifier, SignClassifier, build_model
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    classifier: SignClassifier | EnsembleClassifier
    model_type: str
    input_dim: int
    num_classes: int
    class_names: List[str]
    device: str
    is_trained: bool
    checkpoint_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    dataset_name: Optional[str] = None
    # Model types whose probabilities are averaged for live inference; empty
    # when only ``model_type`` runs. ``metrics`` stay the primary model's.
    ensemble_members: List[str] = field(default_factory=list)

    def info(self) -> dict:
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "num_parameters": self.classifier.num_parameters,
            "class_names": self.class_names,
            "metrics": self.metrics,
            "dataset_name": self.dataset_name,
            "ensemble_members": self.ensemble_members,
        }


def _checkpoint_filename(model_type: str) -> str:
    return settings.transformer_checkpoint if model_type == "transformer" else settings.bilstm_checkpoint


def _load_classes(models_dir: Path) -> List[str]:
    classes_path = models_dir / settings.classes_file
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["classes"] if isinstance(data, dict) else data
    return list(settings.vocabulary)


def load_model(model_type: Optional[str] = None) -> ModelBundle:
    """Load (or build a demo fallback for) the configured model type."""
    model_type = model_type or settings.model_type
    device = settings.resolved_device
    models_dir = settings.models_dir
    checkpoint_path = models_dir / _checkpoint_filename(model_type)

    if checkpoint_path.exists():
        try:
            return _load_trained(checkpoint_path, device)
        except Exception:
            logger.exception(
                "Failed to load checkpoint at %s — falling back to an "
                "untrained demo model.", checkpoint_path,
            )

    return _build_demo(model_type, device, models_dir)


def _load_trained(checkpoint_path: Path, device: str) -> ModelBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    model_type = checkpoint["model_type"]
    cfg = checkpoint.get("config", {})

    model = build_model(
        model_type=model_type,
        input_dim=checkpoint["input_dim"],
        num_classes=len(class_names),
        embedding_dim=cfg.get("embedding_dim", settings.embedding_dim),
        transformer_layers=cfg.get("transformer_layers", settings.transformer_layers),
        transformer_heads=cfg.get("transformer_heads", settings.transformer_heads),
        transformer_ff_dim=cfg.get("transformer_ff_dim", settings.transformer_ff_dim),
        bilstm_hidden_dim=cfg.get("bilstm_hidden_dim", settings.bilstm_hidden_dim),
        bilstm_layers=cfg.get("bilstm_layers", settings.bilstm_layers),
        dropout=cfg.get("dropout", settings.dropout),
        max_seq_len=cfg.get("sequence_length", settings.sequence_length),
    )
    model.load_state_dict(checkpoint["state_dict"])

    classifier = SignClassifier(model, class_names, device=device)
    logger.info(
        "Loaded TRAINED %s checkpoint from %s (%d classes, %d params)",
        model_type, checkpoint_path, len(class_names), classifier.num_parameters,
    )
    return ModelBundle(
        classifier=classifier,
        model_type=model_type,
        input_dim=checkpoint["input_dim"],
        num_classes=len(class_names),
        class_names=class_names,
        device=device,
        is_trained=True,
        checkpoint_path=str(checkpoint_path),
        metrics=checkpoint.get("metrics", {}),
        dataset_name=checkpoint.get("dataset_name") or None,
    )


def _build_demo(model_type: str, device: str, models_dir: Path) -> ModelBundle:
    class_names = _load_classes(models_dir) + [settings.unknown_label]
    input_dim = settings.feature_dim

    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=len(class_names),
        embedding_dim=settings.embedding_dim,
        transformer_layers=settings.transformer_layers,
        transformer_heads=settings.transformer_heads,
        transformer_ff_dim=settings.transformer_ff_dim,
        bilstm_hidden_dim=settings.bilstm_hidden_dim,
        bilstm_layers=settings.bilstm_layers,
        dropout=settings.dropout,
        max_seq_len=settings.sequence_length,
    )
    classifier = SignClassifier(model, class_names, device=device)
    logger.warning(
        "No trained checkpoint found for model_type=%s. Running with an "
        "UNTRAINED demo model (%d params) so the pipeline is runnable end "
        "to end — predictions are NOT meaningful until you run the "
        "training pipeline (see backend/training/) and place a real "
        "checkpoint in %s.", model_type, classifier.num_parameters, models_dir,
    )
    return ModelBundle(
        classifier=classifier,
        model_type=model_type,
        input_dim=input_dim,
        num_classes=len(class_names),
        class_names=class_names,
        device=device,
        is_trained=False,
        checkpoint_path=None,
        metrics={},
    )


def with_live_ensemble(bundle: ModelBundle) -> ModelBundle:
    """Average ``bundle``'s model with the other trained architecture when
    ``settings.live_ensemble`` is on. Falls back to the single model (with a
    warning) unless both checkpoints are trained on the same classes and
    feature layout — an ensemble must never mix label spaces."""
    if not settings.live_ensemble or not bundle.is_trained:
        return bundle
    other_type = "bilstm" if bundle.model_type == "transformer" else "transformer"
    other = load_model(other_type)
    if not other.is_trained or other.class_names != bundle.class_names or other.input_dim != bundle.input_dim:
        logger.warning(
            "live_ensemble is on but the %s checkpoint is missing, untrained or incompatible "
            "(classes/input_dim differ) — using %s alone.", other_type, bundle.model_type,
        )
        return bundle
    bundle.classifier = EnsembleClassifier([bundle.classifier, other.classifier])
    bundle.ensemble_members = [bundle.model_type, other.model_type]
    logger.info("Live ensemble: averaging %s + %s probabilities", bundle.model_type, other.model_type)
    return bundle


class _ModelRegistry:
    """Process-wide singleton so the model is loaded exactly once."""

    _bundle: Optional[ModelBundle] = None

    @classmethod
    def get(cls) -> ModelBundle:
        if cls._bundle is None:
            cls._bundle = with_live_ensemble(load_model())
        return cls._bundle

    @classmethod
    def reload(cls, model_type: Optional[str] = None) -> ModelBundle:
        cls._bundle = with_live_ensemble(load_model(model_type))
        return cls._bundle


def get_model_bundle() -> ModelBundle:
    return _ModelRegistry.get()


def reload_model_bundle(model_type: Optional[str] = None) -> ModelBundle:
    return _ModelRegistry.reload(model_type)
