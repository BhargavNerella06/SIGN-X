"""Unified classifier wrapper around the Transformer/BiLSTM backbones.

Gives the rest of the codebase (inference, training, evaluation) one
consistent interface regardless of which architecture is active, and
centralizes the "logits -> probabilities -> top-k" logic so it is not
duplicated across predictor.py / evaluate.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.bilstm import SignBiLSTM
from app.models.transformer import SignTransformer


@dataclass
class ClassificationResult:
    top_indices: List[int]
    top_labels: List[str]
    top_probs: List[float]

    @property
    def best_label(self) -> str:
        return self.top_labels[0]

    @property
    def best_prob(self) -> float:
        return self.top_probs[0]


def build_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    *,
    embedding_dim: int = 128,
    transformer_layers: int = 4,
    transformer_heads: int = 4,
    transformer_ff_dim: int = 256,
    bilstm_hidden_dim: int = 128,
    bilstm_layers: int = 2,
    dropout: float = 0.2,
    max_seq_len: int = 512,
) -> nn.Module:
    """Factory that builds either backbone from plain config values, so
    training and inference always construct the model the same way."""
    if model_type == "transformer":
        return SignTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    if model_type == "bilstm":
        return SignBiLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=bilstm_hidden_dim,
            num_layers=bilstm_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type!r} (expected 'transformer' or 'bilstm')")


class SignClassifier:
    """Wraps a backbone model + class label list with a convenient
    ``predict`` method returning top-k labels and calibrated probabilities.
    """

    def __init__(self, model: nn.Module, class_names: List[str], device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = device

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None, top_k: int = 3) -> ClassificationResult:
        """
        Args:
            x: (batch, seq_len, input_dim) — batch is normally 1 for
               real-time inference.
            mask: optional (batch, seq_len) bool tensor.
        """
        x = x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        logits = self.model(x, mask=mask)
        probs = F.softmax(logits, dim=-1)[0]  # first (only) item in batch

        k = min(top_k, probs.shape[0])
        top_probs, top_indices = torch.topk(probs, k)
        return ClassificationResult(
            top_indices=top_indices.tolist(),
            top_labels=[self.class_names[i] for i in top_indices.tolist()],
            top_probs=[round(p, 4) for p in top_probs.tolist()],
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class EnsembleClassifier:
    """Averages the softmax probabilities of several ``SignClassifier``s that
    share one class list, behind the same ``predict`` interface.

    Used for live inference: the Transformer+BiLSTM average is more accurate
    and less noisy window to window than either model alone (see
    docs/performance.md), for ~3-4 ms extra per inference on CPU.
    """

    def __init__(self, members: List[SignClassifier]):
        if not members:
            raise ValueError("EnsembleClassifier needs at least one member")
        class_names = members[0].class_names
        if any(m.class_names != class_names for m in members):
            raise ValueError("Ensemble members must share the same class list (same order)")
        self.members = members
        self.class_names = class_names
        self.device = members[0].device

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None, top_k: int = 3) -> ClassificationResult:
        probs = None
        for member in self.members:
            xm = x.to(member.device)
            mm = mask.to(member.device) if mask is not None else None
            p = F.softmax(member.model(xm, mask=mm), dim=-1)[0].cpu()
            probs = p if probs is None else probs + p
        probs = probs / len(self.members)

        k = min(top_k, probs.shape[0])
        top_probs, top_indices = torch.topk(probs, k)
        return ClassificationResult(
            top_indices=top_indices.tolist(),
            top_labels=[self.class_names[i] for i in top_indices.tolist()],
            top_probs=[round(p, 4) for p in top_probs.tolist()],
        )

    @property
    def num_parameters(self) -> int:
        return sum(m.num_parameters for m in self.members)
