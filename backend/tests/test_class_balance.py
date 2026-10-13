"""Tests for class-imbalance handling (Section 8)."""

from __future__ import annotations

import json

import numpy as np
import torch

from training.dataset import LandmarkDataset
from training.train import _compute_class_weights


def test_inverse_frequency_class_weights(tmp_path):
    classes = ["HELLO", "YES"]
    (tmp_path / "class_mapping.json").write_text(json.dumps({"classes": classes}))

    # Deliberately imbalanced: 6 HELLO samples, 2 YES samples.
    for label, count in (("HELLO", 6), ("YES", 2)):
        class_dir = tmp_path / "train" / label
        class_dir.mkdir(parents=True)
        for i in range(count):
            np.save(class_dir / f"s{i}.npy", np.ones((5, 4), dtype=np.float32))

    dataset = LandmarkDataset(tmp_path, "train", sequence_length=5)
    weights = _compute_class_weights(dataset, num_classes=2)

    # weight_c = N / (num_classes * count_c): HELLO -> 8/(2*6)=0.667, YES -> 8/(2*2)=2.0
    hello_idx = dataset.class_to_idx["HELLO"]
    yes_idx = dataset.class_to_idx["YES"]
    assert torch.isclose(weights[hello_idx], torch.tensor(8 / 12), atol=1e-4)
    assert torch.isclose(weights[yes_idx], torch.tensor(8 / 4), atol=1e-4)
    # The minority class must get a strictly larger weight than the majority class.
    assert weights[yes_idx] > weights[hello_idx]
