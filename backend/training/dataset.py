"""PyTorch Dataset for pre-extracted landmark sequences.

Expects the layout produced by ``training/prepare_dataset.py``:

    <dataset_path>/<dataset_name>/<split>/<CLASS_NAME>/*.npy

Each ``.npy`` file holds one sample's raw (variable-length) feature
sequence with shape ``(T, feature_dim)``, as produced by
``training/extract_landmarks.py`` — ``feature_dim`` already includes
velocity features if ``settings.use_velocity_features`` was enabled at
extraction time (see ``app.vision.feature_extractor``), so this dataset
does not recompute them. Uniform resampling/padding to a fixed ``sequence_length``
(``app.vision.preprocessing.resample_or_pad``) happens here (not at extraction time) so the same extracted data can be
reused with different sequence-length configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from app.vision.preprocessing import resample_or_pad


class LandmarkDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        sequence_length: int,
        class_names: Optional[List[str]] = None,
        augment: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split_dir = self.dataset_dir / split
        self.sequence_length = sequence_length
        self.augment = augment

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. Run "
                "training/prepare_dataset.py first."
            )

        self.class_names = class_names or self._load_class_names()
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for class_dir in sorted(self.split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in self.class_to_idx:
                continue
            label_idx = self.class_to_idx[class_dir.name]
            for npy_path in sorted(class_dir.glob("*.npy")):
                self.samples.append((npy_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No .npy samples found under {self.split_dir}")

    def _load_class_names(self) -> List[str]:
        mapping_path = self.dataset_dir / "class_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"class_mapping.json not found in {self.dataset_dir}. Run "
                "training/prepare_dataset.py first."
            )
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)["classes"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        sequence = np.load(path).astype(np.float32)  # (T, feature_dim)

        if self.augment is not None:
            sequence = self.augment(sequence)

        # Long clips are uniformly sampled across their full length (not
        # truncated to the end); short ones are start-padded and masked.
        sequence, mask = resample_or_pad(sequence, self.sequence_length)

        return torch.from_numpy(sequence), torch.from_numpy(mask), label


def collate_fn(batch):
    sequences, masks, labels = zip(*batch)
    return (
        torch.stack(sequences),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.long),
    )
