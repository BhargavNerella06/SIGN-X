"""Tests for training.dataset.LandmarkDataset (PyTorch Dataset over
pre-extracted landmark sequences) using small synthetic .npy fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from training.dataset import LandmarkDataset, collate_fn


def _make_dataset_dir(tmp_path, feature_dim=6):
    """train/validation/test with 2 classes; deliberately variable-length
    sequences to exercise pad/truncate."""
    classes = ["HELLO", "YES"]
    (tmp_path / "class_mapping.json").write_text(json.dumps({"classes": classes}))

    lengths = {"train": {"HELLO": [10, 20], "YES": [15]}, "validation": {"HELLO": [12]}, "test": {"YES": [8, 40]}}
    for split, per_class in lengths.items():
        for label, seq_lens in per_class.items():
            class_dir = tmp_path / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for i, seq_len in enumerate(seq_lens):
                seq = np.random.rand(seq_len, feature_dim).astype(np.float32) + 0.1  # avoid all-zero rows
                np.save(class_dir / f"sample_{i}.npy", seq)
    return tmp_path, classes


def test_dataset_loads_correct_sample_count(tmp_path):
    dataset_dir, _ = _make_dataset_dir(tmp_path)
    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length=16)
    assert len(train_ds) == 3  # 2 HELLO + 1 YES


def test_dataset_pads_and_truncates_to_sequence_length(tmp_path):
    dataset_dir, _ = _make_dataset_dir(tmp_path)
    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length=16)
    for i in range(len(train_ds)):
        sequence, mask, _ = train_ds[i]
        assert sequence.shape == (16, 6)
        assert mask.shape == (16,)


def test_dataset_mask_matches_real_frame_count_when_padded(tmp_path):
    dataset_dir, _ = _make_dataset_dir(tmp_path)
    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length=16)
    # sample_0.npy for HELLO has 10 real frames -> padded to 16 -> 6 masked-out
    for path, label in train_ds.samples:
        if path.name == "sample_0.npy" and path.parent.name == "HELLO":
            idx = train_ds.samples.index((path, label))
            _, mask, _ = train_ds[idx]
            assert mask.sum().item() == 10


def test_dataset_class_names_shared_across_splits(tmp_path):
    dataset_dir, classes = _make_dataset_dir(tmp_path)
    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length=16)
    val_ds = LandmarkDataset(dataset_dir, "validation", sequence_length=16, class_names=train_ds.class_names)
    assert train_ds.class_names == val_ds.class_names == classes


def test_dataset_missing_split_raises(tmp_path):
    dataset_dir, _ = _make_dataset_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        LandmarkDataset(dataset_dir, "does_not_exist", sequence_length=16)


def test_dataset_missing_class_mapping_raises(tmp_path):
    (tmp_path / "train" / "HELLO").mkdir(parents=True)
    np.save(tmp_path / "train" / "HELLO" / "a.npy", np.ones((5, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        LandmarkDataset(tmp_path, "train", sequence_length=16)


def test_collate_fn_batches_correctly(tmp_path):
    dataset_dir, _ = _make_dataset_dir(tmp_path)
    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length=16)
    batch = [train_ds[i] for i in range(len(train_ds))]
    sequences, masks, labels = collate_fn(batch)
    assert sequences.shape == (3, 16, 6)
    assert masks.shape == (3, 16)
    assert labels.dtype == torch.long


def test_dataset_uniformly_samples_long_clips_instead_of_truncating(tmp_path):
    import json

    root = tmp_path / "ds"
    (root / "train" / "HELLO").mkdir(parents=True)
    (root / "class_mapping.json").write_text(json.dumps({"classes": ["HELLO"]}))
    seq = np.repeat(np.arange(1, 65, dtype=np.float32)[:, None], 6, axis=1)  # 64 frames, value = frame no.
    np.save(root / "train" / "HELLO" / "a.npy", seq)
    x, mask, label = LandmarkDataset(root, "train", sequence_length=32)[0]
    assert tuple(x.shape) == (32, 6) and bool(mask.all()) and label == 0
    assert x[0, 0].item() == 1 and x[-1, 0].item() == 64  # start of the sign is kept, not dropped
