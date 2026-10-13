"""Tests for training.datasets (real-dataset adapters). Uses tiny synthetic
fixtures — never the actual multi-GB INCLUDE dataset."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.datasets.include import IncludeAdapter, clean_label
from training.datasets.registry import available_adapters, get_adapter_class


# --- clean_label() ---

@pytest.mark.parametrize("raw,expected", [
    ("48. Hello", "HELLO"),
    ("46. you (plural)", "YOU_PLURAL"),
    ("49. How are you", "HOW_ARE_YOU"),
    ("Ex. Monsoon", "MONSOON"),
    ("1. loud", "LOUD"),
    ("8. Race (ethnicity)", "RACE_ETHNICITY"),
])
def test_clean_label_normalizes_include_labels(raw, expected):
    assert clean_label(raw) == expected


# --- registry ---

def test_registry_lists_include_and_isl_csltr():
    assert "include" in available_adapters()
    assert "isl_csltr" in available_adapters()


def test_registry_raises_for_unknown_adapter():
    with pytest.raises(ValueError):
        get_adapter_class("not_a_real_adapter")


# --- IncludeAdapter ---

def _write_parquet(path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


@pytest.fixture
def fake_include_dataset(tmp_path):
    """A tiny fake INCLUDE dataset: 2 classes, a few samples each, spread
    across the official train/val/test split, plus one row whose video
    file is deliberately missing (to test graceful skipping)."""
    metadata_dir = tmp_path / "meta"
    video_root = tmp_path / "videos"
    metadata_dir.mkdir()
    video_root.mkdir()

    def touch_video(rel_path):
        p = video_root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake video bytes")

    train_rows = [
        {"parent_label": "Greetings", "label": "48. Hello", "video_path": "Greetings/48. Hello/a.MOV", "include_50": True},
        {"parent_label": "Greetings", "label": "48. Hello", "video_path": "Greetings/48. Hello/b.MOV", "include_50": False},
        {"parent_label": "Pronouns", "label": "40. I", "video_path": "Pronouns/40. I/c.MOV", "include_50": True},
        {"parent_label": "Pronouns", "label": "40. I", "video_path": "Pronouns/40. I/missing.MOV", "include_50": False},  # not touched -> missing on disk
        {"parent_label": "Colours", "label": "47. Red", "video_path": "Colours/47. Red/d.MOV", "include_50": False},  # excluded via subset_classes in most tests
    ]
    val_rows = [
        {"parent_label": "Greetings", "label": "48. Hello", "video_path": "Greetings/48. Hello/e.MOV", "include_50": True},
    ]
    test_rows = [
        {"parent_label": "Pronouns", "label": "40. I", "video_path": "Pronouns/40. I/f.MOV", "include_50": True},
    ]

    for row in train_rows + val_rows + test_rows:
        if "missing" not in row["video_path"]:
            touch_video(row["video_path"])

    _write_parquet(metadata_dir / "train.parquet", train_rows)
    _write_parquet(metadata_dir / "val.parquet", val_rows)
    _write_parquet(metadata_dir / "test.parquet", test_rows)

    return {"metadata_dir": metadata_dir, "video_root": video_root}


def test_include_adapter_resolves_samples_and_maps_split_names(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
        subset_classes=["HELLO", "I"],
    )
    samples = adapter.list_samples()
    splits = {s.split for s in samples}
    assert splits == {"train", "validation", "test"}  # 'val' mapped to 'validation'
    labels = {s.label for s in samples}
    assert labels == {"HELLO", "I"}


def test_include_adapter_skips_missing_video_files(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
        subset_classes=["I"],
    )
    samples = adapter.list_samples()
    # 3 rows reference class "I" (c.MOV, missing.MOV, f.MOV); missing.MOV has no file on disk.
    assert len(samples) == 2
    assert all(s.video_path.exists() for s in samples)


def test_include_adapter_filters_by_subset_classes(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
        subset_classes=["HELLO"],
    )
    samples = adapter.list_samples()
    assert all(s.label == "HELLO" for s in samples)
    assert "RED" not in {s.label for s in samples}


def test_include_adapter_respects_max_samples_per_class(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
        subset_classes=["HELLO"],
        max_samples_per_class=1,
    )
    samples = adapter.list_samples()
    assert len(samples) == 1


def test_include_adapter_include_50_only_filter(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
        subset_classes=["HELLO"],
        include_50_only=True,
    )
    samples = adapter.list_samples()
    # Of HELLO's 3 rows (a: include_50=True, b: False, e: True), only a and e qualify.
    assert len(samples) == 2


def test_include_adapter_missing_metadata_raises_clear_error(tmp_path):
    adapter = IncludeAdapter(video_root=tmp_path, metadata_dir=tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        adapter.list_samples()


def test_include_adapter_info_reports_license_and_source(fake_include_dataset):
    adapter = IncludeAdapter(
        video_root=fake_include_dataset["video_root"],
        metadata_dir=fake_include_dataset["metadata_dir"],
    )
    info = adapter.info()
    assert info.license == "CC-BY-4.0"
    assert "AI4Bharat" in info.source


def test_include_adapter_deduplicates_videos_listed_in_multiple_splits(tmp_path):
    """The real INCLUDE metadata lists some videos in several splits (e.g.
    both train and test). Each video must be kept once, in its held-out
    split, so no evaluated video is ever trained on."""
    metadata_dir = tmp_path / "meta"
    video_root = tmp_path / "videos"
    metadata_dir.mkdir()
    for name in ("a", "b", "c"):
        p = video_root / "Greetings/48. Hello" / f"{name}.MOV"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    def row(name):
        return {"parent_label": "Greetings", "label": "48. Hello",
                "video_path": f"Greetings/48. Hello/{name}.MOV", "include_50": False}

    _write_parquet(metadata_dir / "train.parquet", [row("a"), row("a"), row("b"), row("c")])
    _write_parquet(metadata_dir / "val.parquet", [row("b")])
    _write_parquet(metadata_dir / "test.parquet", [row("a")])

    samples = IncludeAdapter(video_root=video_root, metadata_dir=metadata_dir).list_samples()
    by_video = {s.video_path.name: s.split for s in samples}
    assert len(samples) == 3
    assert by_video == {"a.MOV": "test", "b.MOV": "validation", "c.MOV": "train"}


@pytest.mark.parametrize("path_a, path_b", [
    # Real INCLUDE file pairs whose plain slugs used to collide.
    ("Greetings/54. Good night/MVI_0057.MOV", "Greetings/54. Good night/MVI_0057_.MOV"),
    ("Greetings/55. Thank you/MVI_0060.MOV", "Greetings/55. Thank you/MVI_0060_.MOV"),
])
def test_slugify_path_does_not_collide_on_punctuation_variants(path_a, path_b):
    from training.datasets.include import _slugify_path

    assert _slugify_path(path_a).lower() != _slugify_path(path_b).lower()


def test_slugify_path_is_deterministic_and_filesystem_safe():
    from training.datasets.include import _slugify_path

    path = "Colours/47. Red/MVI_3719.MOV"
    assert _slugify_path(path) == _slugify_path(path)
    assert _slugify_path(path).startswith("Colours_47_Red_MVI_3719_MOV__")
    assert all(c.isalnum() or c == "_" for c in _slugify_path(path))


def test_build_dataset_rejects_colliding_output_paths(tmp_path):
    from training.build_dataset_from_adapter import _check_unique_output_paths
    from training.datasets.base import DatasetSample

    a = DatasetSample("Same_ID", tmp_path / "a.MOV", "HELLO", "train")
    b = DatasetSample("same_id", tmp_path / "b.MOV", "HELLO", "train")  # differs only in case
    with pytest.raises(RuntimeError, match="same output file"):
        _check_unique_output_paths([a, b])
    _check_unique_output_paths([a, DatasetSample("same_id", tmp_path / "b.MOV", "HELLO", "test")])
