"""Build a train/validation/test landmark dataset from a real-dataset
adapter (training/datasets/), respecting whatever split the source dataset
itself defines (Section 3/4: don't silently re-shuffle official splits).

This is the adapter-driven counterpart to
``training.extract_landmarks`` + ``training.prepare_dataset`` (which are
for the manual `datasets/raw/<CLASS_NAME>/*.mp4` + random-split workflow).
It reuses the exact same feature-extraction code path
(``training.extract_landmarks.extract_video_fresh``, which itself uses
``app.vision.feature_extractor.FeatureExtractor`` — the same extractor
real-time inference uses) so there is only ever one feature-extraction
implementation in the codebase.

Usage:
    python -m training.build_dataset_from_adapter --config experiments/isl_include_subset28.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from app.config import BACKEND_DIR
from app.utils.logging import get_logger
from app.vision.feature_extractor import FeatureExtractor
from training.datasets.registry import get_adapter_class
from training.extract_landmarks import extract_video_fresh

logger = get_logger(__name__)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BACKEND_DIR / path


def _check_unique_output_paths(samples) -> None:
    """Fail loudly if two samples would write the same .npy file. Compared
    case-insensitively, since Windows filesystems are. Without this check the
    second sample of a colliding pair is silently skipped as 'already done'."""
    seen: dict[str, str] = {}
    collisions = []
    for s in samples:
        key = f"{s.split}/{s.label}/{s.sample_id}".lower()
        if key in seen:
            collisions.append((seen[key], str(s.video_path)))
        else:
            seen[key] = str(s.video_path)
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} sample(s) map to the same output file; fix the adapter's sample_id: {collisions[:5]}"
        )


def run(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg["dataset"]

    adapter_cls = get_adapter_class(ds_cfg["adapter"])
    adapter = adapter_cls(
        video_root=_resolve(ds_cfg["video_root"]),
        metadata_dir=_resolve(ds_cfg["metadata_dir"]),
        subset_classes=ds_cfg.get("subset_classes"),
        max_samples_per_class=ds_cfg.get("max_samples_per_class"),
        include_50_only=ds_cfg.get("include_50_only", False),
    )

    samples = adapter.list_samples()
    if not samples:
        raise RuntimeError(
            "The dataset adapter resolved zero samples. Check that the "
            "video files are downloaded and video_root/metadata_dir in "
            f"{config_path} are correct."
        )

    output_dir = BACKEND_DIR / ds_cfg["dataset_path"] / ds_cfg["name"]
    _check_unique_output_paths(samples)
    for split in ("train", "validation", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    probe = FeatureExtractor()  # fail fast before looping; each video gets its own instance below
    available = probe.available
    probe.close()
    if not available:
        raise RuntimeError(
            "MediaPipe model unavailable — run scripts/download_models.ps1 "
            "(or .sh) first."
        )

    split_class_counts: dict[str, Counter] = defaultdict(Counter)
    failed_samples: list[str] = []
    reused = 0

    for i, sample in enumerate(samples, 1):
        class_dir = output_dir / sample.split / sample.label
        out_path = class_dir / f"{sample.sample_id}.npy"
        # Resumable: an interrupted build keeps every sample already written.
        # np.save writes to a temp name first, so an existing .npy is complete.
        if out_path.exists():
            reused += 1
            split_class_counts[sample.split][sample.label] += 1
            continue

        sequence = extract_video_fresh(sample.video_path)  # fresh MediaPipe instance per video
        if sequence is None:
            failed_samples.append(sample.sample_id)
            continue

        class_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.npy")
        np.save(tmp_path, sequence)
        tmp_path.replace(out_path)
        split_class_counts[sample.split][sample.label] += 1
        if i % 25 == 0:
            logger.info("Extracted %d/%d samples (%d reused from a previous run)", i, len(samples), reused)

    all_classes = sorted({s.label for s in samples if s.sample_id not in failed_samples})
    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"classes": all_classes}, f, indent=2)

    report = {
        "dataset_name": ds_cfg["name"],
        "adapter": ds_cfg["adapter"],
        "source_info": adapter.info().__dict__,
        "total_samples_resolved": len(samples),
        "total_extracted": sum(sum(c.values()) for c in split_class_counts.values()),
        "total_failed": len(failed_samples),
        "failed_sample_ids": failed_samples,
        "classes": all_classes,
        "num_classes": len(all_classes),
        "split_counts": {split: dict(counts) for split, counts in split_class_counts.items()},
        "split_totals": {split: sum(counts.values()) for split, counts in split_class_counts.items()},
    }
    report_path = output_dir / "build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Dataset build complete: %d/%d samples extracted (%d failed), %d classes -> %s",
        report["total_extracted"], report["total_samples_resolved"], report["total_failed"],
        report["num_classes"], output_dir,
    )
    for split, total in report["split_totals"].items():
        logger.info("  %s: %d samples", split, total)
    if failed_samples:
        logger.warning("%d samples failed extraction (see %s): %s",
                        len(failed_samples), report_path, failed_samples[:10])

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a landmark dataset from a real-dataset adapter.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
