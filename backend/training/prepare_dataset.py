"""Split extracted landmark sequences into train/validation/test and write
the class mapping used by training, evaluation and the dataset info API.

Input:  backend/datasets/processed/<dataset_name>/<CLASS_NAME>/*.npy
        (produced by training/extract_landmarks.py)

Output: backend/datasets/<dataset_name>/
            class_mapping.json
            train/<CLASS_NAME>/*.npy
            validation/<CLASS_NAME>/*.npy
            test/<CLASS_NAME>/*.npy

Usage:
    python -m training.prepare_dataset --dataset_name isl_demo
    python -m training.prepare_dataset --dataset_name isl_demo --train_ratio 0.7 --val_ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = max(1, int(round(n * train_ratio))) if n >= 3 else n
    n_val = max(1, int(round(n * val_ratio))) if n - n_train >= 2 else max(0, n - n_train - 1)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def run(processed_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found: {processed_dir}\n"
            "Run training/extract_landmarks.py first."
        )

    class_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found under {processed_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "validation", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    class_names = [d.name for d in class_dirs]
    split_counts = {"train": 0, "validation": 0, "test": 0}

    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            logger.warning("No samples for class %s, skipping", class_dir.name)
            continue

        train_idx, val_idx, test_idx = _split_indices(len(files), train_ratio, val_ratio, seed)
        for split_name, idx_list in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
            split_class_dir = output_dir / split_name / class_dir.name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for i in idx_list:
                shutil.copy2(files[i], split_class_dir / files[i].name)
                split_counts[split_name] += 1

    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, indent=2)

    logger.info(
        "Dataset prepared at %s | classes=%d | train=%d validation=%d test=%d",
        output_dir, len(class_names), split_counts["train"], split_counts["validation"], split_counts["test"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split extracted landmarks into train/validation/test.")
    parser.add_argument("--dataset_name", type=str, default="isl_demo")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else settings.dataset_path / "processed" / args.dataset_name
    output_dir = Path(args.output_dir) if args.output_dir else settings.dataset_path / args.dataset_name

    run(processed_dir, output_dir, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
