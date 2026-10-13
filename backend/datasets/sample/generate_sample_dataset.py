"""Generate a small SYNTHETIC landmark dataset for pipeline testing/demos.

This does NOT contain any real Indian Sign Language data. It exists purely
so the training/evaluation pipeline (dataset.py, train.py, evaluate.py) can
be exercised end-to-end without a real dataset present, per the project
rule against inventing/faking datasets: this generator is explicit about
being synthetic, and a model trained on it learns nothing about actual ISL
signs. Replace it with training/extract_landmarks.py output from a real
dataset (see docs/dataset.md) before drawing any conclusions from accuracy
numbers.

Each class is given a distinct, fixed random "signature" trajectory in
feature space plus per-sample noise, so a correctly implemented model
should be able to separate the classes (unlike pure noise, which validates
that the training loop can actually learn something) while still being
obviously synthetic.

Usage:
    python -m datasets.sample.generate_sample_dataset --dataset_name isl_demo --samples_per_class 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.config import settings


def generate(output_dir: Path, class_names: list[str], samples_per_class: int, seq_len_range: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    feature_dim = settings.feature_dim

    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # A fixed per-class "signature" curve so samples of the same class
        # are correlated, and different classes are separable.
        signature = rng.normal(0, 1, size=(feature_dim,))

        for i in range(samples_per_class):
            t = rng.integers(*seq_len_range)
            phase = np.linspace(0, rng.uniform(0.5, 1.5) * np.pi, t)[:, None]
            base = signature[None, :] * np.sin(phase)
            noise = rng.normal(0, 0.15, size=(t, feature_dim))
            sequence = (base + noise).astype(np.float32)
            np.save(class_dir / f"{class_name}_{i:03d}.npy", sequence)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic landmark dataset for pipeline testing.")
    parser.add_argument("--dataset_name", type=str, default="isl_demo")
    parser.add_argument("--samples_per_class", type=int, default=20)
    parser.add_argument("--min_frames", type=int, default=24)
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = settings.dataset_path / "processed" / args.dataset_name
    generate(output_dir, settings.vocabulary, args.samples_per_class, (args.min_frames, args.max_frames), args.seed)
    print(f"Synthetic dataset written to {output_dir} "
          f"({len(settings.vocabulary)} classes x {args.samples_per_class} samples). "
          "This is SYNTHETIC data for pipeline testing only — not real ISL data.")


if __name__ == "__main__":
    main()
