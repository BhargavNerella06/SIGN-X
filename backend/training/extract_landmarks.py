"""Extract landmark feature sequences from raw ISL videos.

Expected input layout (place your downloaded dataset here — see
docs/dataset.md for where to obtain one; nothing is downloaded
automatically):

    backend/datasets/raw/<CLASS_NAME>/*.mp4   (also accepts .avi, .mov, .mkv)

Output (one .npy per video, shape (T, feature_dim), using the exact same
``FeatureExtractor`` as real-time inference so training and serving never
drift apart):

    backend/datasets/processed/<dataset_name>/<CLASS_NAME>/<video_stem>.npy

Usage:
    python -m training.extract_landmarks --dataset_name isl_demo
    python -m training.extract_landmarks --raw_dir datasets/raw --dataset_name isl_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.utils.logging import get_logger
from app.vision.feature_extractor import FeatureExtractor

logger = get_logger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_video(extractor: FeatureExtractor, video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return None

    extractor.reset()  # each video is an independent sequence; don't leak velocity state across clips
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vector, _ = extractor.extract(frame)
            frames.append(vector)
    finally:
        cap.release()

    if not frames:
        logger.warning("No frames read from %s", video_path)
        return None
    return np.stack(frames, axis=0)


def extract_video_fresh(video_path: Path, extractor_factory=FeatureExtractor) -> np.ndarray | None:
    """Extract one video with its OWN freshly created extractor, closed afterwards.

    MediaPipe's HolisticLandmarker keeps internal state between calls even in
    IMAGE running mode (verified: running the same video twice through one
    instance gives different hand landmarks, while two fresh instances give
    identical output). Reusing one instance across a dataset would make each
    video's features depend on the videos processed before it, so every
    source video gets a fresh instance: create -> process -> close.
    """
    extractor = extractor_factory()
    try:
        if not extractor.available:
            raise RuntimeError(
                "MediaPipe model unavailable — download it first "
                "(scripts/download_models.ps1 / download_models.sh)."
            )
        return extract_video(extractor, video_path)
    finally:
        extractor.close()


def run(raw_dir: Path, output_root: Path, dataset_name: str) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_dir}\n"
            "Place your ISL dataset there first — see docs/dataset.md for "
            "recommended sources. Expected layout: "
            f"{raw_dir}/<CLASS_NAME>/<video files>"
        )

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found under {raw_dir}")

    output_dir = output_root / dataset_name
    total_ok, total_failed = 0, 0

    for class_dir in class_dirs:
        videos = [p for p in class_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS]
        if not videos:
            logger.warning("No video files found for class %s", class_dir.name)
            continue

        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos:
            sequence = extract_video_fresh(video_path)
            if sequence is None:
                total_failed += 1
                continue
            out_path = out_class_dir / f"{video_path.stem}.npy"
            np.save(out_path, sequence)
            total_ok += 1
            logger.info("Extracted %s -> %s (%d frames)", video_path.name, out_path, sequence.shape[0])

    logger.info(
        "Landmark extraction complete: %d succeeded, %d failed. Output: %s",
        total_ok, total_failed, output_dir,
    )
    if total_ok == 0:
        raise RuntimeError("No videos were successfully processed — check the raw dataset layout.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmark sequences from raw ISL videos.")
    parser.add_argument("--raw_dir", type=str, default=str(settings.dataset_path / "raw"))
    parser.add_argument("--output_dir", type=str, default=str(settings.dataset_path / "processed"))
    parser.add_argument("--dataset_name", type=str, default="isl_demo")
    args = parser.parse_args()

    run(Path(args.raw_dir), Path(args.output_dir), args.dataset_name)


if __name__ == "__main__":
    main()
