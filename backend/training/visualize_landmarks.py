"""Visual validation: draw MediaPipe landmarks on real dataset frames so a
human can confirm hands/pose/face are actually being detected before
spending time on a full training run (Section 7 — "do not proceed blindly
if landmark extraction is failing").

Usage:
    python -m training.visualize_landmarks --config experiments/isl_include_subset28.yaml --num_videos 6

Writes side-by-side (original | annotated) PNGs to
training/logs/landmark_validation/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.config import BACKEND_DIR
from app.utils.logging import get_logger
from app.vision.landmarks import FrameLandmarks, LandmarkExtractor
from training.datasets.registry import get_adapter_class

logger = get_logger(__name__)

_COLORS = {
    "left_hand": (244, 99, 55),    # BGR — matches the frontend's blue-ish hand color, roughly
    "right_hand": (34, 197, 94),
    "pose": (245, 158, 11),
    "face": (255, 255, 255),
}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BACKEND_DIR / path


def draw_landmarks(frame: np.ndarray, landmarks: FrameLandmarks) -> np.ndarray:
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for name, group in (
        ("pose", landmarks.pose),
        ("left_hand", landmarks.left_hand),
        ("right_hand", landmarks.right_hand),
        ("face", landmarks.face),
    ):
        if group is None:
            continue
        color = _COLORS[name]
        for x, y, _ in group:
            cv2.circle(annotated, (int(x * w), int(y * h)), 3, color, -1)
    return annotated


def middle_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def run(config_path: Path, num_videos: int, seed: int) -> Path:
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
        raise RuntimeError("No samples resolved — check dataset paths in the config.")

    rng = random.Random(seed)
    chosen = rng.sample(samples, k=min(num_videos, len(samples)))

    output_dir = BACKEND_DIR / "training" / "logs" / "landmark_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor()
    if not extractor.available:
        raise RuntimeError("MediaPipe model unavailable — run scripts/download_models.ps1 first.")

    saved = []
    for sample in chosen:
        frame = middle_frame(sample.video_path)
        if frame is None:
            logger.warning("Could not read a frame from %s", sample.video_path)
            continue
        landmarks = extractor.process(frame)
        annotated = draw_landmarks(frame, landmarks)
        side_by_side = np.hstack([frame, annotated])

        out_path = output_dir / f"{sample.label}__{sample.sample_id}.png"
        cv2.imwrite(str(out_path), side_by_side)
        saved.append(out_path)
        logger.info(
            "%s: hands=%s pose=%s -> %s",
            sample.label, landmarks.has_any_hand, landmarks.pose is not None, out_path,
        )

    extractor.close()
    logger.info("Saved %d annotated validation frames to %s", len(saved), output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visually validate landmark extraction on sample videos.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(Path(args.config), args.num_videos, args.seed)


if __name__ == "__main__":
    main()
