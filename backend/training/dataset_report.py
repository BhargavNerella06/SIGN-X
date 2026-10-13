"""Data quality report for a real-dataset adapter's resolved samples.

Runs directly against the raw videos (via the adapter — no pre-extracted
landmarks required), using the SAME ``LandmarkExtractor`` real-time
inference uses, so the reported detection rates reflect exactly what the
production pipeline will see.

Usage:
    python -m training.dataset_report --config experiments/isl_include_subset28.yaml

    python -m training.dataset_report --config experiments/isl_include_subset28.yaml --from_landmarks

Writes <dataset_path>/<name>/dataset_report.json and prints a terminal summary.
``--from_landmarks`` computes the report from the landmark sequences written
by ``training.build_dataset_from_adapter`` (streamed one file at a time)
instead of running MediaPipe again, and adds per-group feature statistics.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from app.config import BACKEND_DIR
from app.utils.logging import get_logger
from app.vision.landmarks import LandmarkExtractor
from training.datasets.registry import get_adapter_class

logger = get_logger(__name__)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BACKEND_DIR / path


def analyze_video(extractor: LandmarkExtractor, video_path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = 0
    both_hands_missing = 0
    pose_missing = 0
    any_detected = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            landmarks = extractor.process(frame)
            if not landmarks.has_any_hand:
                both_hands_missing += 1
            if landmarks.pose is None:
                pose_missing += 1
            if landmarks.has_any_hand or landmarks.pose is not None:
                any_detected += 1
    finally:
        cap.release()

    if frame_count == 0:
        return None

    return {
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps > 0 else None,
        "both_hands_missing_frames": both_hands_missing,
        "pose_missing_frames": pose_missing,
        "any_detected_frames": any_detected,
    }


# Feature-vector layout (see app/config.py::Settings.feature_dim): hands, pose, then face.
def _feature_groups() -> dict[str, slice]:
    from app.config import settings

    groups = {"left_hand": slice(0, 63), "right_hand": slice(63, 126)}
    end = 126
    if settings.use_pose:
        groups["pose"] = slice(end, end + 99)
        end += 99
    if settings.use_face:
        groups["face"] = slice(end, end + 3 * len(settings.face_landmark_indices))
    return groups


class _LandmarkStats:
    """Streaming per-group feature statistics over saved landmark sequences
    (one file at a time, so memory use is independent of dataset size)."""

    def __init__(self):
        import numpy as np

        self._np = np
        self.groups = _feature_groups()
        self.frames = 0
        self.detected = {g: 0 for g in self.groups}
        self.all_zero_frames = 0
        self.nonfinite_values = 0
        self.count = {g: 0 for g in self.groups}
        self.total = {g: 0.0 for g in self.groups}
        self.total_sq = {g: 0.0 for g in self.groups}
        self.min = {g: float("inf") for g in self.groups}
        self.max = {g: float("-inf") for g in self.groups}

    def add(self, seq) -> dict:
        np = self._np
        self.frames += seq.shape[0]
        self.nonfinite_values += int((~np.isfinite(seq)).sum())
        self.all_zero_frames += int((np.abs(seq).sum(axis=1) == 0).sum())
        detected = {}
        for g, s in self.groups.items():
            part = seq[:, s].astype(np.float64)
            present = np.abs(part).sum(axis=1) > 0  # missing groups are zero-filled (fill_missing)
            detected[g] = int(present.sum())
            self.detected[g] += detected[g]
            vals = part[present]
            if vals.size:
                self.count[g] += vals.size
                self.total[g] += float(vals.sum())
                self.total_sq[g] += float((vals ** 2).sum())
                self.min[g] = min(self.min[g], float(vals.min()))
                self.max[g] = max(self.max[g], float(vals.max()))
        return detected

    def summary(self) -> dict:
        out = {"total_frames": self.frames, "all_zero_frames": self.all_zero_frames,
               "nan_or_inf_values": self.nonfinite_values, "groups": {}}
        for g in self.groups:
            n = self.count[g]
            mean = self.total[g] / n if n else None
            std = (max(self.total_sq[g] / n - mean ** 2, 0.0)) ** 0.5 if n else None
            out["groups"][g] = {
                "detected_frame_rate": round(self.detected[g] / self.frames, 4) if self.frames else None,
                "mean": round(mean, 4) if n else None,
                "std": round(std, 4) if n else None,
                "min": round(self.min[g], 4) if n else None,
                "max": round(self.max[g], 4) if n else None,
            }
        return out


def run_from_landmarks(config_path: Path) -> dict:
    """Quality report computed from already-extracted landmark sequences
    (training.build_dataset_from_adapter output) instead of re-running
    MediaPipe on every video. Streams one .npy at a time."""
    import json

    import numpy as np

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg["dataset"]
    sequence_length = ds_cfg["sequence_length"]
    adapter = get_adapter_class(ds_cfg["adapter"])(
        video_root=_resolve(ds_cfg["video_root"]),
        metadata_dir=_resolve(ds_cfg["metadata_dir"]),
        subset_classes=ds_cfg.get("subset_classes"),
        max_samples_per_class=ds_cfg.get("max_samples_per_class"),
        include_50_only=ds_cfg.get("include_50_only", False),
    )
    samples = adapter.list_samples()
    output_dir = BACKEND_DIR / ds_cfg["dataset_path"] / ds_cfg["name"]

    stats = _LandmarkStats()
    per_class: Counter = Counter()
    per_split_class: dict[str, Counter] = defaultdict(Counter)
    missing, corrupt, frame_counts, durations = [], [], [], []
    low_hand_samples = []
    truncated = padded = 0

    for i, sample in enumerate(samples, 1):
        path = output_dir / sample.split / sample.label / f"{sample.sample_id}.npy"
        if not path.exists():
            missing.append(sample.sample_id)
            continue
        try:
            seq = np.load(path)
            if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] != stats.groups[list(stats.groups)[-1]].stop:
                raise ValueError(f"unexpected shape {seq.shape}")
        except Exception as exc:  # noqa: BLE001 — any unreadable file counts as corrupt
            corrupt.append({"sample_id": sample.sample_id, "error": str(exc)})
            continue

        detected = stats.add(seq)
        n = seq.shape[0]
        frame_counts.append(float(n))
        truncated += n > sequence_length
        padded += n < sequence_length
        cap = cv2.VideoCapture(str(sample.video_path))  # header only, no decoding
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        if fps > 0:
            durations.append(n / fps)
        hand_rate = max(detected["left_hand"], detected["right_hand"]) / n
        if hand_rate < 0.5:
            low_hand_samples.append({"sample_id": sample.sample_id, "label": sample.label,
                                     "split": sample.split, "best_hand_detected_rate": round(hand_rate, 3)})
        per_class[sample.label] += 1
        per_split_class[sample.split][sample.label] += 1
        if i % 100 == 0:
            logger.info("Analyzed %d/%d landmark files...", i, len(samples))

    lengths = np.array(frame_counts) if frame_counts else np.array([0.0])
    report = {
        "mode": "from_landmarks",
        "adapter": ds_cfg["adapter"],
        "dataset_name": ds_cfg["name"],
        "total_videos": len(samples),
        "valid_videos": len(frame_counts),
        "missing_landmark_files": len(missing),
        "missing_sample_ids": missing,
        "corrupt_landmark_files": len(corrupt),
        "corrupt_samples": corrupt,
        "num_classes": len(per_class),
        "classes": sorted(per_class),
        "samples_per_class": dict(sorted(per_class.items())),
        "samples_per_split": {s: sum(c.values()) for s, c in per_split_class.items()},
        "samples_per_split_per_class": {s: dict(sorted(c.items())) for s, c in per_split_class.items()},
        "duration_seconds": _summary_stats(durations),
        "sequence_length_frames": {
            **_summary_stats(frame_counts),
            "p10": float(np.percentile(lengths, 10)), "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "model_sequence_length": sequence_length,
            "samples_longer_than_model_window": int(truncated),
            "samples_shorter_than_model_window": int(padded),
        },
        "feature_statistics": stats.summary(),
        "samples_with_hands_detected_in_under_half_of_frames": low_hand_samples,
    }
    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Landmark-based quality report written to %s", report_path)
    return report


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "stdev": None}
    return {
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


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

    probe = LandmarkExtractor()  # fail fast; each video gets its own instance below
    available = probe.available
    probe.close()
    if not available:
        raise RuntimeError("MediaPipe model unavailable — run scripts/download_models.ps1 first.")

    per_class_counts: Counter = Counter(s.label for s in samples)
    per_split_counts: Counter = Counter(s.split for s in samples)

    valid, failed = 0, []
    durations, frame_counts = [], []
    total_frames = 0
    total_both_hands_missing = 0
    total_pose_missing = 0
    total_any_detected = 0

    for i, sample in enumerate(samples):
        extractor = LandmarkExtractor()  # fresh per video: HolisticLandmarker keeps state across calls
        try:
            stats = analyze_video(extractor, sample.video_path)
        finally:
            extractor.close()
        if stats is None:
            failed.append(sample.sample_id)
            continue
        valid += 1
        frame_counts.append(stats["frame_count"])
        if stats["duration_s"] is not None:
            durations.append(stats["duration_s"])
        total_frames += stats["frame_count"]
        total_both_hands_missing += stats["both_hands_missing_frames"]
        total_pose_missing += stats["pose_missing_frames"]
        total_any_detected += stats["any_detected_frames"]

        if (i + 1) % 50 == 0:
            logger.info("Analyzed %d/%d videos...", i + 1, len(samples))

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "min": None, "max": None, "stdev": None}
        return {
            "mean": round(statistics.mean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        }

    report = {
        "adapter": ds_cfg["adapter"],
        "dataset_name": ds_cfg["name"],
        "total_videos": len(samples),
        "valid_videos": valid,
        "failed_videos": len(failed),
        "failed_video_ids": failed,
        "num_classes": len(per_class_counts),
        "classes": sorted(per_class_counts),
        "samples_per_class": dict(sorted(per_class_counts.items())),
        "samples_per_split": dict(per_split_counts),
        "duration_seconds": _stats(durations),
        "sequence_length_frames": _stats([float(x) for x in frame_counts]),
        "landmark_detection": {
            "total_frames_analyzed": total_frames,
            "any_group_detected_rate": round(total_any_detected / total_frames, 4) if total_frames else None,
            "both_hands_missing_rate": round(total_both_hands_missing / total_frames, 4) if total_frames else None,
            "pose_missing_rate": round(total_pose_missing / total_frames, 4) if total_frames else None,
        },
    }

    output_dir = BACKEND_DIR / ds_cfg["dataset_path"] / ds_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.json"
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, report_path)
    return report


def _print_summary(report: dict, report_path: Path) -> None:
    print("\n=== SIGN-X Dataset Quality Report ===")
    print(f"Adapter: {report['adapter']}    Dataset: {report['dataset_name']}")
    print(f"Total videos: {report['total_videos']}  |  Valid: {report['valid_videos']}  |  Failed: {report['failed_videos']}")
    print(f"Classes: {report['num_classes']}")
    print(f"Samples per split: {report['samples_per_split']}")
    d = report["duration_seconds"]
    print(f"Duration (s): mean={d['mean']} min={d['min']} max={d['max']} stdev={d['stdev']}")
    s = report["sequence_length_frames"]
    print(f"Sequence length (frames): mean={s['mean']} min={s['min']} max={s['max']} stdev={s['stdev']}")
    ld = report["landmark_detection"]
    print(f"Landmark detection — any group detected: {ld['any_group_detected_rate']}")
    print(f"Landmark detection — both-hands-missing rate: {ld['both_hands_missing_rate']}")
    print(f"Landmark detection — pose-missing rate: {ld['pose_missing_rate']}")
    if report["failed_videos"]:
        print(f"WARNING: {report['failed_videos']} videos failed to open/read — see failed_video_ids in {report_path}")
    print(f"\nFull report: {report_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a data quality report for a real-dataset adapter.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--from_landmarks", action="store_true",
        help="Compute the report from already-extracted landmark .npy files instead of re-running "
             "MediaPipe on every video (much faster, constant memory).",
    )
    args = parser.parse_args()
    if args.from_landmarks:
        run_from_landmarks(Path(args.config))
    else:
        run(Path(args.config))


if __name__ == "__main__":
    main()
