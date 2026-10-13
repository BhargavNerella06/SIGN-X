"""Adapter for the INCLUDE dataset (AI4Bharat / IIT Madras).

Source & licensing (verified, not invented — see docs/dataset.md):
    - Metadata (labels, official train/val/test split, video paths):
      https://huggingface.co/datasets/ai4bharat/INCLUDE  (CC-BY-4.0)
    - Video files (hosted separately, per-category zip archives):
      https://zenodo.org/records/4010759  (CC-BY-4.0)
    - Paper: Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian
      Sign Language Recognition", ACM Multimedia 2020.

The HuggingFace side only hosts *metadata* (parent_label, label,
video_path, include_50) as three parquet files (train/val/test) — the
actual videos must be downloaded from Zenodo separately (see
docs/dataset.md for the exact commands used). This adapter expects both
to already be present locally; it does not download anything itself.

Important, verified limitation: the INCLUDE metadata does **not** include
a signer identifier. Signer-independent splitting is therefore not
possible from this metadata alone, so this adapter uses the dataset's
own OFFICIAL train/val/test split as published by AI4Bharat rather than
re-shuffling samples itself (see docs/dataset.md "Data leakage" section).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.utils.logging import get_logger
from training.datasets.base import DatasetAdapter, DatasetInfo, DatasetSample

logger = get_logger(__name__)

_SPLIT_FILES = (("train.parquet", "train"), ("val.parquet", "validation"), ("test.parquet", "test"))
# Verified in the real HuggingFace metadata: 5250 rows but only 4257 unique
# video_paths, and hundreds of videos are listed in more than one split
# (e.g. 273 in both train and test). Each video is kept exactly once, in the
# highest-priority split it appears in, so no held-out video is trained on.
_SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}
# INCLUDE labels are usually "<number>. Name" (e.g. "48. Hello"), but a few
# are "Ex. Name" (verified in the real metadata, e.g. "Ex. Monsoon" under
# Seasons) — strip either prefix form.
_LABEL_PREFIX_RE = re.compile(r"^(?:\d+|Ex)\.\s*", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_label(raw_label: str) -> str:
    """Normalize an INCLUDE label to a training-ready class name.

    Examples: '48. Hello' -> 'HELLO', '46. you (plural)' -> 'YOU_PLURAL',
    '49. How are you' -> 'HOW_ARE_YOU'.
    """
    text = _LABEL_PREFIX_RE.sub("", raw_label).strip()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub("_", text.strip())
    return text.upper()


def _slugify_path(video_path: str) -> str:
    """Deterministic, filesystem-safe, unique id for a source video path.

    The readable slug alone is lossy — real INCLUDE files such as
    'MVI_0057.MOV' and 'MVI_0057_.MOV' both slugify to 'MVI_0057_MOV' — so a
    short hash of the exact original path is appended to keep ids unique.
    """
    readable = re.sub(r"[^A-Za-z0-9]+", "_", video_path).strip("_")
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}"


class IncludeAdapter(DatasetAdapter):
    """Reads INCLUDE's HuggingFace parquet metadata and resolves each row
    against locally downloaded Zenodo video files."""

    def __init__(
        self,
        video_root: Path,
        metadata_dir: Path,
        subset_classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        include_50_only: bool = False,
    ):
        self.video_root = Path(video_root)
        self.metadata_dir = Path(metadata_dir)
        self.subset_classes = {c.upper() for c in subset_classes} if subset_classes else None
        self.max_samples_per_class = max_samples_per_class
        self.include_50_only = include_50_only

    def _load_metadata_rows(self) -> List[dict]:
        import pyarrow.parquet as pq

        rows: List[dict] = []
        for filename, split_name in _SPLIT_FILES:
            path = self.metadata_dir / filename
            if not path.exists():
                hf_name = filename.replace(".parquet", "-00000-of-00001.parquet")
                raise FileNotFoundError(
                    f"INCLUDE metadata parquet not found: {path}\n"
                    "Download it first — see docs/dataset.md 'INCLUDE setup', e.g.:\n"
                    f"  curl -L -o {filename} "
                    f"https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/{hf_name}"
                )
            table = pq.read_table(path)
            for row in table.to_pylist():
                row["split"] = split_name
                rows.append(row)
        return self._deduplicate(rows)

    @staticmethod
    def _deduplicate(rows: List[dict]) -> List[dict]:
        """Keep one row per video_path, in its highest-priority split
        (test > validation > train), to prevent train/test leakage."""
        best: Dict[str, dict] = {}
        cross_split = 0
        for row in rows:
            prev = best.get(row["video_path"])
            if prev is None:
                best[row["video_path"]] = row
                continue
            if prev["split"] != row["split"]:
                cross_split += 1
            if _SPLIT_PRIORITY[row["split"]] < _SPLIT_PRIORITY[prev["split"]]:
                best[row["video_path"]] = row
        if len(best) != len(rows):
            logger.warning(
                "INCLUDE metadata lists %d rows but only %d unique videos (%d duplicate rows, "
                "%d spanning different splits) — kept each video once, preferring test > validation > train.",
                len(rows), len(best), len(rows) - len(best), cross_split,
            )
        return list(best.values())

    def list_samples(self) -> List[DatasetSample]:
        rows = self._load_metadata_rows()
        rows.sort(key=lambda r: r["video_path"])  # deterministic ordering, independent of parquet row order

        samples: List[DatasetSample] = []
        per_class_count: Dict[str, int] = {}
        missing_files = 0
        skipped_by_filter = 0

        for row in rows:
            if self.include_50_only and not row.get("include_50", False):
                continue

            label = clean_label(row["label"])
            if self.subset_classes is not None and label not in self.subset_classes:
                skipped_by_filter += 1
                continue

            if self.max_samples_per_class is not None and per_class_count.get(label, 0) >= self.max_samples_per_class:
                continue

            video_path = self.video_root / row["video_path"]
            if not video_path.exists():
                missing_files += 1
                continue

            samples.append(DatasetSample(
                sample_id=_slugify_path(row["video_path"]),
                video_path=video_path,
                label=label,
                split=row["split"],
                raw_label=row["label"],
                signer_id=None,  # not available in the published metadata — see module docstring
            ))
            per_class_count[label] = per_class_count.get(label, 0) + 1

        if missing_files:
            logger.warning(
                "%d INCLUDE videos referenced in metadata were not found under %s "
                "(not downloaded / not part of the categories fetched for this experiment).",
                missing_files, self.video_root,
            )
        logger.info(
            "INCLUDE adapter resolved %d samples across %d classes (%d filtered out by subset_classes, %d missing on disk).",
            len(samples), len(per_class_count), skipped_by_filter, missing_files,
        )
        return samples

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="INCLUDE",
            source="AI4Bharat / IIT Madras",
            license="CC-BY-4.0",
            citation=(
                "Sridhar, Ganesan, Kumar, Khapra. 'INCLUDE: A Large Scale Dataset "
                "for Indian Sign Language Recognition.' ACM Multimedia 2020. "
                "https://doi.org/10.1145/3394171.3413528"
            ),
            urls=[
                "https://huggingface.co/datasets/ai4bharat/INCLUDE",
                "https://zenodo.org/records/4010759",
            ],
            notes=(
                "Videos shot in Chennai, Tamil Nadu; not representative of all "
                "regional ISL variation. No signer identifier is published in "
                "the metadata, so signer-independent splitting could not be "
                "verified — the dataset's own official train/val/test split is "
                "used as-is instead of being re-shuffled."
            ),
        )
