"""Common interface every real-dataset adapter implements.

An adapter's only job is to translate a specific source dataset's own
directory/metadata layout into a flat list of ``DatasetSample`` records.
Nothing downstream (landmark extraction, dataset_report, training) needs to
know anything about the original dataset's structure — it only ever sees
this common representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DatasetSample:
    """One video sample in the common internal representation."""

    sample_id: str        # stable, filesystem-safe unique id
    video_path: Path       # absolute path to the actual video file on disk
    label: str             # cleaned, training-ready class label (e.g. "HELLO")
    split: str              # "train" | "validation" | "test"
    raw_label: str = ""     # original dataset label, kept for traceability
    signer_id: Optional[str] = None  # None when the source dataset doesn't expose it


@dataclass
class DatasetInfo:
    """Human-readable provenance/licensing metadata surfaced in reports and docs."""

    name: str
    source: str
    license: str
    citation: str = ""
    urls: List[str] = field(default_factory=list)
    notes: str = ""


class DatasetAdapter:
    """Base class for a real-dataset adapter.

    Subclasses implement ``list_samples()`` and ``info()``. They must NOT
    invent files that don't exist on disk — a sample whose video file is
    missing (e.g. not yet downloaded) should be skipped and counted, not
    fabricated.
    """

    def list_samples(self) -> List[DatasetSample]:
        raise NotImplementedError

    def info(self) -> DatasetInfo:
        raise NotImplementedError
