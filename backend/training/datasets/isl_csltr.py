"""Adapter for the ISL-CSLTR dataset (sentence-level, continuous ISL).

Source & licensing (verified, not invented — see docs/dataset.md):
    https://data.mendeley.com/datasets/kcmpdxky7p/1  (CC-BY-4.0)
    "ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language
    Translation and Recognition" — single ~8.9 GB zip
    (`ISL_CSLRT_Corpus.zip`), 700 annotated videos, 100 spoken-language
    sentences, 7 signers (per the dataset's public Mendeley description).

**Status: NOT exercised in the Stage 2 experiment.** SIGN-X's current
training pipeline targets *isolated*-sign classification (see
docs/architecture.md "Continuous recognition (Stage 7+)"); this dataset is
sentence-level/continuous. Downloading and integrating it (~8.9 GB, and a
label/annotation format that has not been inspected against a real
extracted copy in this repository) is out of scope for the isolated-sign
first-target experiment and was not performed here — doing so without
having verified the actual on-disk layout would risk exactly the kind of
fabricated-structure code this project explicitly avoids.

This adapter is provided as a real starting point for that future work,
not as a fully verified implementation. Once the corpus is downloaded and
extracted, inspect its actual folder/annotation layout and adjust
``_ANNOTATION_GLOB`` / ``_parse_annotation_row`` below to match — the
class docstring above tells you exactly what was and wasn't confirmed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from app.utils.logging import get_logger
from training.datasets.base import DatasetAdapter, DatasetInfo, DatasetSample

logger = get_logger(__name__)

# Best-effort guess at an annotation manifest name based on datasets of this
# shape (a CSV mapping video/frame files to sentence text). NOT verified
# against the actual extracted corpus — adjust after inspecting it.
_ANNOTATION_GLOB = "**/*.csv"


class IslCsltrAdapter(DatasetAdapter):
    """Adapter skeleton for ISL-CSLTR. Requires the corpus to already be
    downloaded and extracted under ``corpus_root`` (see docs/dataset.md).

    Because SIGN-X is not yet doing continuous/sentence-level training,
    ``list_samples()`` here treats each annotated sentence video as one
    "sample" labeled with its full sentence text — usable for future
    continuous-recognition work, but the label space is sentence-level
    (100 classes of one sample each on average), not the isolated-sign
    vocabulary the rest of Stage 2 trains on.
    """

    def __init__(self, corpus_root: Path, subset_classes: Optional[List[str]] = None):
        self.corpus_root = Path(corpus_root)
        self.subset_classes = {c.upper() for c in subset_classes} if subset_classes else None

    def list_samples(self) -> List[DatasetSample]:
        if not self.corpus_root.exists():
            raise FileNotFoundError(
                f"ISL-CSLTR corpus not found at {self.corpus_root}. Download "
                "ISL_CSLRT_Corpus.zip from "
                "https://data.mendeley.com/datasets/kcmpdxky7p/1 and extract "
                "it there first — see docs/dataset.md."
            )

        annotation_files = list(self.corpus_root.glob(_ANNOTATION_GLOB))
        if not annotation_files:
            raise RuntimeError(
                f"No annotation CSV found under {self.corpus_root} matching "
                f"'{_ANNOTATION_GLOB}'. The corpus's actual annotation format "
                "has not been verified in this codebase yet — inspect the "
                "extracted corpus and update training/datasets/isl_csltr.py "
                "accordingly instead of assuming this glob is correct."
            )

        logger.warning(
            "IslCsltrAdapter is an unverified skeleton (see module docstring) "
            "— found %d candidate annotation file(s) but sample parsing is "
            "not implemented against a confirmed real layout.",
            len(annotation_files),
        )
        raise NotImplementedError(
            "ISL-CSLTR sample parsing requires inspecting the real extracted "
            "corpus structure first (see this file's module docstring). Not "
            "implemented in Stage 2, which targets isolated-sign recognition "
            "via the INCLUDE adapter instead."
        )

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="ISL-CSLTR",
            source="Mendeley Data (contributors: Navajeevan Residential School for "
                   "the Deaf; SASTRA Deemed University)",
            license="CC-BY-4.0",
            urls=["https://data.mendeley.com/datasets/kcmpdxky7p/1"],
            notes=(
                "Sentence-level continuous ISL corpus: 700 annotated videos, "
                "100 spoken-language sentences, 7 signers (per the dataset's "
                "public description). ~8.9 GB single archive. Not downloaded "
                "or integrated in the current isolated-sign experiment — see "
                "this module's docstring."
            ),
        )
