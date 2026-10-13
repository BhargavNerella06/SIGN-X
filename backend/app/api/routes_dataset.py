"""Dataset information endpoint.

Reports on whatever has actually been prepared under
``settings.dataset_path`` by ``training/prepare_dataset.py`` — it never
fabricates class/sample counts. If nothing has been prepared yet, it says
so explicitly and points at docs/dataset.md.

Expected prepared-dataset layout (created by training/prepare_dataset.py):

    <dataset_path>/<dataset_name>/
        class_mapping.json      {"classes": ["HELLO", "THANK_YOU", ...]}
        train/<class>/*.npy
        validation/<class>/*.npy
        test/<class>/*.npy
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

_SPLITS = ("train", "validation", "test")


def _count_samples(split_dir: Path) -> dict:
    if not split_dir.exists():
        return {"total": 0, "per_class": {}}
    per_class = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        per_class[class_dir.name] = len(list(class_dir.glob("*.npy")))
    return {"total": sum(per_class.values()), "per_class": per_class}


def _scan_dataset(dataset_dir: Path) -> dict:
    mapping_path = dataset_dir / "class_mapping.json"
    classes = []
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", data if isinstance(data, list) else [])

    splits = {split: _count_samples(dataset_dir / split) for split in _SPLITS}
    total_samples = sum(s["total"] for s in splits.values())

    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "num_classes": len(classes) or len({c for s in splits.values() for c in s["per_class"]}),
        "classes": classes,
        "splits": {k: v["total"] for k, v in splits.items()},
        "per_class_counts": splits,
        "total_samples": total_samples,
        "is_prepared": total_samples > 0,
    }


@router.get("/info")
def dataset_info() -> dict:
    root = settings.dataset_path
    if not root.exists():
        return {
            "status": "not_found",
            "message": f"Dataset path {root} does not exist.",
            "configured_vocabulary": settings.vocabulary,
            "see": "docs/dataset.md",
        }

    prepared = [
        _scan_dataset(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name not in ("sample", "raw") and (d / "class_mapping.json").exists()
    ]

    if not prepared:
        return {
            "status": "not_prepared",
            "message": (
                "No prepared dataset found. Run training/extract_landmarks.py "
                "and training/prepare_dataset.py after placing a dataset under "
                f"{root}. See docs/dataset.md for recommended ISL datasets."
            ),
            "configured_vocabulary": settings.vocabulary,
            "see": "docs/dataset.md",
        }

    return {"status": "ok", "datasets": prepared}
