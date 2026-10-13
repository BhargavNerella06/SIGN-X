"""Maps a dataset name (as used in experiment config files) to its adapter."""

from __future__ import annotations

from typing import Dict, Type

from training.datasets.base import DatasetAdapter
from training.datasets.include import IncludeAdapter
from training.datasets.isl_csltr import IslCsltrAdapter

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {
    "include": IncludeAdapter,
    "isl_csltr": IslCsltrAdapter,
}


def get_adapter_class(name: str) -> Type[DatasetAdapter]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset adapter '{name}'. Available: {sorted(_REGISTRY)}"
        ) from None


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)
