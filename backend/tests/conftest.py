"""Shared pytest fixtures. No physical webcam is required for any test —
frames are synthetic numpy arrays and MediaPipe runs on those directly."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def random_frame() -> np.ndarray:
    """A synthetic BGR frame, standing in for a webcam capture."""
    rng = np.random.default_rng(0)
    return (rng.random((480, 640, 3)) * 255).astype(np.uint8)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)
