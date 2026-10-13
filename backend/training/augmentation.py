"""Optional data augmentation for landmark sequences (Section 22).

Operates on already-normalized ``(T, D)`` feature sequences, where ``D`` is
always a multiple of 3 (each point contributes x, y, z). Every transform is
small/realistic — these simulate natural variation (slightly different
signer position, camera angle, timing) rather than arbitrary distortions
that would never occur in real landmark data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    enabled: bool = True
    gaussian_noise_std: float = 0.01
    rotation_degrees: float = 5.0
    scale_range: Tuple[float, float] = (0.95, 1.05)
    translation_std: float = 0.01
    temporal_crop_ratio: float = 0.1
    frame_drop_prob: float = 0.05


def _as_points(sequence: np.ndarray) -> np.ndarray:
    t, d = sequence.shape
    assert d % 3 == 0, f"Feature dim {d} is not a multiple of 3 (x, y, z per point)"
    return sequence.reshape(t, d // 3, 3)


def add_gaussian_noise(sequence: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return sequence
    return sequence + rng.normal(0, std, size=sequence.shape).astype(sequence.dtype)


def random_rotation(sequence: np.ndarray, max_degrees: float, rng: np.random.Generator) -> np.ndarray:
    if max_degrees <= 0:
        return sequence
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = _as_points(sequence).copy()
    x, y = points[..., 0].copy(), points[..., 1].copy()
    points[..., 0] = x * cos_a - y * sin_a
    points[..., 1] = x * sin_a + y * cos_a
    return points.reshape(sequence.shape)


def random_scale(sequence: np.ndarray, scale_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(*scale_range)
    return sequence * factor


def random_translation(sequence: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return sequence
    points = _as_points(sequence).copy()
    shift = rng.normal(0, std, size=(1, 1, 3)).astype(sequence.dtype)
    points[..., :2] += shift[..., :2]  # only shift x, y — keep depth (z) untouched
    return points.reshape(sequence.shape)


def temporal_crop(sequence: np.ndarray, max_ratio: float, rng: np.random.Generator) -> np.ndarray:
    t = sequence.shape[0]
    max_crop = int(t * max_ratio)
    if max_crop <= 0 or t <= 2:
        return sequence
    start = rng.integers(0, max_crop + 1)
    end = t - rng.integers(0, max_crop + 1)
    if end <= start:
        return sequence
    return sequence[start:end]


def frame_dropping(sequence: np.ndarray, drop_prob: float, rng: np.random.Generator) -> np.ndarray:
    if drop_prob <= 0:
        return sequence
    keep_mask = rng.random(sequence.shape[0]) >= drop_prob
    if not keep_mask.any():
        return sequence
    return sequence[keep_mask]


def build_augmentor(config: AugmentationConfig, seed: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a callable that applies the configured augmentations, in a
    realistic order: temporal effects first, then spatial jitter."""
    rng = np.random.default_rng(seed)

    def augment(sequence: np.ndarray) -> np.ndarray:
        if not config.enabled:
            return sequence
        seq = temporal_crop(sequence, config.temporal_crop_ratio, rng)
        seq = frame_dropping(seq, config.frame_drop_prob, rng)
        seq = random_rotation(seq, config.rotation_degrees, rng)
        seq = random_scale(seq, config.scale_range, rng)
        seq = random_translation(seq, config.translation_std, rng)
        seq = add_gaussian_noise(seq, config.gaussian_noise_std, rng)
        return seq

    return augment
