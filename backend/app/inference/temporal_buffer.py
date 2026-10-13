"""Rolling temporal buffer.

Holds the last ``window_size`` per-frame feature vectors so the model
always receives a real temporal sequence — the system never runs
inference on a single isolated frame. A configurable ``stride`` controls
how often (in frames) a new inference is actually triggered, decoupling
"how much history the model sees" from "how often we pay for a forward
pass".

``get_sequence()`` hands the buffered window to
``app.vision.preprocessing.resample_or_pad`` — the same temporal
preprocessing training uses — so an ``output_length`` shorter than
``window_size`` is filled by uniform sampling across the whole window,
exactly as training clips are.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

from app.vision.preprocessing import resample_or_pad


class TemporalBuffer:
    def __init__(self, window_size: int, feature_dim: int, stride: int = 1,
                 output_length: Optional[int] = None):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if output_length is not None and output_length <= 0:
            raise ValueError("output_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.stride = stride
        self.output_length = output_length or window_size  # frames handed to the model
        self._frames: deque[np.ndarray] = deque(maxlen=window_size)
        self._real_frame_flags: deque[bool] = deque(maxlen=window_size)
        self._frames_since_inference = 0

    def push(self, feature_vector: np.ndarray, is_real: bool = True) -> bool:
        """Add one frame's features. Returns True if this push completes a
        stride step and inference should now be triggered."""
        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {feature_vector.shape[0]}"
            )
        self._frames.append(feature_vector)
        self._real_frame_flags.append(is_real)
        self._frames_since_inference += 1

        if len(self._frames) < self.window_size:
            return False  # not enough history yet, regardless of stride
        if self._frames_since_inference >= self.stride:
            self._frames_since_inference = 0
            return True
        return False

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self.window_size

    @property
    def real_frame_count(self) -> int:
        return sum(self._real_frame_flags)

    def get_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (sequence, mask):
            sequence: (output_length, feature_dim) float32 — the buffered
                      window uniformly resampled to output_length frames, or
                      left-padded with zeros if fewer frames are buffered.
            mask:     (output_length,) bool, True where a real (non-padding)
                      frame exists.
        """
        if not self._frames:
            return (np.zeros((self.output_length, self.feature_dim), dtype=np.float32),
                    np.zeros((self.output_length,), dtype=bool))
        stacked = np.stack(list(self._frames), axis=0).astype(np.float32)
        flags = np.array(self._real_frame_flags, dtype=bool)
        return resample_or_pad(stacked, self.output_length, real_flags=flags)

    def reset(self) -> None:
        self._frames.clear()
        self._real_frame_flags.clear()
        self._frames_since_inference = 0
