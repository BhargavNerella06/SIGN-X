"""Landmark preprocessing: normalization, missing-landmark handling and
temporal padding/truncation.

These functions operate on the raw (x, y, z) landmark arrays produced by
``app.vision.landmarks`` and turn them into a stable, scale/translation
invariant representation suitable for learning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings
from app.vision.landmarks import FrameLandmarks, empty_face, empty_hand, empty_pose

_EPS = 1e-6


def normalize_group(points: Optional[np.ndarray], reference: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Translate a landmark group so its centroid (or a given reference
    point) is at the origin, then scale it so its max extent is 1.0.

    This makes the representation invariant to where in the frame the
    signer stands and how close they are to the camera — the model should
    learn the *shape* of the sign, not its absolute pixel position.
    """
    if points is None:
        return None
    ref = reference if reference is not None else points.mean(axis=0, keepdims=True)
    centered = points - ref
    scale = np.abs(centered).max()
    if scale < _EPS:
        return centered
    return centered / scale


def normalize_frame(landmarks: FrameLandmarks) -> FrameLandmarks:
    """Apply translation + scale normalization to every landmark group in a
    frame. Uses the pose's shoulder midpoint as a shared reference when
    available so hands/face stay positioned relative to the body; falls
    back to per-group centering otherwise."""
    if not settings.normalize_landmarks:
        return landmarks

    reference = None
    if landmarks.pose is not None and landmarks.pose.shape[0] > 12:
        # MediaPipe Pose: 11 = left shoulder, 12 = right shoulder
        reference = landmarks.pose[[11, 12]].mean(axis=0, keepdims=True)

    return FrameLandmarks(
        left_hand=normalize_group(landmarks.left_hand, reference),
        right_hand=normalize_group(landmarks.right_hand, reference),
        pose=normalize_group(landmarks.pose, reference),
        face=normalize_group(landmarks.face, reference),
    )


def fill_missing(landmarks: FrameLandmarks, n_face_points: int) -> FrameLandmarks:
    """Replace any missing (None) landmark group with zeros, so downstream
    code always sees fixed-shape arrays. Zero-filled groups act as a mask
    the model can learn to ignore (a real detected landmark is extremely
    unlikely to sit exactly at the origin after normalization)."""
    return FrameLandmarks(
        left_hand=landmarks.left_hand if landmarks.left_hand is not None else empty_hand(),
        right_hand=landmarks.right_hand if landmarks.right_hand is not None else empty_hand(),
        pose=landmarks.pose if landmarks.pose is not None else empty_pose(),
        face=landmarks.face if landmarks.face is not None else empty_face(n_face_points),
    )


def pad_or_truncate(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Force a (T, D) feature sequence to exactly ``target_length`` frames.

    - If shorter: pad at the start with zero-frames (so the most recent,
      most informative frames stay aligned to the end of the window).
    - If longer: keep only the most recent ``target_length`` frames.
    """
    t, d = sequence.shape
    if t == target_length:
        return sequence
    if t > target_length:
        return sequence[-target_length:]
    pad = np.zeros((target_length - t, d), dtype=sequence.dtype)
    return np.concatenate([pad, sequence], axis=0)


def length_based_mask(original_length: int, target_length: int) -> np.ndarray:
    """The correct companion to ``pad_or_truncate``: a (target_length,)
    boolean mask that is True for real frames and False for padding.

    Deliberately based on the ORIGINAL sequence length rather than
    re-scanning the padded array for all-zero rows — a real frame where
    MediaPipe simply detected nothing (poor lighting, hands briefly out of
    frame) is legitimately all-zero after fill_missing(), and must not be
    mistaken for a padding row. Mirrors pad_or_truncate's behavior exactly:
    padding is added at the start, and truncation keeps the trailing
    `target_length` frames (all of which are real).
    """
    if original_length >= target_length:
        return np.ones(target_length, dtype=bool)
    pad_amount = target_length - original_length
    return np.concatenate([np.zeros(pad_amount, dtype=bool), np.ones(original_length, dtype=bool)])


def uniform_sample_indices(original_length: int, target_length: int) -> np.ndarray:
    """``target_length`` frame indices spread evenly over the WHOLE of an
    ``original_length``-frame sequence (requires original_length >=
    target_length). Always includes the first and last frame, is strictly
    increasing (chronological, no repeats) and fully deterministic."""
    if original_length < target_length:
        raise ValueError("uniform_sample_indices needs original_length >= target_length")
    return np.round(np.linspace(0, original_length - 1, target_length)).astype(np.int64)


def resample_or_pad(sequence: np.ndarray, target_length: int,
                    real_flags: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Turn a (T, D) sequence into exactly ``target_length`` frames plus a
    matching (target_length,) boolean mask (True = real frame). This is the
    single temporal-preprocessing step shared by training
    (training/dataset.py) and real-time inference
    (app/inference/temporal_buffer.py).

    - T > target_length: uniformly sample target_length frames across the
      entire sequence, so the whole sign is represented (not just its end).
    - T == target_length: unchanged.
    - T < target_length: zero-pad at the start (pad_or_truncate), mask
      False on the padding (length_based_mask).

    ``real_flags`` optionally marks which input frames are real; it is
    sampled/padded the same way and ANDed into the mask.
    """
    t = sequence.shape[0]
    if t >= target_length:
        idx = uniform_sample_indices(t, target_length) if t > target_length else np.arange(t)
        out = sequence[idx] if t > target_length else sequence
        mask = np.ones(target_length, dtype=bool)
        if real_flags is not None:
            mask &= np.asarray(real_flags, dtype=bool)[idx]
        return out, mask
    out = pad_or_truncate(sequence, target_length)
    mask = length_based_mask(t, target_length)
    if real_flags is not None:
        mask[target_length - t:] &= np.asarray(real_flags, dtype=bool)
    return out, mask


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity (Δx, Δy, Δz per feature) for a
    (T, D) sequence. The first frame's velocity is zero (no prior frame)."""
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return velocity
