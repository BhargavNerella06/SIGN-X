"""Tests for app.vision.preprocessing."""

from __future__ import annotations

import numpy as np

from app.vision.landmarks import FrameLandmarks
from app.vision.preprocessing import (
    compute_velocity,
    fill_missing,
    length_based_mask,
    normalize_group,
    pad_or_truncate,
    resample_or_pad,
    uniform_sample_indices,
)


def test_normalize_group_centers_and_scales():
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
    normalized = normalize_group(points)
    # Centering by the centroid is a mean-preserving linear op, so the
    # normalized cloud should still average to (0, 0, 0).
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-5)
    # And scaling by the max absolute extent bounds every coordinate to [-1, 1].
    assert np.abs(normalized).max() <= 1.0 + 1e-5


def test_normalize_group_handles_degenerate_zero_extent():
    points = np.zeros((5, 3), dtype=np.float32)
    normalized = normalize_group(points)
    assert np.allclose(normalized, 0)


def test_fill_missing_replaces_none_with_zeros():
    landmarks = FrameLandmarks(left_hand=None, right_hand=None, pose=None, face=None)
    filled = fill_missing(landmarks, n_face_points=10)
    assert filled.left_hand.shape == (21, 3)
    assert filled.right_hand.shape == (21, 3)
    assert filled.pose.shape == (33, 3)
    assert filled.face.shape == (10, 3)
    assert np.all(filled.left_hand == 0)


def test_pad_or_truncate_pads_short_sequences():
    seq = np.ones((5, 4), dtype=np.float32)
    result = pad_or_truncate(seq, 10)
    assert result.shape == (10, 4)
    assert np.all(result[:5] == 0)  # left-padded
    assert np.all(result[5:] == 1)


def test_pad_or_truncate_truncates_long_sequences_keeping_most_recent():
    seq = np.arange(20).reshape(10, 2).astype(np.float32)
    result = pad_or_truncate(seq, 4)
    assert result.shape == (4, 2)
    assert np.array_equal(result, seq[-4:])


def test_pad_or_truncate_noop_when_exact_length():
    seq = np.ones((6, 3), dtype=np.float32)
    result = pad_or_truncate(seq, 6)
    assert result is seq


def test_compute_velocity_first_frame_zero_and_matches_diff():
    seq = np.array([[0, 0], [1, 1], [3, 3]], dtype=np.float32)
    velocity = compute_velocity(seq)
    assert np.allclose(velocity[0], 0)
    assert np.allclose(velocity[1], [1, 1])
    assert np.allclose(velocity[2], [2, 2])


def test_length_based_mask_pads_correctly_for_short_sequences():
    mask = length_based_mask(original_length=10, target_length=16)
    assert mask.shape == (16,)
    assert mask.sum() == 10
    assert np.array_equal(mask, [False] * 6 + [True] * 10)


def test_length_based_mask_all_true_when_truncated():
    # pad_or_truncate keeps the trailing target_length frames when the
    # original sequence is longer — every kept frame is a real frame.
    mask = length_based_mask(original_length=40, target_length=16)
    assert mask.shape == (16,)
    assert mask.all()


def test_length_based_mask_all_true_when_exact_length():
    mask = length_based_mask(original_length=16, target_length=16)
    assert mask.all()


def test_length_based_mask_does_not_mistake_real_zero_frames_for_padding():
    # A real frame where nothing was detected is legitimately all-zero
    # (see app.vision.preprocessing.fill_missing) — the mask must still
    # mark it as real, unlike a naive "any(sequence != 0)" check would.
    all_zero_but_real = np.zeros((10, 6), dtype=np.float32)
    padded = pad_or_truncate(all_zero_but_real, 16)
    mask = length_based_mask(original_length=10, target_length=16)
    assert mask.sum() == 10
    assert padded.shape == (16, 6)


# --- resample_or_pad: shared temporal preprocessing (training + realtime) ---

def _frame_ids(n: int, d: int = 3) -> np.ndarray:
    """(n, d) sequence whose every value in row i is i + 1 (so 0 means padding)."""
    return np.repeat(np.arange(1, n + 1, dtype=np.float32)[:, None], d, axis=1)


def test_resample_long_sequence_covers_whole_clip():
    seq = _frame_ids(100)
    out, mask = resample_or_pad(seq, 32)
    assert out.shape == (32, 3)
    assert mask.shape == (32,) and mask.all()
    assert out[0, 0] == 1 and out[-1, 0] == 100  # first AND last frame kept
    assert out[:16, 0].max() <= 50 < out[16:, 0].min()  # first half of output = first half of clip


def test_resample_long_sequence_is_chronological_without_repeats():
    for n in (33, 45, 60, 64, 109, 1000):
        out, _ = resample_or_pad(_frame_ids(n), 32)
        assert np.all(np.diff(out[:, 0]) > 0), n


def test_resample_samples_evenly_spaced_positions():
    idx = uniform_sample_indices(64, 32)
    gaps = np.diff(idx)
    assert set(gaps.tolist()) <= {2, 3} and gaps.max() - gaps.min() <= 1


def test_resample_is_deterministic():
    seq = np.random.default_rng(0).normal(size=(77, 330)).astype(np.float32)
    a, ma = resample_or_pad(seq, 32)
    b, mb = resample_or_pad(seq, 32)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_resample_exact_length_is_unchanged():
    seq = _frame_ids(32)
    out, mask = resample_or_pad(seq, 32)
    assert np.array_equal(out, seq) and mask.all()


def test_resample_short_sequence_pads_at_start_with_correct_mask():
    seq = _frame_ids(20)
    out, mask = resample_or_pad(seq, 32)
    assert out.shape == (32, 3)
    assert np.all(out[:12] == 0) and np.array_equal(out[12:], seq)  # order preserved
    assert np.array_equal(mask, [False] * 12 + [True] * 20)


def test_resample_very_short_sequences():
    out, mask = resample_or_pad(_frame_ids(1), 32)
    assert out.shape == (32, 3) and out[-1, 0] == 1 and np.all(out[:-1] == 0)
    assert mask.sum() == 1 and mask[-1]
    out, mask = resample_or_pad(np.zeros((0, 3), dtype=np.float32), 32)
    assert out.shape == (32, 3) and not mask.any()


def test_resample_keeps_real_all_zero_frames_marked_real():
    # A real frame with nothing detected is all-zero but must stay unmasked.
    out, mask = resample_or_pad(np.zeros((50, 3), dtype=np.float32), 32)
    assert mask.all()


def test_resample_propagates_real_frame_flags():
    flags = np.array([False] * 10 + [True] * 54)
    _, mask = resample_or_pad(_frame_ids(64), 32, real_flags=flags)
    assert np.array_equal(mask, flags[uniform_sample_indices(64, 32)])
    _, mask = resample_or_pad(_frame_ids(4), 8, real_flags=np.array([True, False, True, True]))
    assert np.array_equal(mask, [False] * 4 + [True, False, True, True])


def test_resample_preserves_feature_dimension_330():
    out, _ = resample_or_pad(np.ones((63, 330), dtype=np.float32), 32)
    assert out.shape == (32, 330) and out.dtype == np.float32
