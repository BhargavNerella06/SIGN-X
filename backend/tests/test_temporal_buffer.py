"""Tests for app.inference.temporal_buffer.TemporalBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from app.inference.temporal_buffer import TemporalBuffer


def test_buffer_not_ready_until_window_full():
    buf = TemporalBuffer(window_size=4, feature_dim=3, stride=1)
    results = [buf.push(np.ones(3)) for _ in range(3)]
    assert results == [False, False, False]
    assert not buf.is_full


def test_buffer_triggers_inference_once_full_with_stride_1():
    buf = TemporalBuffer(window_size=4, feature_dim=3, stride=1)
    for _ in range(3):
        buf.push(np.ones(3))
    assert buf.push(np.ones(3)) is True
    assert buf.is_full


def test_buffer_respects_stride_after_initial_fill():
    buf = TemporalBuffer(window_size=4, feature_dim=3, stride=3)
    triggers = [buf.push(np.ones(3)) for _ in range(4)]  # fills window on 4th push
    assert triggers == [False, False, False, True]
    # Next two pushes shouldn't trigger (stride=3), the third should.
    triggers2 = [buf.push(np.ones(3)) for _ in range(3)]
    assert triggers2 == [False, False, True]


def test_get_sequence_left_pads_when_not_full():
    buf = TemporalBuffer(window_size=5, feature_dim=2, stride=1)
    buf.push(np.array([1.0, 1.0]))
    buf.push(np.array([2.0, 2.0]))
    sequence, mask = buf.get_sequence()
    assert sequence.shape == (5, 2)
    assert mask.tolist() == [False, False, False, True, True]
    assert np.array_equal(sequence[3], [1.0, 1.0])
    assert np.array_equal(sequence[4], [2.0, 2.0])
    assert np.all(sequence[:3] == 0)


def test_get_sequence_full_window_all_real():
    buf = TemporalBuffer(window_size=3, feature_dim=2, stride=1)
    for v in (1, 2, 3):
        buf.push(np.array([float(v), float(v)]))
    sequence, mask = buf.get_sequence()
    assert mask.all()
    assert np.array_equal(sequence, [[1, 1], [2, 2], [3, 3]])


def test_push_rejects_wrong_feature_dim():
    buf = TemporalBuffer(window_size=3, feature_dim=4, stride=1)
    with pytest.raises(ValueError):
        buf.push(np.ones(3))


def test_reset_clears_state():
    buf = TemporalBuffer(window_size=3, feature_dim=2, stride=1)
    buf.push(np.ones(2))
    buf.push(np.ones(2))
    buf.reset()
    assert not buf.is_full
    assert buf.real_frame_count == 0


@pytest.mark.parametrize("window_size,stride", [(0, 1), (3, 0), (-1, 1)])
def test_invalid_construction_raises(window_size, stride):
    with pytest.raises(ValueError):
        TemporalBuffer(window_size=window_size, feature_dim=2, stride=stride)


def test_buffer_output_length_uses_shared_uniform_resampling():
    from app.vision.preprocessing import resample_or_pad

    buf = TemporalBuffer(window_size=64, feature_dim=3, stride=1, output_length=32)
    frames = [np.full(3, i + 1, dtype=np.float32) for i in range(64)]
    for f in frames:
        buf.push(f)
    seq, mask = buf.get_sequence()
    expected, expected_mask = resample_or_pad(np.stack(frames), 32)
    assert seq.shape == (32, 3) and mask.all()
    assert np.array_equal(seq, expected) and np.array_equal(mask, expected_mask)
    assert seq[0, 0] == 1 and seq[-1, 0] == 64  # whole window represented


def test_buffer_output_length_pads_when_partially_filled():
    buf = TemporalBuffer(window_size=64, feature_dim=3, stride=1, output_length=32)
    for i in range(10):
        buf.push(np.full(3, i + 1, dtype=np.float32))
    seq, mask = buf.get_sequence()
    assert seq.shape == (32, 3)
    assert np.array_equal(mask, [False] * 22 + [True] * 10)
