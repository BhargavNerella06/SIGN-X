"""Landmark-extraction lifecycle: every source video is processed by its own
freshly created MediaPipe extractor, which is closed afterwards.

MediaPipe's HolisticLandmarker keeps internal state between calls (even in
IMAGE mode), so reusing one instance across videos makes a video's features
depend on the videos processed before it. These tests pin down the
create -> process -> close lifecycle and that it is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from training.extract_landmarks import extract_video_fresh

_REAL_VIDEO = (Path(__file__).resolve().parents[1] / "datasets" / "_include_zips" / "extracted"
               / "Colours" / "49. Blue" / "MVI_3726.MOV")


def _write_video(path: Path, num_frames: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), 10.0, (64, 64))
    for i in range(num_frames):
        writer.write(np.full((64, 64, 3), i * 20, dtype=np.uint8))
    writer.release()


class _FakeExtractor:
    instances: list["_FakeExtractor"] = []

    def __init__(self, available: bool = True, fail: bool = False):
        self.available = available
        self.fail = fail
        self.closed = False
        self.frames_seen = 0
        _FakeExtractor.instances.append(self)

    def reset(self) -> None:
        pass

    def extract(self, frame):
        if self.fail:
            raise ValueError("boom")
        self.frames_seen += 1
        return np.full(4, self.frames_seen, dtype=np.float32), None

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_instances():
    _FakeExtractor.instances.clear()


def test_each_video_gets_a_fresh_extractor_that_is_closed(tmp_path):
    videos = []
    for name in ("a.avi", "b.avi"):
        _write_video(tmp_path / name)
        videos.append(tmp_path / name)

    results = [extract_video_fresh(v, extractor_factory=_FakeExtractor) for v in videos]

    assert len(_FakeExtractor.instances) == 2  # one instance per video, never shared
    assert all(inst.closed for inst in _FakeExtractor.instances)
    # State inside the fake grows per call; a fresh instance means both videos start from scratch.
    assert np.array_equal(results[0], results[1])


def test_extractor_is_closed_even_when_extraction_fails(tmp_path):
    _write_video(tmp_path / "a.avi")
    with pytest.raises(ValueError):
        extract_video_fresh(tmp_path / "a.avi", extractor_factory=lambda: _FakeExtractor(fail=True))
    assert _FakeExtractor.instances[0].closed


def test_unavailable_extractor_raises_and_is_closed(tmp_path):
    _write_video(tmp_path / "a.avi")
    with pytest.raises(RuntimeError, match="MediaPipe model unavailable"):
        extract_video_fresh(tmp_path / "a.avi", extractor_factory=lambda: _FakeExtractor(available=False))
    assert _FakeExtractor.instances[0].closed


@pytest.mark.skipif(not _REAL_VIDEO.exists(), reason="INCLUDE video not downloaded (see docs/dataset.md)")
def test_fresh_extraction_of_real_video_is_deterministic():
    """With real MediaPipe on a real signer video (hands detected), repeated
    fresh-instance extraction must be bit-identical."""
    first = extract_video_fresh(_REAL_VIDEO)
    second = extract_video_fresh(_REAL_VIDEO)
    assert first is not None and first.shape[1] == 330
    assert np.abs(first[:, :126]).sum() > 0  # hands actually detected, so the check is meaningful
    assert np.array_equal(first, second)
