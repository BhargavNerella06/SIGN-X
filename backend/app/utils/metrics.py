"""Lightweight runtime performance metrics: FPS, latency, per-stage timing.

Used by the inference pipeline and exposed to the frontend over the
WebSocket ``prediction``/``status`` messages so the UI can show live
FPS/latency instead of fabricated numbers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RollingRate:
    """Tracks a rolling average of durations (seconds) to derive FPS/latency."""

    window: int = 30
    _timestamps: deque = field(default_factory=deque)
    _durations_ms: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        self._timestamps = deque(maxlen=self.window)
        self._durations_ms = deque(maxlen=self.window)

    def tick(self) -> None:
        """Call once per processed frame to update the FPS estimate."""
        self._timestamps.append(time.perf_counter())

    def record_duration(self, duration_ms: float) -> None:
        """Record a stage duration in milliseconds (e.g. inference time)."""
        self._durations_ms.append(duration_ms)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return round((len(self._timestamps) - 1) / elapsed, 1)

    @property
    def avg_duration_ms(self) -> float:
        if not self._durations_ms:
            return 0.0
        return round(sum(self._durations_ms) / len(self._durations_ms), 2)


class StageTimer:
    """Context manager for timing a pipeline stage in milliseconds.

    Example:
        with StageTimer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


@dataclass
class PipelineMetrics:
    """Aggregate metrics for the full real-time pipeline."""

    fps_tracker: RollingRate = field(default_factory=RollingRate)
    landmark_ms: RollingRate = field(default_factory=RollingRate)
    inference_ms: RollingRate = field(default_factory=RollingRate)
    total_frames: int = 0

    def frame_processed(self) -> None:
        self.fps_tracker.tick()
        self.total_frames += 1

    def to_dict(self) -> dict:
        return {
            "fps": self.fps_tracker.fps,
            "latency_ms": self.inference_ms.avg_duration_ms + self.landmark_ms.avg_duration_ms,
            "landmark_extraction_ms": self.landmark_ms.avg_duration_ms,
            "model_inference_ms": self.inference_ms.avg_duration_ms,
            "frames": self.total_frames,
        }
