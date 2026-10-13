"""OpenCV webcam capture wrapper.

Handles the "no webcam" / "webcam disconnected" cases gracefully instead of
raising uncaught exceptions into the WebSocket loop.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a read fails repeatedly."""


class Camera:
    """Thin wrapper around ``cv2.VideoCapture`` with graceful failure modes."""

    def __init__(self, index: Optional[int] = None):
        self.index = index if index is not None else settings.camera_index
        self._cap = None

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if _is_windows() else 0)
        if not self._cap.isOpened():
            # Retry without the backend hint — CAP_DSHOW is Windows-only and
            # occasionally unavailable in virtualized/headless environments.
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            logger.warning("Could not open camera index %s", self.index)
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, settings.target_fps)
        logger.info("Camera %s opened (%dx%d @ %d fps target)", self.index,
                    settings.frame_width, settings.frame_height, settings.target_fps)
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None (never raises) if the read fails,
        so callers can treat a dropped frame as a no-op rather than a crash."""
        if not self.is_open:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the camera closes or fails."""
        while self.is_open:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _is_windows() -> bool:
    import platform

    return platform.system().lower() == "windows"
