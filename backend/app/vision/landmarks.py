"""MediaPipe landmark extraction.

Wraps the MediaPipe Tasks ``HolisticLandmarker`` (the current MediaPipe API —
the legacy ``mp.solutions.hands/pose/face_mesh`` API has been removed from
recent MediaPipe releases). A single Holistic model pass yields hands, pose
and face landmarks together, which is exactly the combination SIGN-X needs.

The model asset (``holistic_landmarker.task``) is NOT invented or bundled by
guesswork: it is Google's official, publicly hosted MediaPipe Tasks model.
See ``scripts/download_models.ps1`` / ``download_models.sh`` and
``docs/model.md`` for the exact source URL and download instructions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Silence MediaPipe/absl's verbose C++ logging before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_HAND_POINTS = 21
_POSE_POINTS = 33


@dataclass
class FrameLandmarks:
    """Raw (un-normalized) landmarks extracted from a single frame.

    Each array has shape (N, 3) for (x, y, z) in MediaPipe's normalized
    image coordinates, or is ``None`` when that group was not detected.
    """

    left_hand: Optional[np.ndarray]
    right_hand: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    face: Optional[np.ndarray]

    @property
    def has_any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


def _landmarks_to_array(landmarks) -> Optional[np.ndarray]:
    if not landmarks:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


class LandmarkExtractor:
    """Thin, resource-managing wrapper around MediaPipe's HolisticLandmarker.

    Usage:
        extractor = LandmarkExtractor()
        landmarks = extractor.process(bgr_frame)
        extractor.close()

    Or as a context manager:
        with LandmarkExtractor() as extractor:
            ...
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = Path(model_path) if model_path else (
            settings.mediapipe_model_dir / settings.holistic_model_name
        )
        self._landmarker = None
        self._available = False
        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "MediaPipe model not found at %s. Landmark extraction will be "
                "disabled until the model is downloaded — see "
                "scripts/download_models.ps1 or docs/model.md.",
                self._model_path,
            )
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options = vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=settings.min_face_detection_confidence,
                min_pose_detection_confidence=settings.min_pose_detection_confidence,
                min_hand_landmarks_confidence=settings.min_tracking_confidence,
            )
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
            self._mp = mp
            self._available = True
            logger.info("MediaPipe HolisticLandmarker loaded from %s", self._model_path)
        except Exception:
            logger.exception("Failed to initialize MediaPipe HolisticLandmarker")
            self._landmarker = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """Run landmark detection on a single BGR frame (as returned by
        OpenCV). Returns an empty FrameLandmarks (all None) if the model
        is unavailable or nothing was detected — callers must handle this
        gracefully rather than crashing."""
        if not self._available or frame_bgr is None:
            return FrameLandmarks(None, None, None, None)

        try:
            frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, no copy needed for read-only use
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            result = self._landmarker.detect(mp_image)
        except Exception:
            logger.exception("MediaPipe detection failed on frame")
            return FrameLandmarks(None, None, None, None)

        face = _landmarks_to_array(result.face_landmarks)
        if face is not None and settings.face_landmark_indices:
            valid_idx = [i for i in settings.face_landmark_indices if i < face.shape[0]]
            face = face[valid_idx]

        return FrameLandmarks(
            left_hand=_landmarks_to_array(result.left_hand_landmarks),
            right_hand=_landmarks_to_array(result.right_hand_landmarks),
            pose=_landmarks_to_array(result.pose_landmarks),
            face=face,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._available = False

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def empty_hand() -> np.ndarray:
    return np.zeros((_HAND_POINTS, 3), dtype=np.float32)


def empty_pose() -> np.ndarray:
    return np.zeros((_POSE_POINTS, 3), dtype=np.float32)


def empty_face(n_points: int) -> np.ndarray:
    return np.zeros((n_points, 3), dtype=np.float32)
