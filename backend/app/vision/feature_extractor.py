"""Turns a raw video frame into a fixed-dimension feature vector.

This is the single place that assembles "hand features + pose features +
selected face features" into the flat vector the models consume. The
resulting dimension always matches ``settings.feature_dim`` — nothing here
hardcodes a feature size.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings
from app.vision.landmarks import FrameLandmarks, LandmarkExtractor
from app.vision.preprocessing import fill_missing, normalize_frame


def landmarks_to_vector(landmarks: FrameLandmarks) -> np.ndarray:
    """Flatten a (normalized, missing-filled) FrameLandmarks into a 1D
    feature vector honoring the use_pose/use_face config flags."""
    n_face_points = len(settings.face_landmark_indices) if settings.face_landmark_indices else 0
    filled = fill_missing(landmarks, n_face_points)

    parts = [filled.left_hand.reshape(-1), filled.right_hand.reshape(-1)]
    if settings.use_pose:
        parts.append(filled.pose.reshape(-1))
    if settings.use_face and n_face_points:
        parts.append(filled.face.reshape(-1))

    return np.concatenate(parts).astype(np.float32)


class FeatureExtractor:
    """High-level entry point: raw BGR frame -> feature vector.

    Wraps a ``LandmarkExtractor`` so callers (camera loop, training landmark
    extraction script) share one code path. When ``settings.use_velocity_features``
    is enabled, velocity (Δx, Δy, Δz per point since the previous call) is
    computed *here* — the single place feature vectors are produced — and
    appended to the position vector, so the exact same feature layout is
    used for training (training/extract_landmarks.py) and real-time
    inference (app/inference/predictor.py). Downstream code never needs to
    know whether velocity is enabled; it only ever sees vectors of
    ``settings.feature_dim`` length.
    """

    def __init__(self, landmark_extractor: Optional[LandmarkExtractor] = None):
        self._extractor = landmark_extractor or LandmarkExtractor()
        self._previous_vector: Optional[np.ndarray] = None

    @property
    def available(self) -> bool:
        return self._extractor.available

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FrameLandmarks]:
        """Returns (feature_vector, raw_landmarks). ``raw_landmarks`` is
        returned too so the caller (e.g. the WebSocket handler) can draw
        landmark overlays without re-running detection."""
        raw = self._extractor.process(frame_bgr)
        normalized = normalize_frame(raw)
        vector = landmarks_to_vector(normalized)

        if settings.use_velocity_features:
            previous = self._previous_vector if self._previous_vector is not None else np.zeros_like(vector)
            velocity = vector - previous
            self._previous_vector = vector
            vector = np.concatenate([vector, velocity]).astype(np.float32)

        assert vector.shape[0] == settings.feature_dim, (
            f"Feature vector dim mismatch: got {vector.shape[0]}, expected {settings.feature_dim}. "
            "Check use_pose/use_face/face_landmark_indices/use_velocity_features configuration."
        )
        return vector, raw

    def reset(self) -> None:
        """Clear velocity state — call this at the start of each new
        independent sequence (e.g. a new video in extract_landmarks.py, or
        a new sentence session) so velocity isn't computed against a
        leftover frame from an unrelated sequence."""
        self._previous_vector = None

    def close(self) -> None:
        self._extractor.close()
