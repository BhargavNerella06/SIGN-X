"""Real-time inference orchestrator.

Wires together: FeatureExtractor -> TemporalBuffer -> SignClassifier ->
PredictionSmoother -> SentenceBuilder -> metrics, and exposes a single
``process_frame`` call that the WebSocket handler uses per captured frame.

This is the concrete implementation of the pipeline described in
docs/architecture.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from app.config import settings
from app.inference.smoothing import PredictionSmoother
from app.inference.temporal_buffer import TemporalBuffer
from app.language.sentence_builder import SentenceBuilder
from app.models.model_loader import ModelBundle, get_model_bundle
from app.utils.logging import get_logger
from app.utils.metrics import PipelineMetrics, StageTimer
from app.vision.feature_extractor import FeatureExtractor
from app.vision.landmarks import FrameLandmarks

logger = get_logger(__name__)


@dataclass
class PredictionMessage:
    """Matches the WebSocket ``prediction`` message schema (Section 18)."""

    type: str = "prediction"
    sign: str = "UNKNOWN"
    confidence: float = 0.0
    is_unknown: bool = True
    committed: bool = False
    # Running count of commits on this connection; lets the client ignore a
    # commit event it has already recorded.
    commit_id: int = 0
    sentence: str = ""
    top_predictions: List[dict] = field(default_factory=list)
    fps: float = 0.0
    latency_ms: float = 0.0
    landmark_ms: float = 0.0
    inference_ms: float = 0.0
    frames: int = 0
    hands_detected: bool = False
    landmarks: Optional[Dict[str, List[List[float]]]] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _landmarks_overlay_payload(raw: FrameLandmarks) -> Dict[str, List[List[float]]]:
    """Raw (image-space, 0..1 normalized) x/y points for drawing an overlay
    on the video element client-side — deliberately separate from the
    body-relative normalized features used for model input."""
    payload: Dict[str, List[List[float]]] = {}
    for name, group in (("left_hand", raw.left_hand), ("right_hand", raw.right_hand), ("pose", raw.pose)):
        if group is not None:
            payload[name] = [[round(float(x), 4), round(float(y), 4)] for x, y, _ in group]
    return payload


class RealtimePredictor:
    """Stateful per-connection pipeline. One instance should be created per
    active WebSocket client/camera session (it holds temporal state)."""

    def __init__(self, model_bundle: Optional[ModelBundle] = None):
        self.bundle = model_bundle or get_model_bundle()
        self.feature_extractor = FeatureExtractor()
        self.buffer = TemporalBuffer(
            window_size=settings.realtime_window_frames,
            feature_dim=settings.feature_dim,
            stride=settings.sequence_stride,
            output_length=settings.sequence_length,
        )
        # Per-frame "any hand detected" flags, aligned with the buffer window.
        self._hand_flags: deque[bool] = deque(maxlen=settings.realtime_window_frames)
        self.smoother = PredictionSmoother()
        self.sentence_builder = SentenceBuilder()
        self.metrics = PipelineMetrics()
        self._commit_id = 0
        # True once the no-hands gate has fired and the window was dropped;
        # cleared when hands are back and the model runs again.
        self._idle = False
        self._last_prediction = PredictionMessage(sentence="", sign=settings.unknown_label)

    @property
    def vision_available(self) -> bool:
        return self.feature_extractor.available

    def process_frame(self, frame_bgr: np.ndarray) -> PredictionMessage:
        """Run one frame through the full pipeline. Never raises — vision
        or model failures degrade to an UNKNOWN prediction so a single bad
        frame cannot crash the WebSocket loop (Section 25)."""
        self.metrics.frame_processed()

        if not self.vision_available:
            return self._error_message("MediaPipe model unavailable")

        try:
            with StageTimer() as landmark_timer:
                feature_vector, raw_landmarks = self.feature_extractor.extract(frame_bgr)
            self.metrics.landmark_ms.record_duration(landmark_timer.elapsed_ms)
        except Exception:
            logger.exception("Feature extraction failed")
            return self._error_message("Landmark extraction failed on this frame")

        if not np.isfinite(feature_vector).all():
            # One corrupt frame would otherwise turn every prediction over the
            # next window into NaN. Zeros are the "not detected" encoding.
            logger.warning("Non-finite landmark features on this frame; treating them as missing")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        should_infer = self.buffer.push(feature_vector, is_real=True)
        self._hand_flags.append(raw_landmarks.has_any_hand)

        if should_infer:
            try:
                self._run_inference(raw_landmarks)
            except Exception:
                logger.exception("Model inference failed")
                self.smoother.clear_pending()
                return self._error_message("Inference failed on this frame")

        msg = self._last_prediction
        msg.type = "prediction"
        msg.message = None
        # The same message is re-sent on every frame until the next stride;
        # a commit must be reported on exactly one of them.
        if not should_infer:
            msg.committed = False
        msg.hands_detected = raw_landmarks.has_any_hand
        msg.landmarks = _landmarks_overlay_payload(raw_landmarks)
        msg.fps = self.metrics.fps_tracker.fps
        msg.landmark_ms = self.metrics.landmark_ms.avg_duration_ms
        msg.inference_ms = self.metrics.inference_ms.avg_duration_ms
        msg.latency_ms = round(msg.landmark_ms + msg.inference_ms, 2)
        msg.frames = self.metrics.total_frames
        return msg

    def _run_inference(self, raw_landmarks: FrameLandmarks) -> None:
        hand_ratio = sum(self._hand_flags) / max(len(self._hand_flags), 1)
        if hand_ratio < settings.min_hand_frame_ratio:
            # No signing in view: don't let the model classify an empty scene.
            # Drop the window once on entering this state so the few hand
            # frames left over from the previous sign can neither count
            # towards the gate nor reach the model when signing resumes.
            if not self._idle:
                self._idle = True
                self._reset_window()
            self._set_unknown()
            return

        recent_hand_frames = list(self._hand_flags)[-self.buffer.stride:]
        if not any(recent_hand_frames):
            # Hands left the view since the last inference. The window still
            # holds the previous sign, so classifying it would only re-report
            # (or commit) a stale sign.
            self._set_unknown()
            return

        self._idle = False

        sequence, mask = self.buffer.get_sequence()
        x = torch.from_numpy(sequence).unsqueeze(0)       # (1, T, D)
        m = torch.from_numpy(mask).unsqueeze(0)            # (1, T)

        with StageTimer() as inference_timer:
            result = self.bundle.classifier.predict(x, mask=m, top_k=3)
        self.metrics.inference_ms.record_duration(inference_timer.elapsed_ms)

        smoothed = self.smoother.update(result.best_label, result.best_prob)

        committed = False
        if smoothed.committed_label:
            committed = self.sentence_builder.add_gloss(smoothed.committed_label)
        if committed:
            self._commit_id += 1

        self._last_prediction = PredictionMessage(
            sign=smoothed.display_label,
            confidence=smoothed.display_confidence,
            is_unknown=smoothed.is_unknown,
            committed=committed,
            commit_id=self._commit_id,
            sentence=self.sentence_builder.get_text(),
            top_predictions=[
                {"label": label, "confidence": prob}
                for label, prob in zip(result.top_labels, result.top_probs)
            ],
        )

    def _set_unknown(self) -> None:
        """Report UNKNOWN for the current frame and discard any pending
        candidate. Committed signs (sentence, commit_id) are kept."""
        self.smoother.clear_pending()
        self._last_prediction = PredictionMessage(
            sign=settings.unknown_label,
            confidence=0.0,
            is_unknown=True,
            commit_id=self._commit_id,
            sentence=self.sentence_builder.get_text(),
        )

    def _reset_window(self) -> None:
        self.buffer.reset()
        self._hand_flags.clear()
        self.feature_extractor.reset()

    def _error_message(self, reason: str) -> PredictionMessage:
        logger.warning(reason)
        msg = self._last_prediction
        msg.type = "error"
        msg.message = reason
        msg.committed = False  # a commit is reported on one prediction message only
        return msg

    def reset_stream(self) -> None:
        """Start over on a new video source (e.g. the user switched camera):
        drop the temporal window, smoothing state and current prediction so
        nothing from the old source can be displayed or committed. The
        committed sentence is kept."""
        self._reset_window()
        self.smoother.reset()
        self._idle = False
        self._set_unknown()

    def clear_sentence(self) -> None:
        self.sentence_builder.clear()
        self.smoother.reset()
        self.feature_extractor.reset()
        self._last_prediction.sentence = ""

    def close(self) -> None:
        self.feature_extractor.close()
