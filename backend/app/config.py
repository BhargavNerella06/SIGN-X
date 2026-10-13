"""
Centralized configuration for SIGN-X.

All tunable values used across the vision pipeline, model, inference,
training and API layers live here. Nothing else in the codebase should
hardcode these values.

Configuration can be overridden via (in order of precedence):
1. Environment variables (prefixed ``SIGNX_``), e.g. ``SIGNX_CAMERA_INDEX=1``
2. A ``.env`` file in the backend/ directory
3. The defaults declared below

Training-specific overrides (model architecture, batch size, etc.) are
additionally read from ``training/config.yaml`` by the training scripts,
but the *defaults* for those values are still defined here so the runtime
inference server and the training pipeline never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout -----------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SIGNX_",
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "SIGN-X"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- Camera / capture ---
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # --- MediaPipe landmark extraction ---
    mediapipe_model_dir: Path = BACKEND_DIR / "models" / "mediapipe"
    holistic_model_name: str = "holistic_landmarker.task"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_face_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Which face landmark indices to keep (a small, configurable subset
    # instead of the full 478-point face mesh). Defaults follow the
    # standard MediaPipe Face Mesh key-point set: eyebrows, eyes, nose,
    # outer lips -> useful for coarse mouth/eyebrow shape without paying
    # for all 478 points.
    face_landmark_indices: List[int] = Field(
        default_factory=lambda: [
            70, 63, 105, 66, 107,      # left eyebrow
            336, 296, 334, 293, 300,   # right eyebrow
            33, 160, 158, 133, 153, 144,   # left eye
            362, 385, 387, 263, 373, 380,  # right eye
            1, 4, 5, 195, 197,             # nose bridge/tip
            61, 291, 0, 17, 13, 14, 78, 308,  # outer/inner lips
        ]
    )

    # --- Feature engineering ---
    use_pose: bool = True
    use_face: bool = True
    use_velocity_features: bool = False  # Δx, Δy, Δz between frames
    normalize_landmarks: bool = True

    # --- Temporal buffer / sequence ---
    sequence_length: int = 32  # WINDOW_SIZE (T)
    # Raw live frames kept in the realtime buffer; resampled to sequence_length
    # with the same uniform sampling as training clips. At the frontend's
    # default 15 fps capture, 40 frames span ~2.7 s — a full INCLUDE clip
    # (mean ~2.5 s at 25 fps, 1.3-4.4 s) uniformly sampled to 32 frames.
    # 15 fps / 40 / stride 4 / debounce 4 were chosen on the validation split
    # with a live replay of the real pipeline (10 fps lost ~10 points of
    # accuracy versus training's 25 fps; see docs/performance.md).
    realtime_window_frames: int = 40
    sequence_stride: int = 4   # STRIDE for sliding inference window

    # --- Inference / smoothing ---
    # 0.5 / 3 were chosen on the validation split (live replay) for the
    # Transformer+BiLSTM ensemble (live_ensemble), whose averaged
    # probabilities run lower than a single model's. With a single model
    # (live_ensemble=false) the previous 0.6 / 4 is the precision-oriented
    # choice. See docs/performance.md.
    confidence_threshold: float = 0.5
    unknown_label: str = "UNKNOWN"
    smoothing_window: int = 5           # frames considered for majority vote
    debounce_repeats: int = 3           # consecutive agreeing confident votes before commit (<= smoothing_window)
    min_frames_between_commits: int = 10  # debounce: min frames before repeating same sign
    # Skip the model (report UNKNOWN) when fewer than this fraction of the
    # live window's frames contain a detected hand. Without it, an empty
    # scene (all-zero landmarks) is classified as a confident sign. Every
    # isl_include_subset28 clip has hands in >= 40% of its 32 frames.
    min_hand_frame_ratio: float = 0.25

    # --- Model ---
    model_type: Literal["transformer", "bilstm"] = "transformer"
    # Live server only: average model_type's probabilities with the other
    # trained checkpoint (Transformer+BiLSTM). Training/evaluation scripts
    # are unaffected. Falls back to model_type alone if the other checkpoint
    # is missing or incompatible.
    live_ensemble: bool = True
    models_dir: Path = BACKEND_DIR / "models"
    transformer_checkpoint: str = "isl_transformer.pt"
    bilstm_checkpoint: str = "isl_bilstm.pt"
    classes_file: str = "classes.json"

    embedding_dim: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ff_dim: int = 256
    dropout: float = 0.2
    bilstm_hidden_dim: int = 128
    bilstm_layers: int = 2

    device: Literal["auto", "cuda", "cpu"] = "auto"
    # PyTorch CPU threads for the inference server (training is unaffected).
    # Live inference is one tiny sequence at a time (~2.5 ms either way), and
    # idle intra-op threads busy-wait between calls: 6 threads cost ~500% CPU
    # at 15 calls/s versus ~3% with 1 thread (docs/performance.md).
    inference_threads: int = 1

    # --- Dataset / training ---
    dataset_path: Path = PROJECT_ROOT / "backend" / "datasets"
    vocabulary: List[str] = Field(
        default_factory=lambda: [
            "HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY", "GOOD", "BAD",
            "HELP", "STOP", "COME", "GO", "WATER", "FOOD", "NAME", "HOW",
            "YOU", "I", "WE", "WELCOME",
        ]
    )
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 50

    # --- Speech (TTS) ---
    tts_engine: Literal["pyttsx3", "none"] = "pyttsx3"
    tts_rate: int = 175
    tts_volume: float = 1.0
    tts_voice_id: Optional[str] = None

    # --- API / WebSocket ---
    # Env override is a JSON list, e.g. SIGNX_CORS_ORIGINS='["https://app.example.com"]'.
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    websocket_send_interval_ms: int = 0  # 0 = send every processed frame
    # POST /api/model/reload is unauthenticated: turn it off on a public deployment.
    allow_model_reload: bool = True

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to an actual torch device string, without importing
        torch at module import time (keeps config.py lightweight)."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def feature_dim(self) -> int:
        """Total per-frame feature dimension, derived from which landmark
        groups are enabled. Never hardcode this elsewhere — always read it
        from here so the model input layer stays consistent with whatever
        feature groups are turned on.
        """
        # 2 hands * 21 landmarks * 3 coords (x, y, z)
        dim = 2 * 21 * 3
        if self.use_pose:
            dim += 33 * 3  # MediaPipe pose has 33 landmarks
        if self.use_face:
            dim += len(self.face_landmark_indices) * 3
        if self.use_velocity_features:
            dim *= 2  # position features + velocity features
        return dim


settings = Settings()
