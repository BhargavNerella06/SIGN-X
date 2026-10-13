"""End-to-end offline pipeline test: recorded video -> landmark extraction
-> feature preprocessing -> temporal sequence -> model -> prediction
(Section 17/21). Uses a small synthetically generated video file (not the
real ISL dataset, which is far too large for a unit test) purely to verify
the pipeline's plumbing is correct end to end.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from app.models.classifier import SignClassifier, build_model
from training.extract_landmarks import extract_video_fresh
from app.vision.preprocessing import resample_or_pad


def _write_synthetic_video(path, num_frames=12, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        frame = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def test_offline_pipeline_video_to_prediction(tmp_path):
    video_path = tmp_path / "sample.avi"
    _write_synthetic_video(video_path, num_frames=12)

    # 1. video -> landmark extraction / feature preprocessing (the SAME
    #    extractor used by both training and real-time inference).
    #    A fresh MediaPipe instance per video, as the dataset builders use.
    sequence = extract_video_fresh(video_path)

    assert sequence is not None
    assert sequence.shape[0] == 12  # one feature vector per frame
    feature_dim = sequence.shape[1]

    # 2. temporal sequence: the shared resample/pad step + real-frame mask.
    window = 32
    padded, mask = resample_or_pad(sequence, window)
    assert padded.shape == (window, feature_dim)
    assert mask.sum() == 12  # 12 real frames, rest is left-padding

    # 3. trained (here: freshly initialized, since no real checkpoint is
    #    required for a plumbing test) model -> prediction.
    class_names = ["HELLO", "YES", "NO"]
    model = build_model("transformer", input_dim=feature_dim, num_classes=len(class_names),
                         embedding_dim=16, transformer_layers=1, transformer_heads=2,
                         transformer_ff_dim=32, max_seq_len=window)
    classifier = SignClassifier(model, class_names, device="cpu")

    x = torch.from_numpy(padded).unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0)
    result = classifier.predict(x, mask=m, top_k=3)

    # 4. "compare prediction with known label" — for a synthetic clip there
    #    is no ground truth, so we only assert the pipeline produced a
    #    well-formed prediction (this is a plumbing test, not an accuracy claim).
    assert result.best_label in class_names
    assert 0.0 <= result.best_prob <= 1.0
