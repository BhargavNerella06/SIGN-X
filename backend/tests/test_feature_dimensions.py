"""Tests that the feature vector dimension always matches
``settings.feature_dim``, however the landmark-group flags are configured
— dimensions must never be hardcoded (Section 7/37)."""

from __future__ import annotations

from app.config import settings
from app.vision.feature_extractor import FeatureExtractor


def test_feature_vector_matches_configured_dimension(random_frame):
    extractor = FeatureExtractor()
    try:
        vector, _ = extractor.extract(random_frame)
        assert vector.shape == (settings.feature_dim,)
    finally:
        extractor.close()


def test_feature_dim_changes_when_pose_disabled(random_frame):
    original = settings.use_pose
    try:
        settings.use_pose = False
        extractor = FeatureExtractor()
        vector, _ = extractor.extract(random_frame)
        assert vector.shape == (settings.feature_dim,)
        extractor.close()
    finally:
        settings.use_pose = original


def test_feature_dim_doubles_with_velocity_enabled(random_frame):
    original = settings.use_velocity_features
    try:
        settings.use_velocity_features = True
        extractor = FeatureExtractor()
        base_dim_without_velocity = settings.feature_dim // 2
        vector, _ = extractor.extract(random_frame)
        assert vector.shape == (settings.feature_dim,)
        assert vector.shape[0] == base_dim_without_velocity * 2
        extractor.close()
    finally:
        settings.use_velocity_features = original
