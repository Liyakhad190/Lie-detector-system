"""
Unit tests for DeceptionModel, heuristic fallback, and inference pipeline.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.ml_model import DeceptionModel


def test_model_initialization():
    """Verify DeceptionModel loads or falls back gracefully."""
    model = DeceptionModel()
    assert model is not None
    assert hasattr(model, "predict")


def test_model_prediction_shape_and_keys():
    """Verify inference returns all required contract fields."""
    model = DeceptionModel()
    dummy_features = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.5
    
    result = model.predict(dummy_features)
    assert isinstance(result, dict)
    assert "deception_probability" in result
    assert "truth_probability" in result
    assert "confidence" in result
    assert "confidence_interval" in result
    assert "label" in result
    assert "risk_tier" in result
    assert "explainability" in result
    
    # Check probability bounds
    assert 0.0 <= result["deception_probability"] <= 1.0
    assert 0.0 <= result["truth_probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    ci = result["confidence_interval"]
    assert len(ci) == 2
    assert ci[0] <= ci[1]


def test_model_handles_invalid_or_missing_features():
    """Verify robust handling of empty, short, or invalid feature vectors."""
    model = DeceptionModel()
    
    # None input
    res_none = model.predict(None)
    assert res_none["label"] == "NO DATA"
    assert res_none["deception_probability"] == 0.0
    
    # Truncated input
    res_short = model.predict(np.array([0.1, 0.2]))
    assert res_short["label"] == "NO DATA"


def test_model_temporal_smoothing():
    """Verify smoothing dampens abrupt frame-to-frame spikes."""
    model = DeceptionModel()
    model.reset()
    
    feat_low = np.zeros(len(config.FEATURE_NAMES), dtype=np.float32)
    feat_high = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 5.0
    
    p1 = model.predict(feat_low)["deception_probability"]
    p2 = model.predict(feat_high)["deception_probability"]
    
    # Temporal smoothing should prevent instant leap to 1.0
    assert p2 < 0.95


def test_model_autonomic_fusion():
    """Verify autonomic physiological input affects the decomposition."""
    model = DeceptionModel()
    dummy_features = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.2
    
    # Low heart rate
    res_low_hr = model.predict(dummy_features, polygraph_data={"estimated_hr": 70.0})
    # High heart rate
    res_high_hr = model.predict(dummy_features, polygraph_data={"estimated_hr": 125.0})
    
    auto_low = res_low_hr["explainability"]["channels"]["autonomic"]["score"]
    auto_high = res_high_hr["explainability"]["channels"]["autonomic"]["score"]
    assert auto_high > auto_low
