"""
Tests for ML audit inference fixes (2026-10):
  1. ear_std (idx=3) must read from blink_data dict, not be hardcoded 0.0
  2. blink_rate_change (idx=1) must be the detector change ratio, not blink_rate/15.0
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.blink_detector import BlinkDetector
from modules.feature_extractor import FeatureExtractor


def _blink(blink_rate=15.0, avg_ear=0.28, ear_std=0.03, change_ratio=1.2):
    return {
        "blink_rate_per_min": blink_rate, "avg_ear": avg_ear,
        "ear_std": ear_std, "blink_rate_change": change_ratio,
        "total_blinks": 5, "blink_detected": False, "blink_in_progress": False,
        "left_ear": avg_ear, "right_ear": avg_ear, "ear_threshold": 0.21,
    }

def _move():
    return {
        "gaze_h_std": 0.1, "gaze_v_std": 0.08, "gaze_fixation_ratio": 0.85,
        "head_pitch_std": 1.5, "head_yaw_std": 1.2, "head_roll_std": 0.8,
        "movement_magnitude": 0.5, "nervous_score": 0.1,
    }

def _micro():
    return {
        "face_symmetry": 0.92, "face_symmetry_change": 0.02,
        "lip_movement": 8.0, "lip_compression": 0.4,
        "eyebrow_tension": 0.2, "eyebrow_asymmetry": 0.05,
        "micro_expression_count": 0, "micro_expression_intensity": 0.05,
        "facial_tension": 0.2, "jaw_clench": 0.1,
    }

def _stress():
    return {"stress_score": 0.2, "stress_level": "LOW"}


def test_ear_std_uses_blink_data_value():
    """ear_std (idx=3) must read from blink_data, not be hardcoded 0.0."""
    ext = FeatureExtractor()
    feats = ext.extract(_blink(ear_std=0.035), _move(), _micro(), _stress())
    assert feats[3] == pytest.approx(0.035, abs=1e-5), (
        f"Got {feats[3]}, expected 0.035. Inference was hardcoding 0.0."
    )


def test_ear_std_defaults_to_zero_when_absent():
    """ear_std defaults to 0.0 when key absent (backward compat)."""
    ext = FeatureExtractor()
    bd = _blink()
    del bd["ear_std"]
    feats = ext.extract(bd, _move(), _micro(), _stress())
    assert feats[3] == pytest.approx(0.0, abs=1e-5)


def test_blink_rate_change_is_ratio_not_rate_over_15():
    """blink_rate_change (idx=1) must be the detector ratio, not blink_rate/15."""
    ext = FeatureExtractor()
    # blink_rate=30: old code produced 30/15=2.0; correct value is 1.5
    feats = ext.extract(_blink(blink_rate=30.0, change_ratio=1.5), _move(), _micro(), _stress())
    assert feats[1] == pytest.approx(1.5, abs=1e-5), (
        f"Got {feats[1]}, old code would have given 2.0."
    )


def test_blink_detector_exposes_ear_std_key():
    """BlinkDetector.get_ear_stats() returns nonzero std after multiple frames."""
    det = BlinkDetector()
    for v in [0.28, 0.29, 0.27, 0.30, 0.28, 0.26]:
        det._ear_history.append(v)
    _, std = det.get_ear_stats()
    assert std > 0.0, "get_ear_stats() std should be nonzero after enough frames"


def test_feature_vector_length():
    """Feature vector must always have exactly len(config.FEATURE_NAMES) elements."""
    ext = FeatureExtractor()
    feats = ext.extract(_blink(), _move(), _micro(), _stress())
    assert len(feats) == len(config.FEATURE_NAMES)
