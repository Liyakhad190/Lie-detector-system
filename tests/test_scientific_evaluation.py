"""
Unit tests for scientific evaluation metrics (Wilson score intervals, ECE, Brier score).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate_model import wilson_score_interval, compute_expected_calibration_error


def test_wilson_score_interval():
    """Verify Wilson score confidence interval calculation."""
    # 50 out of 100 successes (50%)
    lower, upper = wilson_score_interval(50, 100, confidence=0.95)
    assert 0.39 <= lower <= 0.41
    assert 0.59 <= upper <= 0.61
    assert lower < upper
    
    # 100 out of 100 successes (100%)
    lower, upper = wilson_score_interval(100, 100, confidence=0.95)
    assert lower > 0.95
    assert upper == 1.0
    
    # Zero total
    lower, upper = wilson_score_interval(0, 0)
    assert lower == 0.0 and upper == 0.0


def test_expected_calibration_error():
    """Verify ECE calculation with perfect and miscalibrated probabilities."""
    y_true = np.array([1, 1, 0, 0])
    
    # Perfectly calibrated
    y_perf = np.array([1.0, 1.0, 0.0, 0.0])
    ece_perf = compute_expected_calibration_error(y_true, y_perf, n_bins=5)
    assert ece_perf == pytest.approx(0.0, abs=1e-5)
    
    # Completely miscalibrated
    y_bad = np.array([0.0, 0.0, 1.0, 1.0])
    ece_bad = compute_expected_calibration_error(y_true, y_bad, n_bins=5)
    assert ece_bad > 0.8
