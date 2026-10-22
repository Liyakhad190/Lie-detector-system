"""
Unit tests for PolygraphAnalyzer (rPPG pulse estimation and history buffering).
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.polygraph_analyzer import PolygraphAnalyzer


def test_polygraph_initialization():
    """Verify polygraph analyzer sets up display buffers."""
    poly = PolygraphAnalyzer(history_len=100, rppg_buffer_len=100)
    assert len(poly.history["stress"]) == 100
    assert len(poly.history["hr"]) == 100
    assert len(poly.history["deception"]) == 100
    assert poly.current_hr == 75.0


def test_polygraph_update_with_mock_frame():
    """Verify polygraph updates without throwing exceptions on synthetic frame and landmarks."""
    poly = PolygraphAnalyzer(history_len=50, rppg_buffer_len=50)
    
    # Create synthetic 720x1280 frame and 468 landmarks
    mock_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    # Give green channel slight warmth
    mock_frame[:, :, 1] = 140
    
    mock_landmarks = [(0.5, 0.4) for _ in range(468)]
    
    for _ in range(10):
        poly.update(mock_frame, mock_landmarks, stress_score=0.35, deception_prob=0.42)
        
    assert len(poly.green_signal) == 10
    assert 40.0 <= poly.current_hr <= 180.0
