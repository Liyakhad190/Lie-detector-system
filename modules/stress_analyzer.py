"""
============================================================================
 STRESS ANALYZER MODULE
============================================================================
 Combines multiple behavioral indicators into a composite stress score.
============================================================================
"""

import numpy as np
from collections import deque
from typing import Dict
import config


class StressAnalyzer:
    """Computes composite stress level from multiple behavioral signals."""

    def __init__(self):
        self._stress_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._baseline_stress = None

    def compute_stress(self, blink_data: Dict, movement_data: Dict,
                       micro_data: Dict) -> Dict:
        """
        Compute composite stress score from all behavioral indicators.
        
        Weights:
          - Blink rate anomaly: 20%
          - Eye movement instability: 15%
          - Head movement: 15%
          - Face symmetry changes: 15%
          - Lip compression: 10%
          - Eyebrow tension: 10%
          - Micro-expressions: 10%
          - Jaw clenching: 5%
        """
        scores = {}

        # Blink rate stress (deviation from normal range)
        blink_rate = blink_data.get("blink_rate_per_min", 15)
        if blink_rate > config.STRESS_BLINK_RATE:
            scores["blink_stress"] = min(1.0, (blink_rate - 20) / 30.0)
        elif blink_rate < 8:
            scores["blink_stress"] = min(1.0, (8 - blink_rate) / 8.0)
        else:
            scores["blink_stress"] = 0.0

        # Gaze instability
        gaze_std = movement_data.get("gaze_h_std", 0) + movement_data.get("gaze_v_std", 0)
        scores["gaze_stress"] = min(1.0, gaze_std * 2.0)

        # Head movement
        head_mag = movement_data.get("movement_magnitude", 0)
        scores["head_stress"] = min(1.0, head_mag * 0.5)

        # Face symmetry
        sym_change = micro_data.get("face_symmetry_change", 0)
        scores["symmetry_stress"] = min(1.0, sym_change * 5.0)

        # Lip compression
        lip_comp = micro_data.get("lip_compression", 0.5)
        scores["lip_stress"] = min(1.0, max(0, 1.0 - lip_comp * 3.0))

        # Eyebrow tension
        scores["eyebrow_stress"] = micro_data.get("eyebrow_tension", 0)

        # Micro-expressions
        micro_int = micro_data.get("micro_expression_intensity", 0)
        scores["micro_stress"] = min(1.0, micro_int)

        # Jaw clenching
        scores["jaw_stress"] = micro_data.get("jaw_clench", 0)

        # Weighted composite
        weights = {
            "blink_stress": 0.20,
            "gaze_stress": 0.15,
            "head_stress": 0.15,
            "symmetry_stress": 0.15,
            "lip_stress": 0.10,
            "eyebrow_stress": 0.10,
            "micro_stress": 0.10,
            "jaw_stress": 0.05,
        }

        composite = sum(scores[k] * weights[k] for k in weights)
        composite = float(np.clip(composite, 0.0, 1.0))
        self._stress_history.append(composite)

        # Smoothed stress
        if len(self._stress_history) > 3:
            smoothed = np.mean(list(self._stress_history)[-10:])
        else:
            smoothed = composite

        # Stress level category
        if smoothed < 0.3:
            level = "LOW"
            color_key = "stress_low"
        elif smoothed < 0.6:
            level = "MEDIUM"
            color_key = "stress_medium"
        else:
            level = "HIGH"
            color_key = "stress_high"

        return {
            "stress_score": float(smoothed),
            "stress_raw": composite,
            "stress_level": level,
            "stress_color_key": color_key,
            "component_scores": scores,
        }

    def set_baseline(self, baseline_stress: float):
        """Set calibration baseline."""
        self._baseline_stress = baseline_stress

    def reset(self):
        """Reset state."""
        self._stress_history.clear()
