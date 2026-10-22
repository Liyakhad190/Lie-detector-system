"""
============================================================================
 CALIBRATION MANAGER MODULE
============================================================================
 Manages the calibration phase where baseline behavioral metrics are
 established before lie detection begins.
============================================================================
"""

import time
import numpy as np
from typing import Dict, Optional, List
import config


class CalibrationManager:
    """Manages baseline calibration before detection mode."""

    def __init__(self):
        self._is_calibrating = False
        self._calibration_complete = False
        self._start_time = 0.0
        self._samples: List[Dict] = []
        self._baseline: Optional[Dict] = None
        self._duration = config.CALIBRATION_DURATION

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    @property
    def is_complete(self) -> bool:
        return self._calibration_complete

    @property
    def baseline(self) -> Optional[Dict]:
        return self._baseline

    def start(self):
        """Start calibration phase."""
        self._is_calibrating = True
        self._calibration_complete = False
        self._start_time = time.time()
        self._samples.clear()
        self._baseline = None
        print("[CALIBRATION] Started - Please look at the camera naturally")

    def add_sample(self, blink_data: Dict, movement_data: Dict,
                   micro_data: Dict, stress_data: Dict, features: np.ndarray):
        """Add a calibration sample."""
        if not self._is_calibrating:
            return

        self._samples.append({
            "blink_data": blink_data.copy(),
            "movement_data": movement_data.copy(),
            "micro_data": micro_data.copy(),
            "stress_data": stress_data.copy(),
            "features": features.copy(),
        })

        # Check if calibration is complete
        elapsed = time.time() - self._start_time
        if elapsed >= self._duration and len(self._samples) >= 30:
            self._complete_calibration()

    def get_progress(self) -> float:
        """Get calibration progress (0.0 to 1.0)."""
        if not self._is_calibrating:
            return 1.0 if self._calibration_complete else 0.0
        elapsed = time.time() - self._start_time
        return min(1.0, elapsed / self._duration)

    def get_remaining_seconds(self) -> float:
        """Get remaining calibration time."""
        if not self._is_calibrating:
            return 0.0
        elapsed = time.time() - self._start_time
        return max(0.0, self._duration - elapsed)

    def _complete_calibration(self):
        """Compute baseline from collected samples."""
        self._is_calibrating = False
        self._calibration_complete = True

        if not self._samples:
            print("[CALIBRATION] No samples collected!")
            return

        # Average all metrics
        self._baseline = {
            "avg_ear": np.mean([s["blink_data"]["avg_ear"] for s in self._samples]),
            "blink_rate": np.mean([s["blink_data"]["blink_rate_per_min"] for s in self._samples]),
            "head_pitch_std": np.mean([s["movement_data"]["head_pitch_std"] for s in self._samples]),
            "head_yaw_std": np.mean([s["movement_data"]["head_yaw_std"] for s in self._samples]),
            "gaze_h_std": np.mean([s["movement_data"]["gaze_h_std"] for s in self._samples]),
            "gaze_v_std": np.mean([s["movement_data"]["gaze_v_std"] for s in self._samples]),
            "face_symmetry": np.mean([s["micro_data"]["face_symmetry"] for s in self._samples]),
            "lip_movement": np.mean([s["micro_data"]["lip_movement"] for s in self._samples]),
            "facial_tension": np.mean([s["micro_data"]["facial_tension"] for s in self._samples]),
            "stress_score": np.mean([s["stress_data"]["stress_score"] for s in self._samples]),
            "avg_features": np.mean([s["features"] for s in self._samples], axis=0),
        }

        print(f"[CALIBRATION] Complete! Baseline established from {len(self._samples)} samples")
        print(f"  Baseline EAR: {self._baseline['avg_ear']:.3f}")
        print(f"  Baseline Blink Rate: {self._baseline['blink_rate']:.1f}/min")
        print(f"  Baseline Symmetry: {self._baseline['face_symmetry']:.3f}")

    def reset(self):
        """Reset calibration."""
        self._is_calibrating = False
        self._calibration_complete = False
        self._samples.clear()
        self._baseline = None
