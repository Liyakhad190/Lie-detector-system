"""
============================================================================
 BLINK DETECTOR MODULE
============================================================================
 Detects eye blinks using the Eye Aspect Ratio (EAR) method.
 EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
============================================================================
"""

import time
import numpy as np
from collections import deque
from typing import Tuple, Dict
import config


class BlinkDetector:
    """Real-time eye blink detection and frequency analysis using EAR."""

    def __init__(self):
        self._ear_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 10)
        self._ear_threshold = config.EAR_THRESHOLD
        self._consec_frames = 0
        self._blink_timestamps = deque(maxlen=200)
        self._total_blinks = 0
        self._blink_in_progress = False
        self._left_ear = 0.0
        self._right_ear = 0.0
        self._avg_ear = 0.0
        self._baseline_ear = None
        self._baseline_blink_rate = None
        self._start_time = time.time()

    @staticmethod
    def _compute_ear(eye_landmarks: np.ndarray) -> float:
        """Compute Eye Aspect Ratio from 6 landmark points."""
        if len(eye_landmarks) < 6:
            return 0.3
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        if h < 1e-6:
            return 0.3
        return (v1 + v2) / (2.0 * h)

    def update(self, landmarks: np.ndarray) -> Dict:
        """Process frame landmarks and detect blinks."""
        left_eye_pts = landmarks[config.LEFT_EYE_LANDMARKS]
        right_eye_pts = landmarks[config.RIGHT_EYE_LANDMARKS]
        self._left_ear = self._compute_ear(left_eye_pts)
        self._right_ear = self._compute_ear(right_eye_pts)
        self._avg_ear = (self._left_ear + self._right_ear) / 2.0
        self._ear_history.append(self._avg_ear)

        threshold = self._ear_threshold
        if self._baseline_ear is not None:
            threshold = self._baseline_ear * 0.75

        blink_detected = False
        if self._avg_ear < threshold:
            self._consec_frames += 1
        else:
            if self._consec_frames >= config.EAR_CONSEC_FRAMES:
                blink_detected = True
                self._total_blinks += 1
                self._blink_timestamps.append(time.time())
            self._consec_frames = 0
        self._blink_in_progress = self._avg_ear < threshold

        _, ear_std = self.get_ear_stats()
        return {
            "blink_detected": blink_detected,
            "blink_in_progress": self._blink_in_progress,
            "left_ear": self._left_ear,
            "right_ear": self._right_ear,
            "avg_ear": self._avg_ear,
            "ear_std": ear_std,                            # rolling EAR std
            "blink_rate_change": self.get_blink_rate_change(),  # ratio vs baseline
            "total_blinks": self._total_blinks,
            "blink_rate_per_min": self.get_blink_rate(),
            "ear_threshold": threshold,
        }

    def get_blink_rate(self, window_seconds: float = 60.0) -> float:
        """Calculate blinks per minute over a time window."""
        now = time.time()
        cutoff = now - window_seconds
        recent_blinks = sum(1 for ts in self._blink_timestamps if ts > cutoff)
        elapsed = min(window_seconds, now - self._start_time)
        if elapsed < 1.0:
            return 0.0
        return (recent_blinks / elapsed) * 60.0

    def get_blink_rate_change(self) -> float:
        """Compute blink rate change ratio relative to baseline."""
        if self._baseline_blink_rate is None or self._baseline_blink_rate < 1.0:
            return 1.0
        return self.get_blink_rate() / self._baseline_blink_rate

    def get_ear_stats(self) -> Tuple[float, float]:
        """Get mean and std of recent EAR values."""
        if len(self._ear_history) < 5:
            return self._avg_ear, 0.0
        arr = np.array(list(self._ear_history))
        return float(np.mean(arr)), float(np.std(arr))

    def set_baseline(self, baseline_ear: float, baseline_blink_rate: float):
        """Set calibration baseline values."""
        self._baseline_ear = baseline_ear
        self._baseline_blink_rate = baseline_blink_rate

    def reset(self):
        """Reset all state."""
        self._ear_history.clear()
        self._blink_timestamps.clear()
        self._total_blinks = 0
        self._consec_frames = 0
        self._blink_in_progress = False
        self._start_time = time.time()
