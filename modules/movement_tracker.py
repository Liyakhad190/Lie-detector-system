"""
============================================================================
 MOVEMENT TRACKER MODULE
============================================================================
 Tracks head pose (pitch/yaw/roll), eye gaze direction, and overall
 movement patterns to detect nervous behaviors.
============================================================================
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple
import config


class MovementTracker:
    """Tracks head movement, gaze direction, and nervous motion patterns."""

    def __init__(self):
        self._head_pitch_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._head_yaw_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._head_roll_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._gaze_h_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._gaze_v_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._nose_positions = deque(maxlen=60)
        self._prev_nose = None
        self._baseline_head = None
        self._baseline_gaze = None

    def update(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict:
        """Process landmarks to extract head pose and gaze data."""
        h, w = frame_shape
        # Head pose estimation using key facial points
        nose = landmarks[config.NOSE_TIP][:2]
        chin = landmarks[config.CHIN[0]][:2]
        forehead = np.mean(landmarks[config.FOREHEAD][:, :2], axis=0)
        left_face = np.mean(landmarks[config.LEFT_FACE_CONTOUR][:, :2], axis=0)
        right_face = np.mean(landmarks[config.RIGHT_FACE_CONTOUR][:, :2], axis=0)

        # Approximate head angles (normalized to frame)
        face_center = (left_face + right_face) / 2.0
        face_width = np.linalg.norm(right_face - left_face)
        face_height = np.linalg.norm(chin - forehead)

        # Yaw: horizontal head rotation
        yaw = (nose[0] - face_center[0]) / (face_width + 1e-6) * 90.0
        # Pitch: vertical head tilt
        pitch = (nose[1] - face_center[1]) / (face_height + 1e-6) * 90.0
        # Roll: head tilt
        dx = right_face[0] - left_face[0]
        dy = right_face[1] - left_face[1]
        roll = np.degrees(np.arctan2(dy, dx))

        self._head_pitch_history.append(pitch)
        self._head_yaw_history.append(yaw)
        self._head_roll_history.append(roll)

        # Eye gaze tracking using iris landmarks
        gaze_h, gaze_v = self._compute_gaze(landmarks, w, h)
        self._gaze_h_history.append(gaze_h)
        self._gaze_v_history.append(gaze_v)

        # Track nose movement for nervous motion
        self._nose_positions.append(nose.copy())
        head_movement = 0.0
        if self._prev_nose is not None:
            head_movement = np.linalg.norm(nose - self._prev_nose)
        self._prev_nose = nose.copy()

        return {
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
            "gaze_horizontal": gaze_h,
            "gaze_vertical": gaze_v,
            "head_movement": head_movement,
            "head_pitch_std": self._get_std(self._head_pitch_history),
            "head_yaw_std": self._get_std(self._head_yaw_history),
            "head_roll_std": self._get_std(self._head_roll_history),
            "gaze_h_std": self._get_std(self._gaze_h_history),
            "gaze_v_std": self._get_std(self._gaze_v_history),
            "movement_magnitude": self._get_movement_magnitude(),
            "gaze_fixation_ratio": self._get_fixation_ratio(),
            "nervous_score": self._compute_nervous_score(),
        }

    def _compute_gaze(self, landmarks: np.ndarray, w: int, h: int) -> Tuple[float, float]:
        """Compute gaze direction from iris and eye landmarks."""
        try:
            left_iris_center = np.mean(landmarks[config.LEFT_IRIS][:, :2], axis=0)
            right_iris_center = np.mean(landmarks[config.RIGHT_IRIS][:, :2], axis=0)
            left_eye_center = np.mean(landmarks[config.LEFT_EYE_LANDMARKS][:, :2], axis=0)
            right_eye_center = np.mean(landmarks[config.RIGHT_EYE_LANDMARKS][:, :2], axis=0)
            left_offset = left_iris_center - left_eye_center
            right_offset = right_iris_center - right_eye_center
            avg_offset = (left_offset + right_offset) / 2.0
            gaze_h = avg_offset[0] / (w * 0.01 + 1e-6)
            gaze_v = avg_offset[1] / (h * 0.01 + 1e-6)
            return float(gaze_h), float(gaze_v)
        except (IndexError, ValueError):
            return 0.0, 0.0

    def _get_std(self, history: deque) -> float:
        if len(history) < 5:
            return 0.0
        return float(np.std(list(history)))

    def _get_movement_magnitude(self) -> float:
        """Compute overall head movement magnitude from recent positions."""
        if len(self._nose_positions) < 10:
            return 0.0
        positions = np.array(list(self._nose_positions))
        diffs = np.diff(positions, axis=0)
        magnitudes = np.linalg.norm(diffs, axis=1)
        return float(np.mean(magnitudes))

    def _get_fixation_ratio(self) -> float:
        """Compute gaze fixation ratio (how steady is the gaze)."""
        if len(self._gaze_h_history) < 10:
            return 1.0
        h_std = np.std(list(self._gaze_h_history)[-30:])
        v_std = np.std(list(self._gaze_v_history)[-30:])
        total_std = h_std + v_std
        fixation = max(0.0, 1.0 - total_std * 2.0)
        return float(fixation)

    def _compute_nervous_score(self) -> float:
        """Compute nervous movement score (0-1) from movement patterns."""
        if len(self._nose_positions) < 20:
            return 0.0
        movement_mag = self._get_movement_magnitude()
        pitch_std = self._get_std(self._head_pitch_history)
        yaw_std = self._get_std(self._head_yaw_history)
        gaze_var = self._get_std(self._gaze_h_history) + self._get_std(self._gaze_v_history)
        score = min(1.0, (movement_mag * 0.3 + pitch_std * 0.02 +
                          yaw_std * 0.02 + gaze_var * 0.5))
        return float(score)

    def set_baseline(self, head_data: Dict, gaze_data: Dict):
        """Set calibration baseline."""
        self._baseline_head = head_data
        self._baseline_gaze = gaze_data

    def reset(self):
        """Reset all tracking state."""
        for h in [self._head_pitch_history, self._head_yaw_history,
                  self._head_roll_history, self._gaze_h_history,
                  self._gaze_v_history, self._nose_positions]:
            h.clear()
        self._prev_nose = None
