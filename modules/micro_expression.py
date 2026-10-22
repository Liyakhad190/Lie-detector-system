"""
============================================================================
 MICRO-EXPRESSION ANALYZER MODULE
============================================================================
 Detects rapid facial changes (micro-expressions) that may indicate
 concealed emotions. Tracks face symmetry, lip, and eyebrow changes.
============================================================================
"""

import numpy as np
from collections import deque
from typing import Dict
import config


class MicroExpressionAnalyzer:
    """Detects micro-expressions through rapid facial landmark changes."""

    def __init__(self):
        self._symmetry_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._lip_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._eyebrow_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._tension_history = deque(maxlen=config.FEATURE_WINDOW_SIZE * 5)
        self._micro_events = deque(maxlen=100)
        self._frame_count = 0
        self._baseline_symmetry = None
        self._baseline_lip = None

    def update(self, landmarks: np.ndarray) -> Dict:
        """Analyze landmarks for micro-expression indicators."""
        self._frame_count += 1

        # Face symmetry analysis
        symmetry = self._compute_symmetry(landmarks)
        self._symmetry_history.append(symmetry)

        # Lip movement analysis
        lip_data = self._analyze_lips(landmarks)
        self._lip_history.append(lip_data["lip_movement"])

        # Eyebrow analysis
        brow_data = self._analyze_eyebrows(landmarks)
        self._eyebrow_history.append(brow_data["eyebrow_height"])

        # Facial tension
        tension = self._compute_facial_tension(landmarks)
        self._tension_history.append(tension)

        # Detect micro-expression events (rapid changes)
        micro_count, micro_intensity = self._detect_micro_events()

        # Symmetry change from baseline
        sym_change = 0.0
        if self._baseline_symmetry is not None:
            sym_change = abs(symmetry - self._baseline_symmetry)

        return {
            "face_symmetry": symmetry,
            "face_symmetry_change": sym_change,
            "lip_movement": lip_data["lip_movement"],
            "lip_compression": lip_data["lip_compression"],
            "lip_openness": lip_data["lip_openness"],
            "eyebrow_tension": brow_data["eyebrow_tension"],
            "eyebrow_asymmetry": brow_data["eyebrow_asymmetry"],
            "eyebrow_height": brow_data["eyebrow_height"],
            "micro_expression_count": micro_count,
            "micro_expression_intensity": micro_intensity,
            "facial_tension": tension,
            "jaw_clench": self._detect_jaw_clench(landmarks),
        }

    def _compute_symmetry(self, landmarks: np.ndarray) -> float:
        """Compute face symmetry score (0=asymmetric, 1=symmetric)."""
        left_pts = landmarks[config.LEFT_FACE_CONTOUR][:, :2]
        right_pts = landmarks[config.RIGHT_FACE_CONTOUR][:, :2]
        nose = landmarks[config.NOSE_TIP][:2]

        # Mirror right points across nose vertical axis
        left_dists = np.linalg.norm(left_pts - nose, axis=1)
        right_dists = np.linalg.norm(right_pts - nose, axis=1)

        min_len = min(len(left_dists), len(right_dists))
        left_dists = left_dists[:min_len]
        right_dists = right_dists[:min_len]

        if np.max(left_dists) < 1e-6:
            return 1.0

        diff = np.abs(left_dists - right_dists)
        avg_dist = (left_dists + right_dists) / 2.0 + 1e-6
        normalized_diff = np.mean(diff / avg_dist)

        symmetry = max(0.0, 1.0 - normalized_diff * 3.0)
        return float(symmetry)

    def _analyze_lips(self, landmarks: np.ndarray) -> Dict:
        """Analyze lip movement and compression."""
        upper = np.mean(landmarks[config.INNER_UPPER_LIP][:, :2], axis=0)
        lower = np.mean(landmarks[config.INNER_LOWER_LIP][:, :2], axis=0)
        outer_upper = np.mean(landmarks[config.UPPER_LIP][:, :2], axis=0)
        outer_lower = np.mean(landmarks[config.LOWER_LIP][:, :2], axis=0)

        lip_openness = np.linalg.norm(upper - lower)
        lip_width = np.linalg.norm(
            landmarks[config.UPPER_LIP[0]][:2] - landmarks[config.UPPER_LIP[-1]][:2]
        )
        lip_compression = lip_openness / (lip_width + 1e-6)

        # Movement = change from previous frame
        movement = 0.0
        if len(self._lip_history) > 0:
            prev = self._lip_history[-1]
            movement = abs(lip_openness - prev)

        return {
            "lip_movement": float(lip_openness),
            "lip_compression": float(lip_compression),
            "lip_openness": float(lip_openness),
        }

    def _analyze_eyebrows(self, landmarks: np.ndarray) -> Dict:
        """Analyze eyebrow position and asymmetry."""
        left_brow = np.mean(landmarks[config.LEFT_EYEBROW][:, 1])
        right_brow = np.mean(landmarks[config.RIGHT_EYEBROW][:, 1])
        left_eye = np.mean(landmarks[config.LEFT_EYE_LANDMARKS][:, 1])
        right_eye = np.mean(landmarks[config.RIGHT_EYE_LANDMARKS][:, 1])

        # Brow height relative to eye
        left_height = left_eye - left_brow
        right_height = right_eye - right_brow
        avg_height = (left_height + right_height) / 2.0

        # Asymmetry
        asymmetry = abs(left_height - right_height) / (avg_height + 1e-6)

        # Tension (low eyebrows = tension)
        tension = max(0.0, 1.0 - avg_height * 0.05)

        return {
            "eyebrow_height": float(avg_height),
            "eyebrow_asymmetry": float(min(1.0, asymmetry)),
            "eyebrow_tension": float(min(1.0, tension)),
        }

    def _compute_facial_tension(self, landmarks: np.ndarray) -> float:
        """Compute overall facial tension from landmark positions."""
        # Measure distances between key points and detect compression
        brow_tension = self._analyze_eyebrows(landmarks)["eyebrow_tension"]
        jaw = self._detect_jaw_clench(landmarks)
        lip_comp = self._analyze_lips(landmarks)["lip_compression"]
        tension = (brow_tension * 0.4 + jaw * 0.3 + (1.0 - lip_comp) * 0.3)
        return float(min(1.0, tension))

    def _detect_jaw_clench(self, landmarks: np.ndarray) -> float:
        """Detect jaw clenching from chin-to-nose distance."""
        chin = landmarks[config.CHIN[0]][:2]
        nose = landmarks[config.NOSE_TIP][:2]
        distance = np.linalg.norm(chin - nose)
        # Normalize: smaller distance = more clenching
        clench = max(0.0, 1.0 - distance * 0.008)
        return float(clench)

    def _detect_micro_events(self):
        """Detect rapid changes in facial features (micro-expressions)."""
        count = 0
        intensity = 0.0

        if len(self._symmetry_history) >= config.MICRO_EXPRESSION_FRAMES:
            recent = list(self._symmetry_history)[-config.MICRO_EXPRESSION_FRAMES:]
            sym_change = max(recent) - min(recent)
            if sym_change > 0.15:
                count += 1
                intensity += sym_change

        if len(self._lip_history) >= config.MICRO_EXPRESSION_FRAMES:
            recent = list(self._lip_history)[-config.MICRO_EXPRESSION_FRAMES:]
            lip_change = max(recent) - min(recent)
            if lip_change > 3.0:
                count += 1
                intensity += lip_change * 0.1

        if len(self._eyebrow_history) >= config.MICRO_EXPRESSION_FRAMES:
            recent = list(self._eyebrow_history)[-config.MICRO_EXPRESSION_FRAMES:]
            brow_change = max(recent) - min(recent)
            if brow_change > 2.0:
                count += 1
                intensity += brow_change * 0.1

        return count, min(1.0, intensity)

    def set_baseline(self, symmetry: float, lip_movement: float):
        """Set calibration baseline."""
        self._baseline_symmetry = symmetry
        self._baseline_lip = lip_movement

    def reset(self):
        """Reset all state."""
        for h in [self._symmetry_history, self._lip_history,
                  self._eyebrow_history, self._tension_history,
                  self._micro_events]:
            h.clear()
        self._frame_count = 0
