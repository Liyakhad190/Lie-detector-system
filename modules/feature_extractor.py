"""
============================================================================
 FEATURE EXTRACTOR MODULE
============================================================================
 Aggregates features from all analyzer modules into a single feature
 vector suitable for ML model input.
============================================================================
"""

import numpy as np
from typing import Dict, List
import config


class FeatureExtractor:
    """Extracts and assembles ML feature vectors from raw analysis data."""

    def __init__(self):
        self._feature_names = config.FEATURE_NAMES

    def extract(self, blink_data: Dict, movement_data: Dict,
                micro_data: Dict, stress_data: Dict) -> np.ndarray:
        """
        Combine all behavioral metrics into a single feature vector.
        
        Returns:
            1D numpy array of features matching FEATURE_NAMES order.
        """
        features = np.zeros(len(self._feature_names), dtype=np.float32)

        # Blink features
        features[0] = blink_data.get("blink_rate_per_min", 0)
        features[1] = blink_data.get("blink_rate_per_min", 15) / 15.0  # ratio
        features[2] = blink_data.get("avg_ear", 0.3)
        features[3] = 0.0  # EAR std (computed externally)

        # Gaze features
        features[4] = movement_data.get("gaze_h_std", 0)
        features[5] = movement_data.get("gaze_v_std", 0)
        features[6] = movement_data.get("gaze_fixation_ratio", 1.0)

        # Head movement features
        features[7] = movement_data.get("head_pitch_std", 0)
        features[8] = movement_data.get("head_yaw_std", 0)
        features[9] = movement_data.get("head_roll_std", 0)
        features[10] = movement_data.get("movement_magnitude", 0)

        # Face symmetry
        features[11] = micro_data.get("face_symmetry", 1.0)
        features[12] = micro_data.get("face_symmetry_change", 0)

        # Lip features
        features[13] = micro_data.get("lip_movement", 0)
        features[14] = micro_data.get("lip_compression", 0.5)

        # Eyebrow features
        features[15] = micro_data.get("eyebrow_tension", 0)
        features[16] = micro_data.get("eyebrow_asymmetry", 0)

        # Micro-expression features
        features[17] = micro_data.get("micro_expression_count", 0)
        features[18] = micro_data.get("micro_expression_intensity", 0)

        # Stress/tension features
        features[19] = micro_data.get("facial_tension", 0)
        features[20] = movement_data.get("nervous_score", 0)
        features[21] = micro_data.get("jaw_clench", 0)

        # Replace NaN/inf with zeros
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        return features

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return self._feature_names.copy()

    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Convert feature vector to named dictionary."""
        return {name: float(val) for name, val in zip(self._feature_names, features)}
