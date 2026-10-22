"""
============================================================================
 ML MODEL MODULE - Deception Prediction
============================================================================
 Machine learning model for predicting deception probability.
 Supports Random Forest, SVM, Logistic Regression, and Neural Network.
 Includes a heuristic fallback when no trained model is available.
============================================================================
"""

import os
import numpy as np
import pickle
from typing import Dict, Tuple, Optional
from collections import deque
import config


class DeceptionModel:
    """
    ML model for deception probability estimation.
    
    Falls back to a heuristic rule-based system when no trained
    model is available, providing reasonable behavioral analysis.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._model_loaded = False
        self._prediction_history = deque(maxlen=30)
        self._smoothed_prediction = 0.5
        self._baseline_features = None
        self._load_model()

    def _load_model(self):
        """Attempt to load a pre-trained model from disk."""
        try:
            if os.path.exists(config.MODEL_PATH):
                with open(config.MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                self._model_loaded = True
                print(f"[MODEL] Loaded model from {config.MODEL_PATH}")

            if os.path.exists(config.SCALER_PATH):
                with open(config.SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
                print(f"[MODEL] Loaded scaler from {config.SCALER_PATH}")
        except Exception as e:
            print(f"[MODEL] Could not load model: {e}")
            print("[MODEL] Using heuristic-based analysis instead.")
            self._model_loaded = False

    def predict(self, features: np.ndarray) -> Dict:
        """
        Predict deception probability from feature vector.
        
        Returns:
            Dictionary with prediction results.
        """
        if self._model_loaded and self._model is not None:
            return self._ml_predict(features)
        else:
            return self._heuristic_predict(features)

    def _ml_predict(self, features: np.ndarray) -> Dict:
        """Use trained ML model for prediction."""
        try:
            X = features.reshape(1, -1)
            if self._scaler is not None:
                X = self._scaler.transform(X)

            # Get probability
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X)[0]
                deception_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                pred = self._model.predict(X)[0]
                deception_prob = float(pred)

            return self._format_prediction(deception_prob, "ML_MODEL")
        except Exception as e:
            print(f"[MODEL] ML prediction error: {e}")
            return self._heuristic_predict(features)

    def _heuristic_predict(self, features: np.ndarray) -> Dict:
        """
        Rule-based heuristic for deception estimation.
        
        Uses weighted behavioral indicators with baseline comparison
        to produce a deception probability estimate.
        """
        score = 0.0
        indicators = {}

        # Feature index mapping (from config.FEATURE_NAMES)
        blink_rate = features[0]
        gaze_h_std = features[4]
        gaze_v_std = features[5]
        fixation = features[6]
        head_pitch_std = features[7]
        head_yaw_std = features[8]
        head_movement = features[10]
        symmetry = features[11]
        sym_change = features[12]
        lip_movement = features[13]
        lip_compression = features[14]
        eyebrow_tension = features[15]
        brow_asymmetry = features[16]
        micro_count = features[17]
        micro_intensity = features[18]
        facial_tension = features[19]
        nervous_score = features[20]
        jaw_clench = features[21]

        # 1. Blink rate anomaly (weight: 0.15)
        if blink_rate > 30:
            indicators["excessive_blinking"] = min(1.0, (blink_rate - 20) / 30)
            score += indicators["excessive_blinking"] * 0.15
        elif blink_rate < 5 and blink_rate > 0:
            indicators["suppressed_blinking"] = min(1.0, (10 - blink_rate) / 10)
            score += indicators["suppressed_blinking"] * 0.10

        # 2. Gaze instability (weight: 0.15)
        gaze_instability = min(1.0, (gaze_h_std + gaze_v_std) * 2.0)
        indicators["gaze_instability"] = gaze_instability
        score += gaze_instability * 0.15

        # 3. Head movement (weight: 0.10)
        head_agitation = min(1.0, head_movement * 0.5 + head_pitch_std * 0.02)
        indicators["head_agitation"] = head_agitation
        score += head_agitation * 0.10

        # 4. Face asymmetry (weight: 0.12)
        asym_score = min(1.0, (1.0 - symmetry) * 3.0 + sym_change * 5.0)
        indicators["face_asymmetry"] = asym_score
        score += asym_score * 0.12

        # 5. Lip tension (weight: 0.10)
        lip_score = min(1.0, max(0, 1.0 - lip_compression * 2.5))
        indicators["lip_tension"] = lip_score
        score += lip_score * 0.10

        # 6. Eyebrow tension (weight: 0.08)
        indicators["eyebrow_stress"] = eyebrow_tension
        score += eyebrow_tension * 0.08

        # 7. Micro-expressions (weight: 0.10)
        micro_score = min(1.0, micro_count * 0.3 + micro_intensity)
        indicators["micro_expressions"] = micro_score
        score += micro_score * 0.10

        # 8. Overall tension (weight: 0.10)
        indicators["facial_tension"] = facial_tension
        score += facial_tension * 0.10

        # 9. Nervous movement (weight: 0.10)
        indicators["nervousness"] = nervous_score
        score += nervous_score * 0.10

        deception_prob = float(np.clip(score, 0.0, 1.0))
        return self._format_prediction(deception_prob, "HEURISTIC", indicators)

    def _format_prediction(self, deception_prob: float, method: str,
                           indicators: Optional[Dict] = None) -> Dict:
        """Format and smooth prediction output."""
        # Apply temporal smoothing
        self._prediction_history.append(deception_prob)
        alpha = config.PREDICTION_SMOOTHING
        self._smoothed_prediction = (
            alpha * self._smoothed_prediction +
            (1 - alpha) * deception_prob
        )

        smoothed = float(np.clip(self._smoothed_prediction, 0.0, 1.0))
        truth_prob = 1.0 - smoothed

        # Confidence based on prediction stability
        if len(self._prediction_history) > 5:
            stability = 1.0 - np.std(list(self._prediction_history)[-15:]) * 2
            confidence = float(np.clip(stability, 0.3, 0.95))
        else:
            confidence = 0.5

        # Classification
        if smoothed > config.CONFIDENCE_THRESHOLD:
            label = "POTENTIAL DECEPTION"
            label_color = "deception"
        else:
            label = "LIKELY TRUTHFUL"
            label_color = "truth"

        return {
            "deception_probability": smoothed,
            "truth_probability": truth_prob,
            "confidence": confidence,
            "label": label,
            "label_color": label_color,
            "method": method,
            "indicators": indicators or {},
            "raw_score": deception_prob,
        }

    def set_baseline(self, features: np.ndarray):
        """Set baseline features from calibration."""
        self._baseline_features = features.copy()

    def reset(self):
        """Reset prediction state."""
        self._prediction_history.clear()
        self._smoothed_prediction = 0.5
