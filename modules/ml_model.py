"""
============================================================================
 ML MODEL MODULE - Calibrated Behavioral Anomaly & Deception Estimation
============================================================================
 Machine learning model for estimating behavioral anomaly probability.
 Supports Random Forest, SVM, Logistic Regression, and Neural Network.
 Integrates Explainability Decomposition and Autonomic (rPPG) inputs.
 Includes a heuristic fallback when no trained model is available.
============================================================================
"""

import os
import numpy as np
import pickle
from typing import Dict, Tuple, Optional, Any
from collections import deque
import config
from modules.explainability import ExplainabilityDecomposer


class DeceptionModel:
    """
    ML model for behavioral anomaly and deception probability estimation.
    
    Provides calibrated probabilities, confidence bounds, multi-channel
    explainability breakdown, and graceful fallback heuristics.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._model_loaded = False
        self._prediction_history = deque(maxlen=30)
        self._smoothed_prediction = 0.5
        self._baseline_features: Optional[np.ndarray] = None
        self._decomposer = ExplainabilityDecomposer()
        self.metadata: Dict[str, Any] = {}
        self.provenance: str = "UNKNOWN"
        self.model_category: str = "UNKNOWN"
        self.label_verified: bool = False
        self.label_protocol: str = "UNKNOWN"
        self.intended_interpretation: str = ""
        self.limitations: list = []
        self._load_model()

    def _load_model(self):
        """Attempt to load a pre-trained model and scaler from disk."""
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

            metadata_path = "models/model_metadata.json"
            if os.path.exists(metadata_path):
                import json
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f_meta:
                        self.metadata = json.load(f_meta)
                        self.provenance = self.metadata.get("data_provenance", "UNKNOWN")
                        self.model_category = self.metadata.get(
                            "model_category",
                            "SYNTHETIC_BENCHMARK" if self.provenance == "SYNTHETIC" else "UNKNOWN"
                        )
                        self.label_verified = self.metadata.get("label_verified", False)
                        self.label_protocol = self.metadata.get("label_protocol", "UNKNOWN")
                        self.intended_interpretation = self.metadata.get("intended_interpretation", "")
                        self.limitations = self.metadata.get("limitations", [])

                        if self.provenance == "SYNTHETIC":
                            print("[MODEL] Checkpoint provenance: SYNTHETIC (Trained on simulated distribution; not validated on human deception).")
                        elif not self.label_verified:
                            print(f"[MODEL] Checkpoint category: {self.model_category} (Task-directed instructions; label_verified=False).")
                except Exception:
                    self.metadata = {}
                    self.provenance = "UNKNOWN"
                    self.model_category = "UNKNOWN"
                    self.label_verified = False
                    self.label_protocol = "UNKNOWN"
                    self.intended_interpretation = ""
                    self.limitations = []
            else:
                self.metadata = {}
                self.provenance = "UNKNOWN"
                self.model_category = "UNKNOWN"
                self.label_verified = False
                self.label_protocol = "UNKNOWN"
                self.intended_interpretation = ""
                self.limitations = []
        except Exception as e:
            print(f"[MODEL] Could not load model: {e}")
            print("[MODEL] Using heuristic-based analysis instead.")
            self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded and self._model is not None

    def predict(self, features: np.ndarray, polygraph_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Predict deception/behavioral anomaly probability from feature vector.
        
        Args:
            features: 22-dimensional feature vector.
            polygraph_data: Optional physiological dict (rPPG HR, stress signal).
            
        Returns:
            Dictionary containing calibrated probability, confidence bounds,
            evidence decomposition, and scientific labels.
        """
        if features is None or len(features) < len(config.FEATURE_NAMES):
            return self._empty_prediction()

        # Run Explainability Decomposition across 4 channels
        explainability = self._decomposer.decompose(
            features, polygraph_data=polygraph_data, baseline=self._baseline_features
        )

        if self._model_loaded and self._model is not None:
            return self._ml_predict(features, explainability)
        else:
            return self._heuristic_predict(features, explainability)

    def _ml_predict(self, features: np.ndarray, explainability: Dict[str, Any]) -> Dict[str, Any]:
        """Use trained ML model for prediction with explainability fusion."""
        try:
            X = features.reshape(1, -1)
            if self._scaler is not None:
                X = self._scaler.transform(X)

            # Get raw probability
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X)[0]
                deception_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                pred = self._model.predict(X)[0]
                deception_prob = float(pred)

            # Fuse slightly with explainability composite to stabilize edge anomalies
            composite_decomp = explainability.get("composite_anomaly_score", deception_prob)
            fused_prob = float(0.70 * deception_prob + 0.30 * composite_decomp)

            return self._format_prediction(fused_prob, "ML_MODEL", explainability)
        except Exception as e:
            print(f"[MODEL] ML prediction error: {e}")
            return self._heuristic_predict(features, explainability)

    def _heuristic_predict(self, features: np.ndarray, explainability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based heuristic for deception estimation when no trained ML weights exist.
        Directly driven by explainability channel weights.
        """
        composite_score = explainability.get("composite_anomaly_score", 0.0)
        deception_prob = float(np.clip(composite_score, 0.0, 1.0))
        return self._format_prediction(deception_prob, "HEURISTIC", explainability)

    def _format_prediction(self, deception_prob: float, method: str,
                           explainability: Dict[str, Any]) -> Dict[str, Any]:
        """Format, smooth, and attach prediction stability band and scientific diagnostics."""
        # Apply temporal smoothing
        self._prediction_history.append(deception_prob)
        alpha = config.PREDICTION_SMOOTHING
        self._smoothed_prediction = (
            alpha * self._smoothed_prediction +
            (1.0 - alpha) * deception_prob
        )

        smoothed = float(np.clip(self._smoothed_prediction, 0.0, 1.0))
        truth_prob = 1.0 - smoothed

        # Temporal prediction stability: measures how consistent the last 15 frames
        # of model output are. This is NOT a statistical confidence interval for
        # deception probability — it reflects signal stability over time only.
        if len(self._prediction_history) > 5:
            hist = list(self._prediction_history)[-15:]
            std_dev = float(np.std(hist))
            stability = 1.0 - std_dev * 2.0
            confidence = float(np.clip(stability, 0.30, 0.95))
            band_lower = max(0.0, smoothed - 1.96 * std_dev)
            band_upper = min(1.0, smoothed + 1.96 * std_dev)
        else:
            confidence = 0.50
            band_lower = max(0.0, smoothed - 0.15)
            band_upper = min(1.0, smoothed + 0.15)

        # Scientifically responsible classification labels
        if smoothed > 0.65:
            label = "HIGH BEHAVIORAL ANOMALY"
            label_color = "deception"
            risk_tier = "HIGH"
        elif smoothed > 0.40:
            label = "MODERATE BEHAVIORAL STRESS"
            label_color = "warning"
            risk_tier = "MODERATE"
        else:
            label = "BASELINE CONSISTENT"
            label_color = "truth"
            risk_tier = "LOW"

        _band = (round(band_lower, 3), round(band_upper, 3))
        return {
            "deception_probability": round(smoothed, 3),
            "truth_probability": round(truth_prob, 3),
            "confidence": round(confidence, 3),           # temporal signal stability [0.30–0.95]
            # prediction_stability_band: range of model scores over the last 15 frames.
            # This measures temporal consistency, NOT a statistical CI for deception probability.
            "prediction_stability_band": _band,
            "confidence_interval": _band,                 # backward-compat alias (deprecated)
            "label": label,
            "label_color": label_color,
            "risk_tier": risk_tier,
            "method": method,
            "explainability": explainability,
            "raw_score": round(deception_prob, 3),
            "data_provenance": self.provenance,
            "model_category": self.model_category,
            "label_verified": self.label_verified,
            "label_protocol": self.label_protocol,
            "scientific_disclaimer": config.DISCLAIMER
        }

    def _empty_prediction(self) -> Dict[str, Any]:
        return {
            "deception_probability": 0.0,
            "truth_probability": 1.0,
            "confidence": 0.0,
            "prediction_stability_band": (0.0, 0.0),
            "confidence_interval": (0.0, 0.0),           # backward-compat alias (deprecated)
            "label": "NO DATA",
            "label_color": "warning",
            "risk_tier": "UNKNOWN",
            "method": "NONE",
            "explainability": self._decomposer._empty_decomposition(),
            "raw_score": 0.0,
            "data_provenance": self.provenance,
            "model_category": self.model_category,
            "label_verified": self.label_verified,
            "label_protocol": self.label_protocol,
            "scientific_disclaimer": config.DISCLAIMER
        }

    def set_baseline(self, features: np.ndarray):
        """Set baseline features from calibration."""
        if features is not None:
            self._baseline_features = features.copy()

    def reset(self):
        """Reset prediction state."""
        self._prediction_history.clear()
        self._smoothed_prediction = 0.5
