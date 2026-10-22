"""
============================================================================
 EXPLAINABILITY MODULE - Behavioral Anomaly Decomposition
============================================================================
 Decomposes composite deception/stress predictions into interpretable
 behavioral and physiological channels with evidence attribution:
 
 1. Ocular Channel (Blink rate, Eye Aspect Ratio, Gaze deviation, Fixation)
 2. Myofacial Tension Channel (Brow tension, Lip compression, Jaw clench)
 3. Micro-Dynamics Channel (Micro-expressions, Facial asymmetry changes)
 4. Autonomic / Physiological Channel (Forehead rPPG pulse rate & variability)
============================================================================
"""

import numpy as np
from typing import Dict, List, Any, Optional
import config


class ExplainabilityDecomposer:
    """
    Decomposes multi-modal feature vectors and predictions into
    interpretable diagnostic channels with attribution weights.
    """

    CHANNEL_WEIGHTS = {
        "ocular": 0.30,
        "myofacial": 0.30,
        "micro_dynamics": 0.25,
        "autonomic": 0.15
    }

    def __init__(self):
        self.feature_names = config.FEATURE_NAMES

    def decompose(self, features: np.ndarray,
                  polygraph_data: Optional[Dict] = None,
                  baseline: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Decompose a 22-dimensional feature vector into four channel scores [0.0 - 1.0]
        and generate evidence attribution explanations.
        """
        if features is None or len(features) < 22:
            return self._empty_decomposition()

        # Map features by index
        blink_rate = float(features[0])
        blink_rate_change = float(features[1])
        ear_mean = float(features[2])
        ear_std = float(features[3])
        gaze_h_std = float(features[4])
        gaze_v_std = float(features[5])
        gaze_fixation = float(features[6])
        head_pitch_std = float(features[7])
        head_yaw_std = float(features[8])
        head_roll_std = float(features[9])
        head_movement = float(features[10])
        face_sym_score = float(features[11])
        face_sym_change = float(features[12])
        lip_movement = float(features[13])
        lip_compression = float(features[14])
        eyebrow_tension = float(features[15])
        eyebrow_asym = float(features[16])
        micro_count = float(features[17])
        micro_intensity = float(features[18])
        facial_tension = float(features[19])
        nervous_score = float(features[20])
        jaw_clench = float(features[21])

        # 1. OCULAR CHANNEL
        # Elevated or suppressed blinking + gaze aversion + fixation drop
        blink_anomaly = 0.0
        if blink_rate > 28:
            blink_anomaly = min(1.0, (blink_rate - 20) / 25)
        elif 0 < blink_rate < 8:
            blink_anomaly = min(1.0, (10 - blink_rate) / 10)
        
        gaze_instability = min(1.0, (gaze_h_std + gaze_v_std) * 2.5)
        fixation_drop = max(0.0, 1.0 - gaze_fixation)
        ocular_score = float(np.clip(
            0.4 * blink_anomaly + 0.35 * gaze_instability + 0.25 * fixation_drop,
            0.0, 1.0
        ))

        # 2. MYOFACIAL TENSION CHANNEL
        # Eyebrow furrowing, lip compression, jaw clench, global facial tension
        lip_tension = min(1.0, max(0.0, 1.0 - lip_compression * 2.5))
        myofacial_score = float(np.clip(
            0.30 * eyebrow_tension + 0.25 * lip_tension + 
            0.25 * jaw_clench * 2.0 + 0.20 * facial_tension,
            0.0, 1.0
        ))

        # 3. MICRO-DYNAMICS CHANNEL
        # Asymmetry spikes, micro-expressions, rapid head agitation
        asym_anomaly = min(1.0, (1.0 - face_sym_score) * 2.5 + face_sym_change * 4.0)
        micro_burst = min(1.0, micro_count * 0.4 + micro_intensity * 1.5)
        head_fidget = min(1.0, head_movement * 0.4 + (head_pitch_std + head_yaw_std) * 0.05)
        micro_score = float(np.clip(
            0.40 * asym_anomaly + 0.35 * micro_burst + 0.25 * head_fidget,
            0.0, 1.0
        ))

        # 4. AUTONOMIC / rPPG CHANNEL
        # Forehead optical pulse rate change and stress signal
        autonomic_score = 0.2 # default baseline
        hr_bpm = 75.0
        if polygraph_data and isinstance(polygraph_data, dict):
            hr_bpm = polygraph_data.get("estimated_hr", 75.0)
            if hr_bpm > 85:
                autonomic_score = min(1.0, (hr_bpm - 70) / 50.0)
            elif hr_bpm < 55:
                autonomic_score = min(1.0, (60 - hr_bpm) / 20.0)
            else:
                autonomic_score = 0.15

        # Weighted composite anomaly
        composite_score = float(np.clip(
            self.CHANNEL_WEIGHTS["ocular"] * ocular_score +
            self.CHANNEL_WEIGHTS["myofacial"] * myofacial_score +
            self.CHANNEL_WEIGHTS["micro_dynamics"] * micro_score +
            self.CHANNEL_WEIGHTS["autonomic"] * autonomic_score,
            0.0, 1.0
        ))

        # Key evidence factors
        evidence_factors = []
        if ocular_score > 0.45:
            detail = f"Gaze instability ({gaze_instability:.2f}) & blink anomaly ({blink_rate:.1f} bpm)"
            evidence_factors.append({"channel": "Ocular", "score": ocular_score, "evidence": detail})
        if myofacial_score > 0.45:
            detail = f"Elevated brow tension ({eyebrow_tension:.2f}) & jaw clench indicator ({jaw_clench:.2f})"
            evidence_factors.append({"channel": "Myofacial Tension", "score": myofacial_score, "evidence": detail})
        if micro_score > 0.45:
            detail = f"Facial asymmetry change ({face_sym_change:.3f}) & micro-movements ({micro_count:.1f}/sec)"
            evidence_factors.append({"channel": "Micro-Dynamics", "score": micro_score, "evidence": detail})
        if autonomic_score > 0.50:
            detail = f"Autonomic cardiovascular pulse elevation ({hr_bpm:.1f} BPM)"
            evidence_factors.append({"channel": "Autonomic / rPPG", "score": autonomic_score, "evidence": detail})

        # Primary contributing channel
        channel_scores = {
            "Ocular": ocular_score,
            "Myofacial Tension": myofacial_score,
            "Micro-Dynamics": micro_score,
            "Autonomic Pulse": autonomic_score
        }
        primary_channel = max(channel_scores.items(), key=lambda x: x[1])

        return {
            "composite_anomaly_score": composite_score,
            "channels": {
                "ocular": {
                    "score": ocular_score,
                    "weight": self.CHANNEL_WEIGHTS["ocular"],
                    "metrics": {
                        "blink_rate": blink_rate,
                        "gaze_h_std": gaze_h_std,
                        "gaze_v_std": gaze_v_std,
                        "fixation_ratio": gaze_fixation
                    }
                },
                "myofacial": {
                    "score": myofacial_score,
                    "weight": self.CHANNEL_WEIGHTS["myofacial"],
                    "metrics": {
                        "eyebrow_tension": eyebrow_tension,
                        "lip_compression": lip_compression,
                        "jaw_clench": jaw_clench,
                        "facial_tension": facial_tension
                    }
                },
                "micro_dynamics": {
                    "score": micro_score,
                    "weight": self.CHANNEL_WEIGHTS["micro_dynamics"],
                    "metrics": {
                        "symmetry_score": face_sym_score,
                        "symmetry_change": face_sym_change,
                        "micro_count": micro_count,
                        "nervous_score": nervous_score
                    }
                },
                "autonomic": {
                    "score": autonomic_score,
                    "weight": self.CHANNEL_WEIGHTS["autonomic"],
                    "metrics": {
                        "estimated_hr": hr_bpm
                    }
                }
            },
            "primary_contributor": primary_channel[0],
            "primary_channel_score": primary_channel[1],
            "evidence_factors": evidence_factors,
            "scientific_interpretation": self._generate_scientific_summary(composite_score, primary_channel[0], evidence_factors)
        }

    def _generate_scientific_summary(self, composite: float, primary: str, evidence: List[Dict]) -> str:
        """Generate objective, non-accusatory behavioral summary."""
        if composite < 0.35:
            return "Behavioral indicators remain within standard calibrated baseline parameters."
        elif composite < 0.60:
            return f"Moderate behavioral reactivity detected, primarily in the {primary} channel."
        else:
            factors_str = "; ".join([e["evidence"] for e in evidence]) if evidence else primary
            return f"Significant behavioral stress response observed: {factors_str}."

    def _empty_decomposition(self) -> Dict[str, Any]:
        return {
            "composite_anomaly_score": 0.0,
            "channels": {
                "ocular": {"score": 0.0, "weight": 0.30, "metrics": {}},
                "myofacial": {"score": 0.0, "weight": 0.30, "metrics": {}},
                "micro_dynamics": {"score": 0.0, "weight": 0.25, "metrics": {}},
                "autonomic": {"score": 0.0, "weight": 0.15, "metrics": {}}
            },
            "primary_contributor": "None",
            "primary_channel_score": 0.0,
            "evidence_factors": [],
            "scientific_interpretation": "No facial or biometric data available."
        }
