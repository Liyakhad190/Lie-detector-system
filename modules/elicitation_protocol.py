"""
============================================================================
 ELICITATION PROTOCOL MODULE - Structured Investigative Protocols
============================================================================
 Implements structured questioning protocols adapted from psychological
 research and conversational deception elicitation frameworks:
 
 Protocols:
   1. STANDARD_POLYGRAPH:
      - Baseline calibration questions (Neutral factual anchors)
      - Control questions (Emotional baseline)
      - Critical target questions (Investigative core)
      
   2. ALIBI_INVESTIGATION (Adapted from Reference Alibi Game):
      - Whereabouts declaration
      - Witness corroboration inquiry
      - Unexpected peripheral detail probe
      - Reverse-chronology recall (Cognitive load induction)
      
   3. COGNITIVE_LOAD_PROTOCOL:
      - High cognitive demand questioning designed to amplify micro-behavioral
        cues by depleting executive cognitive bandwidth.
============================================================================
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum


class QuestionCategory(Enum):
    BASELINE = "BASELINE"
    NEUTRAL = "NEUTRAL"
    CONTROL = "CONTROL"
    CRITICAL = "CRITICAL"
    COGNITIVE_LOAD = "COGNITIVE_LOAD"


class QuestionState(Enum):
    PENDING = "PENDING"
    RECORDING = "RECORDING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"


class ElicitationQuestion:
    """Individual question within an elicitation protocol."""

    def __init__(self, qid: int, text: str, category: QuestionCategory, target_intent: str):
        self.qid = qid
        self.text = text
        self.category = category
        self.target_intent = target_intent
        self.state = QuestionState.PENDING
        
        # Timing metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.response_latency: float = 0.0
        
        # Aggregated behavioral features during this question's answer window
        self.buffer_predictions: List[Dict[str, Any]] = []
        self.summary_metrics: Optional[Dict[str, Any]] = None

    def start_recording(self):
        self.state = QuestionState.RECORDING
        self.start_time = time.time()
        self.buffer_predictions.clear()

    def record_frame(self, prediction_dict: Dict[str, Any]):
        if self.state == QuestionState.RECORDING and prediction_dict:
            self.buffer_predictions.append(prediction_dict)

    def finish_recording(self) -> Dict[str, Any]:
        self.state = QuestionState.COMPLETED
        self.end_time = time.time()
        duration = self.end_time - (self.start_time or self.end_time)

        if not self.buffer_predictions:
            self.summary_metrics = {
                "qid": self.qid,
                "text": self.text,
                "category": self.category.value,
                "duration": duration,
                "sample_count": 0,
                "mean_deception_prob": 0.0,
                "mean_stress_score": 0.0,
                "mean_ocular": 0.0,
                "mean_tension": 0.0,
                "mean_micro": 0.0,
                "verdict": "INSUFFICIENT_DATA",
                "risk_level": "UNKNOWN"
            }
            return self.summary_metrics

        probs = [p.get("deception_probability", 0.0) for p in self.buffer_predictions]
        raw_scores = [p.get("raw_score", 0.0) for p in self.buffer_predictions]
        
        mean_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        std_prob = float(np.std(probs))

        # Channel breakdown if explainability was present
        ocular_vals = []
        tension_vals = []
        micro_vals = []
        for p in self.buffer_predictions:
            exp = p.get("explainability", {})
            channels = exp.get("channels", {})
            if "ocular" in channels:
                ocular_vals.append(channels["ocular"].get("score", 0.0))
            if "myofacial" in channels:
                tension_vals.append(channels["myofacial"].get("score", 0.0))
            if "micro_dynamics" in channels:
                micro_vals.append(channels["micro_dynamics"].get("score", 0.0))

        mean_ocular = float(np.mean(ocular_vals)) if ocular_vals else 0.0
        mean_tension = float(np.mean(tension_vals)) if tension_vals else 0.0
        mean_micro = float(np.mean(micro_vals)) if micro_vals else 0.0

        if mean_prob > 0.65:
            verdict = "ELEVATED_ANOMALY"
            risk = "HIGH"
        elif mean_prob > 0.45:
            verdict = "MODERATE_REACTION"
            risk = "MEDIUM"
        else:
            verdict = "BASELINE_CONSISTENT"
            risk = "LOW"

        self.summary_metrics = {
            "qid": self.qid,
            "text": self.text,
            "category": self.category.value,
            "target_intent": self.target_intent,
            "duration": round(duration, 2),
            "sample_count": len(self.buffer_predictions),
            "mean_deception_prob": round(mean_prob, 3),
            "max_deception_prob": round(max_prob, 3),
            "std_deception_prob": round(std_prob, 3),
            "mean_ocular": round(mean_ocular, 3),
            "mean_tension": round(mean_tension, 3),
            "mean_micro": round(mean_micro, 3),
            "verdict": verdict,
            "risk_level": risk
        }
        return self.summary_metrics


class ElicitationProtocolManager:
    """Manages protocol execution, active question tracking, and multi-question summaries."""

    PROTOCOLS = {
        "STANDARD_POLYGRAPH": [
            ("State your full legal name and current occupation.", QuestionCategory.BASELINE, "Ground-truth factual baseline"),
            ("Did you eat breakfast today? Describe it briefly.", QuestionCategory.NEUTRAL, "Neutral factual recall"),
            ("Have you ever taken something that did not belong to you in your life?", QuestionCategory.CONTROL, "General emotional reaction control"),
            ("Did you directly participate in the incident under inquiry?", QuestionCategory.CRITICAL, "Primary critical target inquiry"),
            ("Do you know who is responsible for the incident?", QuestionCategory.CRITICAL, "Knowledge-detection probe"),
            ("Is there anything you are intentionally withholding from this interview?", QuestionCategory.CRITICAL, "Secondary withholding check")
        ],
        "ALIBI_INVESTIGATION": [
            ("State your name and confirm you are ready to answer truthfully.", QuestionCategory.BASELINE, "Identity & engagement baseline"),
            ("Where were you yesterday evening between 8:00 PM and 10:00 PM?", QuestionCategory.CRITICAL, "Primary alibi declaration"),
            ("Was anyone present with you who can independently corroborate this?", QuestionCategory.CRITICAL, "Corroboration probe"),
            ("What was the weather like outside, or what specific clothes were you wearing?", QuestionCategory.CONTROL, "Peripheral detail check"),
            ("Describe the events of yesterday evening backwards, from 10:00 PM down to 8:00 PM.", QuestionCategory.COGNITIVE_LOAD, "Reverse-order cognitive load probe")
        ],
        "COGNITIVE_LOAD_PROTOCOL": [
            ("Count backwards from 100 by 7s while looking straight at the camera.", QuestionCategory.COGNITIVE_LOAD, "Mathematical dual-task load"),
            ("Describe your daily routine starting from bedtime back to waking up.", QuestionCategory.COGNITIVE_LOAD, "Reverse narrative sequencing"),
            ("Explain the rules of your favorite sport or hobby in detail.", QuestionCategory.CONTROL, "Fluent recall comparison")
        ]
    }

    def __init__(self, default_protocol: str = "ALIBI_INVESTIGATION"):
        self.protocol_name = default_protocol
        self.questions: List[ElicitationQuestion] = []
        self.current_idx: int = 0
        self.is_active: bool = False
        self.load_protocol(default_protocol)

    def load_protocol(self, protocol_name: str):
        """Load a question set for the selected protocol."""
        if protocol_name not in self.PROTOCOLS:
            protocol_name = "ALIBI_INVESTIGATION"
        self.protocol_name = protocol_name
        self.questions = [
            ElicitationQuestion(i + 1, text, cat, intent)
            for i, (text, cat, intent) in enumerate(self.PROTOCOLS[protocol_name])
        ]
        self.current_idx = 0
        self.is_active = False

    def start_protocol(self):
        self.current_idx = 0
        self.is_active = True
        if self.questions:
            self.questions[0].start_recording()

    def get_current_question(self) -> Optional[ElicitationQuestion]:
        if not self.is_active or self.current_idx >= len(self.questions):
            return None
        return self.questions[self.current_idx]

    def advance_question(self) -> Optional[Dict[str, Any]]:
        """Complete current question and advance to next."""
        curr = self.get_current_question()
        summary = None
        if curr:
            summary = curr.finish_recording()
            self.current_idx += 1
            if self.current_idx < len(self.questions):
                self.questions[self.current_idx].start_recording()
            else:
                self.is_active = False # Protocol completed
        return summary

    def record_frame(self, prediction_dict: Dict[str, Any]):
        curr = self.get_current_question()
        if curr:
            curr.record_frame(prediction_dict)

    def get_protocol_summary(self) -> Dict[str, Any]:
        """Generate complete protocol summary across all questions."""
        q_summaries = []
        critical_probs = []
        baseline_probs = []
        for q in self.questions:
            s = q.summary_metrics or (q.finish_recording() if q.state == QuestionState.RECORDING else None)
            if s:
                q_summaries.append(s)
                if q.category in (QuestionCategory.CRITICAL, QuestionCategory.COGNITIVE_LOAD):
                    critical_probs.append(s["mean_deception_prob"])
                elif q.category == QuestionCategory.BASELINE:
                    baseline_probs.append(s["mean_deception_prob"])

        mean_critical = float(np.mean(critical_probs)) if critical_probs else 0.0
        mean_baseline = float(np.mean(baseline_probs)) if baseline_probs else 0.0
        delta = mean_critical - mean_baseline

        if delta > 0.30 or mean_critical > 0.65:
            overall_verdict = "ELEVATED BEHAVIORAL REACTIVITY TO CRITICAL PROBES"
            verdict_color = "deception"
        elif delta > 0.15:
            overall_verdict = "MODERATE REACTIVITY DETECTED"
            verdict_color = "warning"
        else:
            overall_verdict = "NO SIGNIFICANT DIFFERENTIAL REACTIVITY OBSERVED"
            verdict_color = "truth"

        return {
            "protocol_name": self.protocol_name,
            "total_questions": len(self.questions),
            "completed_questions": len(q_summaries),
            "mean_critical_score": round(mean_critical, 3),
            "mean_baseline_score": round(mean_baseline, 3),
            "probe_reactivity_delta": round(delta, 3),
            "overall_verdict": overall_verdict,
            "verdict_color": verdict_color,
            "question_summaries": q_summaries
        }
