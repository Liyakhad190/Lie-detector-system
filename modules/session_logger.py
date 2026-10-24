"""
============================================================================
 SESSION LOGGER MODULE
============================================================================
 Logs session data to CSV files for post-session analysis.
============================================================================
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional
import config


class SessionLogger:
    """Logs behavioral analysis data to CSV files."""

    def __init__(self):
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file: Optional[str] = None
        self._writer = None
        self._file_handle = None
        self._entries: List[Dict] = []
        self._last_log_time = 0.0
        self._headers_written = False

    def start_session(self):
        """Start a new logging session."""
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._log_file = os.path.join(
            config.LOG_DIR, f"session_{self._session_id}.csv"
        )
        print(f"[LOGGER] Session log: {self._log_file}")

    def log(self, timestamp: float, face_id: int,
            blink_data: Dict, movement_data: Dict,
            micro_data: Dict, stress_data: Dict,
            prediction: Dict, question: str = ""):
        """Log a single frame's analysis data."""
        # Rate-limit logging
        if time.time() - self._last_log_time < config.LOG_INTERVAL:
            return
        self._last_log_time = time.time()

        band = prediction.get("prediction_stability_band") or prediction.get("confidence_interval", (0.0, 0.0))
        entry = {
            "timestamp": f"{timestamp:.3f}",
            "face_id": face_id,
            "blink_rate": f"{blink_data.get('blink_rate_per_min', 0):.1f}",
            "avg_ear": f"{blink_data.get('avg_ear', 0):.4f}",
            "total_blinks": blink_data.get("total_blinks", 0),
            "head_pitch": f"{movement_data.get('head_pitch', 0):.2f}",
            "head_yaw": f"{movement_data.get('head_yaw', 0):.2f}",
            "head_roll": f"{movement_data.get('head_roll', 0):.2f}",
            "gaze_h": f"{movement_data.get('gaze_horizontal', 0):.3f}",
            "gaze_v": f"{movement_data.get('gaze_vertical', 0):.3f}",
            "head_movement": f"{movement_data.get('movement_magnitude', 0):.3f}",
            "face_symmetry": f"{micro_data.get('face_symmetry', 0):.3f}",
            "lip_movement": f"{micro_data.get('lip_movement', 0):.2f}",
            "lip_compression": f"{micro_data.get('lip_compression', 0):.3f}",
            "eyebrow_tension": f"{micro_data.get('eyebrow_tension', 0):.3f}",
            "facial_tension": f"{micro_data.get('facial_tension', 0):.3f}",
            "micro_expressions": micro_data.get("micro_expression_count", 0),
            "stress_score": f"{stress_data.get('stress_score', 0):.3f}",
            "stress_level": stress_data.get("stress_level", "N/A"),
            "anomaly_score": f"{prediction.get('deception_probability', 0):.3f}",
            "stability_score": f"{prediction.get('confidence', 0):.3f}",
            "stability_band_lower": f"{band[0]:.3f}",
            "stability_band_upper": f"{band[1]:.3f}",
            "deception_prob": f"{prediction.get('deception_probability', 0):.3f}",
            "truth_prob": f"{prediction.get('truth_probability', 0):.3f}",
            "confidence": f"{prediction.get('confidence', 0):.3f}",
            "label": prediction.get("label", "N/A"),
            "question": question,
        }
        self._entries.append(entry)

    def save(self, protocol_summary: Optional[Dict] = None, baseline: Optional[Dict] = None) -> str:
        """Save all logged entries to CSV file and generate Markdown/JSON reports."""
        if not self._entries:
            print("[LOGGER] No entries to save.")
            return ""

        if self._log_file is None:
            self.start_session()

        try:
            with open(self._log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._entries[0].keys())
                writer.writeheader()
                writer.writerows(self._entries)
            print(f"[LOGGER] Saved {len(self._entries)} entries to {self._log_file}")

            # Generate Markdown and JSON Audit Reports
            from modules.report_generator import SessionReportGenerator
            generator = SessionReportGenerator()
            generator.generate_report(
                session_id=self._session_id,
                log_entries=self._entries,
                protocol_summary=protocol_summary,
                baseline=baseline
            )
            return self._log_file
        except Exception as e:
            print(f"[LOGGER] Error saving: {e}")
            return ""

    def get_entries(self) -> List[Dict]:
        """Get all raw logged entries."""
        return self._entries

    def get_entry_count(self) -> int:
        """Get number of logged entries."""
        return len(self._entries)

    def close(self, protocol_summary: Optional[Dict] = None, baseline: Optional[Dict] = None):
        """Save and close the session."""
        self.save(protocol_summary=protocol_summary, baseline=baseline)
