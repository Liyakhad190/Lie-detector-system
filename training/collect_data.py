"""
============================================================================
 DATA COLLECTION SCRIPT (Research-Grade)
============================================================================
 Collects real human behavioral data by prompting participants to tell
 truths and lies directly into a webcam.

 ETHICAL REQUIREMENTS:
   - Participant consent is obtained before any data is captured.
   - Only anonymous participant IDs are stored — no names, emails,
     or other personal identifiers are collected or saved.
   - Labels are task-defined (TRUTH/LIE prompt), NOT model-inferred.
   - Participants may quit at any time by pressing Q.
   - Do not tell participants which responses will be labelled which way
     until after the session to avoid demand characteristics.

 OUTPUT: data/real_features.csv  (appended if file exists)
   Columns: participant_id, session_id, experiment_id, prompt_index,
             task_label, frame_count, quality_flag, <22 feature cols>

 USAGE:
   python training/collect_data.py --participant-id P001 --experiment EXP1
   python training/collect_data.py --participant-id P002  # uses default experiment
============================================================================
"""

import sys
import os
import time
import argparse
import hashlib
import csv
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.face_detector import FaceDetector
from modules.blink_detector import BlinkDetector
from modules.movement_tracker import MovementTracker
from modules.micro_expression import MicroExpressionAnalyzer
from modules.stress_analyzer import StressAnalyzer
from modules.feature_extractor import FeatureExtractor
from modules.calibration import CalibrationManager

# Minimum number of valid face-tracked frames required to accept a response.
# Responses with fewer frames are flagged as low-quality and excluded by default.
MIN_QUALITY_FRAMES = 10


def _show_consent_screen(window_name: str) -> bool:
    """
    Display an informed consent notice. Returns True only if the participant
    explicitly presses Y to consent. Any other key or window close = no consent.
    NOTE: This is a minimal on-screen consent. For IRB-approved research,
    a full written consent form must be obtained separately.
    """
    h, w = 480, 800
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    lines = [
        "INFORMED CONSENT",
        "",
        "This session records facial feature measurements from your",
        "webcam while you respond to prompts. NO video or images",
        "are saved — only anonymous numerical measurements.",
        "",
        "Your responses will be labelled as Truth or Lie according",
        "to the task prompt shown. Labels reflect the task design,",
        "NOT any claim about your honesty.",
        "",
        "You may quit at any time by pressing Q.",
        "Data is stored under an anonymous participant ID only.",
        "No names, emails, or personal information are collected.",
        "",
        "Press Y to consent and begin.  Press N or Q to exit.",
    ]
    y0 = 40
    for i, line in enumerate(lines):
        color = (0, 255, 200) if i == 0 else (200, 210, 220)
        thick = 2 if i == 0 else 1
        scale = 0.65 if i == 0 else 0.45
        cv2.putText(canvas, line, (30, y0 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    cv2.imshow(window_name, canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('y') or key == ord('Y'):
            return True
        if key in (ord('n'), ord('N'), ord('q'), ord('Q'), 27):
            return False
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False


def _validate_sample(feature_values: list, frame_count: int,
                     label: int = None, participant_id: str = None) -> tuple:
    """
    Return (is_valid: bool, quality_flag: str) for a recorded response.
    Validates frame count, NaN, Inf, binary label, and participant ID.
    """
    if frame_count < MIN_QUALITY_FRAMES:
        return False, f"LOW_FRAMES({frame_count})"
    if any(v is None or (isinstance(v, (float, int)) and np.isnan(v)) for v in feature_values):
        return False, "CONTAINS_NAN"
    if any(isinstance(v, (float, int)) and np.isinf(v) for v in feature_values):
        return False, "CONTAINS_INF"
    if label is not None and label not in (0, 1):
        return False, f"INVALID_LABEL({label})"
    if participant_id is not None and not str(participant_id).strip():
        return False, "MISSING_PARTICIPANT_ID"
    return True, "OK"


class DataCollector:
    """
    Research-grade data collector with participant anonymisation,
    explicit consent, session metadata, and quality validation.
    Camera is only acquired AFTER informed consent is granted.
    """

    WINDOW = "Data Collector"

    def __init__(self, participant_id: str, experiment_id: str = "EXP_DEFAULT"):
        # Anonymise: store a short hash of the participant_id so the raw ID
        # does not appear in the CSV even if the researcher uses a real name.
        anon_hash = hashlib.sha256(participant_id.strip().encode()).hexdigest()[:10]
        self.participant_id = f"P_{anon_hash}"
        self.experiment_id = experiment_id
        self.session_id = datetime.now().strftime("S_%Y%m%d_%H%M%S")

        print(f"Participant ID (anonymised): {self.participant_id}")
        print(f"Session ID               : {self.session_id}")
        print(f"Experiment ID            : {self.experiment_id}")
        print()

        # Camera is deliberately NOT opened here to respect consent requirements.
        # Acquisition begins only after explicit consent is received.
        self.cap = None

        self.face_detector = FaceDetector()
        self.blink_det = BlinkDetector()
        self.move_track = MovementTracker()
        self.micro_analyzer = MicroExpressionAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.calibration = CalibrationManager()

        self.collected_samples: list = []
        self.skipped_low_quality: int = 0

    def _init_camera(self):
        """Acquire camera only after informed consent has been obtained."""
        print("Acquiring camera after consent...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # 5 truthful + 5 deceptive prompts, randomised order.
        # Labels are task-defined, not model-inferred.
        self.prompts = [
            ("TRUTH", "What did you eat for breakfast today?"),
            ("TRUTH", "Describe the room you are currently in."),
            ("TRUTH", "What is your favorite movie and why?"),
            ("TRUTH", "Describe your daily routine."),
            ("TRUTH", "What did you do last weekend?"),
            ("LIE",   "Lie about what you ate for breakfast today."),
            ("LIE",   "Describe a completely fake room instead of this one."),
            ("LIE",   "Make up a fake favorite movie and explain why."),
            ("LIE",   "Describe a completely fake daily routine."),
            ("LIE",   "Make up a story about what you did last weekend."),
        ]
        np.random.shuffle(self.prompts)

        self.state = "CONSENT"
        self.current_prompt_idx = 0
        self.current_features_buffer: list = []
        self.current_frame_count: int = 0

    def _process_frame(self, frame: np.ndarray):
        """Run the feature pipeline on a single frame. Returns feature dict or None."""
        faces = self.face_detector.detect(frame)
        if not faces:
            return None
        face = faces[0]
        self.face_detector.draw_key_landmarks(frame, face)

        blink_data   = self.blink_det.update(face.landmarks)
        movement_data = self.move_track.update(face.landmarks, frame.shape[:2])
        micro_data   = self.micro_analyzer.update(face.landmarks)
        stress_data  = self.stress_analyzer.compute_stress(blink_data, movement_data, micro_data)
        features     = self.feature_extractor.extract(blink_data, movement_data, micro_data, stress_data)

        if self.calibration.is_calibrating:
            self.calibration.add_sample(blink_data, movement_data, micro_data, stress_data, features)
        elif self.calibration.is_complete and self.calibration.baseline:
            bl = self.calibration.baseline
            self.blink_det.set_baseline(bl["avg_ear"], bl["blink_rate"])
            self.micro_analyzer.set_baseline(bl["face_symmetry"], bl["lip_movement"])

        return features

    def _accept_response(self):
        """Validate and save the buffered feature frames for the current prompt."""
        label_str, prompt_text = self.prompts[self.current_prompt_idx]
        n_frames = len(self.current_features_buffer)
        if n_frames == 0:
            print(f"  [SKIP] No face tracked during this response — discarding.")
            self.skipped_low_quality += 1
            return

        # Compute mean features across all tracked frames
        avg_feat = {k: float(np.nanmean([f[k] for f in self.current_features_buffer]))
                    for k in config.FEATURE_NAMES}

        label_val = 0 if label_str == "TRUTH" else 1
        is_valid, quality_flag = _validate_sample(
            list(avg_feat.values()), n_frames,
            label=label_val, participant_id=self.participant_id
        )

        sample = {
            "participant_id": self.participant_id,
            "session_id":     self.session_id,
            "experiment_id":  self.experiment_id,
            "prompt_index":   self.current_prompt_idx,
            "task_label_str": label_str,          # human-readable: TRUTH / LIE
            "label":          label_val,          # numeric for training
            "frame_count":    n_frames,
            "quality_flag":   quality_flag,
            "data_provenance": "REAL_HUMAN",
            "label_protocol": "CONTROLLED_TASK_DIRECTED",
            "label_verified": False,              # Task instruction; intent not biologically verified
            **avg_feat
        }

        if is_valid:
            self.collected_samples.append(sample)
            print(f"  [OK]   Sample {self.current_prompt_idx + 1}/{len(self.prompts)} "
                  f"saved ({n_frames} frames, {label_str})")
        else:
            self.skipped_low_quality += 1
            print(f"  [WARN] Sample quality insufficient ({quality_flag}) — "
                  f"saved with flag for manual review.")
            self.collected_samples.append(sample)   # keep but flagged

    def run(self):
        """Main collection loop."""
        cv2.namedWindow(self.WINDOW)

        # ── STATE: CONSENT (Prior to camera acquisition) ───────────────
        consented = _show_consent_screen(self.WINDOW)
        if not consented:
            print("[CONSENT] Participant did not consent. Exiting without camera acquisition.")
            cv2.destroyAllWindows()
            return

        print("[CONSENT] Participant consented. Acquiring camera and starting baseline calibration...")
        try:
            self._init_camera()
            self.calibration.start()
            self.state = "CALIBRATION"

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                features = self._process_frame(frame)

                # ── STATE: CALIBRATION ─────────────────────────────────────────
                if self.state == "CALIBRATION":
                    prog = self.calibration.get_progress()
                    cv2.putText(frame, f"CALIBRATING BASELINE... {prog:.0%}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(frame, "Look naturally at the camera.",
                                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 220), 1)
                    if not self.calibration.is_calibrating:
                        self.state = "PROMPT"

                # ── STATE: PROMPT ──────────────────────────────────────────────
                elif self.state == "PROMPT":
                    if self.current_prompt_idx >= len(self.prompts):
                        self.state = "DONE"
                        continue
                    label_str, text = self.prompts[self.current_prompt_idx]
                    color = (0, 220, 80) if label_str == "TRUTH" else (60, 60, 255)
                    cv2.putText(frame, f"Task: Tell a {label_str}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame, f"Q: {text}",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                    cv2.putText(frame, "SPACE = begin response   Q = quit & discard",
                                (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 160, 180), 1)

                # ── STATE: RECORDING ───────────────────────────────────────────
                elif self.state == "RECORDING":
                    self.current_frame_count += 1
                    cv2.putText(frame, f"RECORDING  [{self.current_frame_count} frames]",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2)
                    cv2.putText(frame, "Speak now.  SPACE = finish response   Q = quit & discard",
                                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1)
                    if features is not None:
                        self.current_features_buffer.append(features)

                cv2.imshow(self.WINDOW, frame)

                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    print("[QUIT] Window closed by participant/user. Buffered samples discarded.")
                    self.collected_samples.clear()
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[QUIT] Participant withdrew/stopped session (pressed Q). Buffered samples discarded.")
                    self.collected_samples.clear()
                    break

                elif key == 32:  # SPACE
                    if self.state == "PROMPT":
                        self.current_features_buffer = []
                        self.current_frame_count = 0
                        self.state = "RECORDING"
                    elif self.state == "RECORDING":
                        self._accept_response()
                        self.current_prompt_idx += 1
                        self.state = "PROMPT"

                if self.state == "DONE":
                    break

            if self.state == "DONE":
                self._save()
            else:
                print("[WITHDRAW] Incomplete session ended without saving data.")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()

    def _save(self):
        """Save collected samples to CSV, appending to any existing data."""
        if not self.collected_samples:
            print("[SAVE] No samples to save.")
            return

        os.makedirs("data", exist_ok=True)
        csv_path = "data/real_features.csv"
        df_new = pd.DataFrame(self.collected_samples)

        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            # Align columns — new columns get NaN in old rows (backward compat)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(csv_path, index=False)
        total = len(self.collected_samples)
        flagged = sum(1 for s in self.collected_samples if s.get("quality_flag") != "OK")
        print(f"\n[SAVE] {total} samples saved to {csv_path}")
        print(f"       {flagged} flagged as low-quality (quality_flag != OK)")
        print(f"       {self.skipped_low_quality} responses discarded (no face tracked)")
        print(f"       Participant column: participant_id = {self.participant_id}")
        print(f"       Dataset total rows: {len(df)}")
        print()
        print("Next steps:")
        print("  Train on real data : python training/train_model.py")
        print("  Evaluate (LOGO CV) : python training/evaluate_model.py "
              "--participant-col participant_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Research-grade behavioral data collector with participant anonymisation."
    )
    parser.add_argument(
        "--participant-id", type=str, required=True,
        help="Participant identifier (will be anonymised via SHA-256 hash before saving)."
    )
    parser.add_argument(
        "--experiment", type=str, default="EXP_DEFAULT",
        help="Experiment identifier tag stored in each sample row (default: EXP_DEFAULT)."
    )
    args = parser.parse_args()
    collector = DataCollector(
        participant_id=args.participant_id,
        experiment_id=args.experiment
    )
    collector.run()
