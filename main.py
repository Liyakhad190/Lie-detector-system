"""
============================================================================
 LIE DETECTION SYSTEM - Main Application
============================================================================
 Real-time behavioral analysis system using facial landmark tracking.
 
 DISCLAIMER: This is an EDUCATIONAL and EXPERIMENTAL project only.
 It does NOT provide scientifically validated lie detection.
 Results are behavioral estimates, not definitive truth assessments.
 
 Controls:
   Q - Quit
   S - Save session logs to CSV
   C - Start/restart calibration
   R - Reset all analysis
   M - Toggle mesh visibility
   
 Architecture:
   Camera -> Face Detection -> Feature Extraction -> ML Prediction -> HUD
============================================================================
"""

import sys
import os
import time
import cv2
import numpy as np
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.face_detector import FaceDetector, FaceData
from modules.blink_detector import BlinkDetector
from modules.movement_tracker import MovementTracker
from modules.micro_expression import MicroExpressionAnalyzer
from modules.stress_analyzer import StressAnalyzer
from modules.feature_extractor import FeatureExtractor
from modules.ml_model import DeceptionModel
from modules.hud_renderer import HUDRenderer
from modules.calibration import CalibrationManager
from modules.session_logger import SessionLogger
from modules.polygraph_analyzer import PolygraphAnalyzer
from modules.voice_analyzer import VoiceAnalyzer
from modules.elicitation_protocol import ElicitationProtocolManager, QuestionCategory


class LieDetectionSystem:
    """
    Main application class orchestrating all analysis modules.
    
    Pipeline:
    1. Capture frame from webcam
    2. Detect faces and extract landmarks
    3. Analyze blink patterns, movement, micro-expressions
    4. Compute stress indicators & remote photoplethysmography (rPPG)
    5. Extract feature vector
    6. Predict calibrated deception/anomaly probability with explainability
    7. Render futuristic HUD overlay & interview protocol prompts
    8. Log results and generate Markdown/JSON audit reports
    """

    def __init__(self):
        print("=" * 60)
        print(" DECEPTION ANALYSIS SYSTEM v2.0")
        print(" Educational & Experimental Use Only")
        print("=" * 60)

        # Initialize camera — use DirectShow backend (MSMF is broken on this system)
        print("[INIT] Opening camera...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera connection.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INIT] Camera resolution: {actual_w}x{actual_h}")


        # Initialize all modules
        print("[INIT] Loading modules...")
        self.face_detector = FaceDetector()
        self.blink_detectors: Dict[int, BlinkDetector] = {}
        self.movement_trackers: Dict[int, MovementTracker] = {}
        self.micro_analyzers: Dict[int, MicroExpressionAnalyzer] = {}
        self.stress_analyzers: Dict[int, StressAnalyzer] = {}
        self.polygraph_analyzers: Dict[int, PolygraphAnalyzer] = {}
        self.feature_extractor = FeatureExtractor()
        self.model = DeceptionModel()
        self.hud = HUDRenderer(actual_w, actual_h)
        cv2.namedWindow("Deception Analysis System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deception Analysis System", actual_w, actual_h)
        cv2.waitKey(1)
        self.calibration = CalibrationManager()
        self.logger = SessionLogger()
        self.voice_analyzer = VoiceAnalyzer()
        self.protocol_manager = ElicitationProtocolManager("ALIBI_INVESTIGATION")

        # State
        self.show_mesh = False
        self.running = True
        self._fps = 0.0
        self._frame_times = []
        self._frame_count = 0
        
        # Interview Mode State
        self.interview_state = 0 # 0=IDLE, 1=RECORDING, 2=VERDICT
        self.recording_buffer = []
        self.final_verdict = None

        # Start session
        self.logger.start_session()
        print("[INIT] System ready! Starting calibration...")
        print("[CONTROLS] Q=Quit | S=Save/Audit | C=Calibrate | R=Reset | M=Mesh | P=Cycle Protocol | SPACE=Next Question | T=Record Voice")
        print()

        # Auto-start calibration
        self.calibration.start()

    def _get_or_create_analyzers(self, face_id: int):
        """Get or create per-face analyzer instances."""
        if face_id not in self.blink_detectors:
            self.blink_detectors[face_id] = BlinkDetector()
        if face_id not in self.movement_trackers:
            self.movement_trackers[face_id] = MovementTracker()
        if face_id not in self.micro_analyzers:
            self.micro_analyzers[face_id] = MicroExpressionAnalyzer()
        if face_id not in self.stress_analyzers:
            self.stress_analyzers[face_id] = StressAnalyzer()
        if face_id not in self.polygraph_analyzers:
            self.polygraph_analyzers[face_id] = PolygraphAnalyzer()

        return (
            self.blink_detectors[face_id],
            self.movement_trackers[face_id],
            self.micro_analyzers[face_id],
            self.stress_analyzers[face_id],
            self.polygraph_analyzers[face_id],
        )

    def _process_face(self, face: FaceData, frame: np.ndarray) -> Dict:
        """Run the full analysis pipeline for a single face."""
        blink_det, move_track, micro_analyzer, stress_analyzer, polygraph = \
            self._get_or_create_analyzers(face.face_id)
            
        frame_shape = frame.shape

        # Step 1: Blink analysis
        blink_data = blink_det.update(face.landmarks)

        # Step 2: Movement tracking
        movement_data = move_track.update(face.landmarks, frame_shape[:2])

        # Step 3: Micro-expression analysis
        micro_data = micro_analyzer.update(face.landmarks)

        # Step 4: Stress computation
        stress_data = stress_analyzer.compute_stress(
            blink_data, movement_data, micro_data
        )

        # Step 5: Feature extraction
        features = self.feature_extractor.extract(
            blink_data, movement_data, micro_data, stress_data
        )

        # Step 6: Physiological rPPG update (forehead pulse)
        polygraph_data = polygraph.update(
            frame,
            face.landmarks,
            stress_data.get("stress_score", 0.0),
            deception_prob=None
        )

        # Step 7: Calibration sample or prediction with explainability & autonomic fusion
        prediction = {}
        if self.calibration.is_calibrating:
            self.calibration.add_sample(
                blink_data, movement_data, micro_data, stress_data, features
            )
            prediction = {
                "deception_probability": 0.0,
                "truth_probability": 1.0,
                "confidence": 0.0,
                "prediction_stability_band": (0.0, 0.0),
                "confidence_interval": (0.0, 0.0),        # deprecated alias
                "label": "CALIBRATING...",
                "label_color": "calibration",
                "method": "CALIBRATION",
                "indicators": {},
                "explainability": {},
            }
        else:
            # Apply baseline if available
            if self.calibration.is_complete and self.calibration.baseline:
                bl = self.calibration.baseline
                blink_det.set_baseline(bl["avg_ear"], bl["blink_rate"])
                micro_analyzer.set_baseline(bl["face_symmetry"], bl["lip_movement"])
                self.model.set_baseline(bl["avg_features"])

            prediction = self.model.predict(features, polygraph_data=polygraph_data)

        # Record prediction into active protocol question
        self.protocol_manager.record_frame(prediction)

        # Log data
        current_q_text = ""
        active_q = self.protocol_manager.get_current_question()
        if active_q:
            current_q_text = f"[{active_q.category.value}] {active_q.text}"
        elif self.voice_analyzer.get_question():
            current_q_text = self.voice_analyzer.get_question()

        self.logger.log(
            time.time(), face.face_id,
            blink_data, movement_data, micro_data,
            stress_data, prediction,
            question=current_q_text
        )

        # Update Polygraph with final prediction probability for chart (once per frame)
        polygraph.add_deception(prediction.get("deception_probability", 0.0))
        
        # Buffer for interview mode
        if self.interview_state == 1:
            self.recording_buffer.append(prediction)

        return {
            "face_id": face.face_id,
            "bbox": face.bbox,
            "blink_data": blink_data,
            "movement_data": movement_data,
            "micro_data": micro_data,
            "stress_data": stress_data,
            "features": features,
            "prediction": prediction,
            "polygraph_data": polygraph_data,
        }

    def _compute_fps(self):
        """Compute rolling FPS."""
        now = time.time()
        self._frame_times.append(now)
        # Keep last 30 timestamps
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

    def _handle_keys(self, key: int):
        """Process keyboard input."""
        if key == ord('q') or key == ord('Q'):
            self.running = False
        elif key == ord('s') or key == ord('S'):
            saved = self.logger.save(
                protocol_summary=self.protocol_manager.get_protocol_summary(),
                baseline=self.calibration.baseline
            )
            if saved:
                print(f"[SAVE] Session data & audit reports saved to: {saved}")
        elif key == ord('p') or key == ord('P'):
            protocols = list(self.protocol_manager.PROTOCOLS.keys())
            curr_proto = self.protocol_manager.protocol_name
            next_proto = protocols[(protocols.index(curr_proto) + 1) % len(protocols)]
            self.protocol_manager.load_protocol(next_proto)
            print(f"[PROTOCOL] Switched active protocol to: {next_proto}")
        elif key == ord('c') or key == ord('C'):
            print("[CALIBRATION] Restarting calibration...")
            self.calibration.reset()
            self.calibration.start()
        elif key == ord('r') or key == ord('R'):
            print("[RESET] Resetting all analysis...")
            for d in self.blink_detectors.values():
                d.reset()
            for d in self.movement_trackers.values():
                d.reset()
            for d in self.micro_analyzers.values():
                d.reset()
            for d in self.stress_analyzers.values():
                d.reset()
            self.polygraph_analyzers.clear()
            self.model.reset()
        elif key == ord('m') or key == ord('M'):
            self.show_mesh = not self.show_mesh
            print(f"[UI] Mesh display: {'ON' if self.show_mesh else 'OFF'}")
        elif key == 32: # Spacebar
            if not self.protocol_manager.is_active:
                self.protocol_manager.start_protocol()
                self.interview_state = 1
                q = self.protocol_manager.get_current_question()
                print(f"[PROTOCOL] Started '{self.protocol_manager.protocol_name}'. Q1: {q.text if q else ''}")
            else:
                summary = self.protocol_manager.advance_question()
                q = self.protocol_manager.get_current_question()
                if q:
                    print(f"[PROTOCOL] Q{q.qid} [{q.category.value}]: {q.text}")
                    self.interview_state = 1
                else:
                    print("[PROTOCOL] All questions completed! Computing final protocol verdict...")
                    self.interview_state = 2
                    self._compute_final_verdict()
        elif key == ord('t') or key == ord('T'):
            print("[VOICE] Listening for custom spoken question...")
            self.voice_analyzer.start_listening()

    def _compute_final_verdict(self):
        proto_summary = self.protocol_manager.get_protocol_summary()
        mean_crit = proto_summary.get("mean_critical_score", 0.0)
        mean_base = proto_summary.get("mean_baseline_score", 0.0)
        delta = proto_summary.get("probe_reactivity_delta", 0.0)
        verdict = proto_summary.get("overall_verdict", "INCONCLUSIVE")
        color = proto_summary.get("verdict_color", "warning")
        
        self.final_verdict = {
            "deception_probability": mean_crit,
            "truth_probability": max(0.0, 1.0 - mean_crit),
            "label": verdict,
            "label_color": color,
            "confidence": min(0.95, max(0.40, 0.50 + abs(delta))),
            "details": f"Probe Reactivity Delta: {delta:+.2f} (Critical: {mean_crit:.2f} vs Base: {mean_base:.2f})"
        }

    def run(self):
        """Main application loop with bounded camera-read retry on transient failures.

        Retry policy (configurable via config.py):
          - On cap.read() failure: increment consecutive-failure counter,
            sleep config.CAMERA_READ_RETRY_DELAY_MS / 1000.0 seconds via
            time.sleep(), then retry.
          - After config.CAMERA_READ_RETRY_LIMIT consecutive failures: log a
            [CAMERA] error and exit cleanly via shutdown().
          - After any successful read: reset the consecutive-failure counter to 0.
          - Invalid / None frames are never passed downstream.

        Note: this retry logic was validated with unit-test mocks only.
        Actual webcam recovery depends on hardware and OS driver behaviour.
        """
        print("[RUN] Starting main loop...")
        _consecutive_failures = 0
        _retry_limit = config.CAMERA_READ_RETRY_LIMIT
        _retry_delay_ms = config.CAMERA_READ_RETRY_DELAY_MS

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    _consecutive_failures += 1
                    if _consecutive_failures == 1:
                        print(
                            f"[CAMERA] Frame read failed "
                            f"(attempt {_consecutive_failures}/{_retry_limit}). "
                            "Retrying..."
                        )
                    else:
                        print(
                            f"[CAMERA] Frame read failed "
                            f"(attempt {_consecutive_failures}/{_retry_limit})."
                        )

                    if _consecutive_failures >= _retry_limit:
                        print(
                            f"[CAMERA] {_retry_limit} consecutive frame failures. "
                            "Camera may be disconnected. Exiting."
                        )
                        break

                    # time.sleep() avoids busy-waiting and works correctly
                    # in headless environments and unit tests where an OpenCV
                    # window is not guaranteed to exist.
                    # cv2.waitKey() is kept only in the normal frame loop below
                    # where a display window is active.
                    time.sleep(_retry_delay_ms / 1000.0)
                    continue

                # Successful read: reset failure counter
                if _consecutive_failures > 0:
                    print(
                        f"[CAMERA] Frame recovered after "
                        f"{_consecutive_failures} failure(s)."
                    )
                _consecutive_failures = 0

                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)
                self._frame_count += 1
                self._compute_fps()

                # Detect faces
                faces = self.face_detector.detect(frame)

                # Draw landmarks on frame only if mesh toggle is ON (default OFF)
                if self.show_mesh:
                    for face in faces:
                        self.face_detector.draw_landmarks(
                            frame, face,
                            draw_mesh=True, draw_contour=True, draw_iris=True
                        )

                # Process each face through analysis pipeline
                faces_data = []
                for face in faces:
                    try:
                        result = self._process_face(face, frame)
                        faces_data.append(result)
                    except Exception as e:
                        print(f"[FACE] Processing error for face_id={face.face_id}: {e}")

                # Determine active question text
                active_q = self.protocol_manager.get_current_question()
                if active_q:
                    q_display = f"[{self.protocol_manager.protocol_name[:5]}|{active_q.category.value}] Q{active_q.qid}: {active_q.text}"
                else:
                    q_display = self.voice_analyzer.get_question()

                # Render HUD overlay
                frame = self.hud.render(
                    frame,
                    faces_data,
                    calibration_progress=self.calibration.get_progress(),
                    is_calibrating=self.calibration.is_calibrating,
                    calibration_remaining=self.calibration.get_remaining_seconds(),
                    fps=self._fps,
                    log_count=self.logger.get_entry_count(),
                    interview_state=self.interview_state,
                    final_verdict=self.final_verdict,
                    current_question=q_display
                )

                # Display
                cv2.imshow("Deception Analysis System", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_keys(key)

        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted by user.")
        except Exception as e:
            print(f"\n[FATAL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()


    def shutdown(self):
        """Clean shutdown of all resources."""
        print("\n[SHUTDOWN] Saving session data & generating audit report...")
        self.logger.close(
            protocol_summary=self.protocol_manager.get_protocol_summary(),
            baseline=self.calibration.baseline
        )
        self.voice_analyzer.stop_listening()
        print("[SHUTDOWN] Releasing camera...")
        self.cap.release()
        print("[SHUTDOWN] Releasing face detector...")
        self.face_detector.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete. Goodbye!")


def main():
    """Entry point."""
    try:
        system = LieDetectionSystem()
        system.run()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
