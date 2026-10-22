"""
============================================================================
 DATA COLLECTION SCRIPT
============================================================================
 Collects real human behavioral data by prompting you to tell truths and lies
 directly into your webcam. Saves to data/real_features.csv.
============================================================================
"""

import sys
import os
import time
import cv2
import numpy as np
import pandas as pd

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


class DataCollector:
    def __init__(self):
        print("Initializing Real Data Collector...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        self.face_detector = FaceDetector()
        self.blink_det = BlinkDetector()
        self.move_track = MovementTracker()
        self.micro_analyzer = MicroExpressionAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.calibration = CalibrationManager()

        self.collected_samples = []
        
        # We will collect 5 truths and 5 lies per session
        self.prompts = [
            ("TRUTH", "What did you eat for breakfast today?"),
            ("TRUTH", "Describe the room you are currently in."),
            ("TRUTH", "What is your favorite movie and why?"),
            ("TRUTH", "Describe your daily routine."),
            ("TRUTH", "What did you do last weekend?"),
            ("LIE", "Lie about what you ate for breakfast today."),
            ("LIE", "Describe a completely fake room instead of this one."),
            ("LIE", "Make up a fake favorite movie and explain why."),
            ("LIE", "Describe a completely fake daily routine."),
            ("LIE", "Make up a story about what you did last weekend."),
        ]
        np.random.shuffle(self.prompts)
        
        self.state = "CALIBRATION"
        self.current_prompt_idx = 0
        self.current_features_buffer = []

        self.calibration.start()

    def _process_frame(self, frame):
        faces = self.face_detector.detect(frame)
        if not faces:
            return None
        
        face = faces[0] # Just use the first face found
        self.face_detector.draw_key_landmarks(frame, face)

        blink_data = self.blink_det.update(face.landmarks)
        movement_data = self.move_track.update(face.landmarks, frame.shape[:2])
        micro_data = self.micro_analyzer.update(face.landmarks)
        stress_data = self.stress_analyzer.compute_stress(blink_data, movement_data, micro_data)
        features = self.feature_extractor.extract(blink_data, movement_data, micro_data, stress_data)

        if self.calibration.is_calibrating:
            self.calibration.add_sample(blink_data, movement_data, micro_data, stress_data, features)
        elif self.calibration.is_complete and self.calibration.baseline:
            bl = self.calibration.baseline
            self.blink_det.set_baseline(bl["avg_ear"], bl["blink_rate"])
            self.micro_analyzer.set_baseline(bl["face_symmetry"], bl["lip_movement"])

        return features

    def run(self):
        print("\n[READY] Look at the camera. Calibration starting...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            features = self._process_frame(frame)
            
            # Rendering UI based on state
            if self.state == "CALIBRATION":
                prog = self.calibration.get_progress()
                cv2.putText(frame, f"CALIBRATING... {prog:.0%}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if not self.calibration.is_calibrating:
                    self.state = "PROMPT"
                    
            elif self.state == "PROMPT":
                if self.current_prompt_idx >= len(self.prompts):
                    self.state = "DONE"
                    continue
                    
                label, text = self.prompts[self.current_prompt_idx]
                color = (0, 255, 0) if label == "TRUTH" else (0, 0, 255)
                
                cv2.putText(frame, f"NEXT: Tell a {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Prompt: {text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to start recording your answer.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
            elif self.state == "RECORDING":
                cv2.putText(frame, "RECORDING... PLEASE SPEAK NOW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press SPACE when finished.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                if features:
                    self.current_features_buffer.append(features)

            cv2.imshow("Data Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32: # SPACE
                if self.state == "PROMPT":
                    self.state = "RECORDING"
                    self.current_features_buffer = []
                elif self.state == "RECORDING":
                    # Aggregate features and save
                    if self.current_features_buffer:
                        label, _ = self.prompts[self.current_prompt_idx]
                        avg_features = {k: np.mean([f[k] for f in self.current_features_buffer]) for k in config.FEATURE_NAMES}
                        avg_features['label'] = 0 if label == "TRUTH" else 1
                        self.collected_samples.append(avg_features)
                        print(f"Recorded sample {self.current_prompt_idx+1}/{len(self.prompts)} -> {label}")
                        
                    self.current_prompt_idx += 1
                    self.state = "PROMPT"
                    
            if self.state == "DONE":
                break

        # Save data
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.collected_samples:
            os.makedirs("data", exist_ok=True)
            df = pd.DataFrame(self.collected_samples)
            csv_path = "data/real_features.csv"
            
            # Append if exists
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df = pd.concat([df_old, df], ignore_index=True)
                
            df.to_csv(csv_path, index=False)
            print(f"\n[SUCCESS] Saved {len(self.collected_samples)} real samples to {csv_path}")
            print(f"Total dataset size is now {len(df)} samples.")
            print("You can now run 'python training/train_model.py' to train on your real data!")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
