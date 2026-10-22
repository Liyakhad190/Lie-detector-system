"""
============================================================================
 VIDEO PROCESSING SCRIPT
============================================================================
 Extracts behavioral features from .mp4 video files of real people.
 Place truthful videos in data/raw_videos/truth/
 Place deceptive videos in data/raw_videos/lie/
 
 It will process them frame-by-frame and save to data/real_features.csv.
============================================================================
"""

import sys
import os
import cv2
import glob
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


class VideoProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
        
        self.truth_dir = os.path.join("data", "raw_videos", "truth")
        self.lie_dir = os.path.join("data", "raw_videos", "lie")
        
        os.makedirs(self.truth_dir, exist_ok=True)
        os.makedirs(self.lie_dir, exist_ok=True)
        
    def _process_video(self, video_path: str, label_idx: int) -> dict:
        """Processes a single video and returns average features."""
        print(f"Processing: {os.path.basename(video_path)} ...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open {video_path}")
            return None
            
        # Re-initialize stateful detectors for each new video
        blink_det = BlinkDetector()
        move_track = MovementTracker()
        micro_analyzer = MicroExpressionAnalyzer()
        stress_analyzer = StressAnalyzer()
        
        features_buffer = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Process every 2nd frame to save time (optional)
            # if frame_count % 2 != 0: continue
            
            faces = self.face_detector.detect(frame)
            if not faces:
                continue
                
            face = faces[0]
            
            blink_data = blink_det.update(face.landmarks)
            movement_data = move_track.update(face.landmarks, frame.shape[:2])
            micro_data = micro_analyzer.update(face.landmarks)
            stress_data = stress_analyzer.compute_stress(blink_data, movement_data, micro_data)
            features = self.feature_extractor.extract(blink_data, movement_data, micro_data, stress_data)
            
            # Note: We skip the first 30 frames to allow smoothing buffers to fill
            if frame_count > 30:
                features_buffer.append(features)
                
        cap.release()
        
        if not features_buffer:
            print("  [WARN] No usable face data found in video.")
            return None
            
        # Aggregate the video's features into a single row
        avg_features = {k: np.mean([f[k] for f in features_buffer]) for k in config.FEATURE_NAMES}
        avg_features['label'] = label_idx
        
        print(f"  [OK] Processed {frame_count} frames. Extracted features.")
        return avg_features

    def run(self):
        print("="*50)
        print(" VIDEO BATCH PROCESSOR")
        print("="*50)
        
        truth_videos = glob.glob(os.path.join(self.truth_dir, "*.mp4")) + glob.glob(os.path.join(self.truth_dir, "*.avi"))
        lie_videos = glob.glob(os.path.join(self.lie_dir, "*.mp4")) + glob.glob(os.path.join(self.lie_dir, "*.avi"))
        
        all_videos = [(v, 0) for v in truth_videos] + [(v, 1) for v in lie_videos]
        
        if not all_videos:
            print(f"No videos found!\nDrop .mp4 files into:\n - {self.truth_dir}\n - {self.lie_dir}")
            return
            
        print(f"Found {len(truth_videos)} truth videos and {len(lie_videos)} lie videos.")
        
        collected_samples = []
        for v_path, label in all_videos:
            feats = self._process_video(v_path, label)
            if feats:
                collected_samples.append(feats)
                
        if collected_samples:
            df = pd.DataFrame(collected_samples)
            csv_path = "data/real_features.csv"
            
            # Append if exists
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df = pd.concat([df_old, df], ignore_index=True)
                
            df.to_csv(csv_path, index=False)
            print(f"\n[SUCCESS] Saved {len(collected_samples)} real video samples to {csv_path}")
            print(f"Total dataset size is now {len(df)} samples.")
        
        print("Done.")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()
