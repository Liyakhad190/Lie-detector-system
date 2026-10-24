"""
============================================================================
 LIE DETECTION SYSTEM - Configuration Constants
============================================================================
 Central configuration file for all system parameters.
 Adjust these values to tune detection sensitivity and UI appearance.
============================================================================
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# CAMERA SETTINGS
# ──────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0                  # Default webcam
CAMERA_WIDTH = 1280               # Capture width
CAMERA_HEIGHT = 720               # Capture height
TARGET_FPS = 30                   # Target frame rate

# Bounded retry for transient frame-read failures.
# After CAMERA_READ_RETRY_LIMIT consecutive failures the loop exits cleanly.
# Between retries the loop sleeps CAMERA_READ_RETRY_DELAY_MS milliseconds to
# avoid busy-waiting.  The counter resets to 0 after any successful read.
CAMERA_READ_RETRY_LIMIT = 5       # Max consecutive failed cap.read() calls
CAMERA_READ_RETRY_DELAY_MS = 50   # Sleep (ms) between failed read attempts

# ──────────────────────────────────────────────────────────────────────────
# MEDIAPIPE FACE MESH SETTINGS
# ──────────────────────────────────────────────────────────────────────────
MAX_NUM_FACES = 3                 # Multi-face support
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True           # Enable iris tracking

# ──────────────────────────────────────────────────────────────────────────
# FACIAL LANDMARK INDICES (MediaPipe Face Mesh 468+ landmarks)
# ──────────────────────────────────────────────────────────────────────────

# Eye landmarks for blink detection (EAR - Eye Aspect Ratio)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Iris landmarks (refined mode)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Lip landmarks for mouth movement
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Face contour for symmetry analysis
LEFT_FACE_CONTOUR = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_FACE_CONTOUR = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# Eyebrow landmarks for tension
LEFT_EYEBROW = [276, 283, 282, 295, 300]
RIGHT_EYEBROW = [46, 53, 52, 65, 70]

# Nose tip for head movement
NOSE_TIP = 1

# Forehead reference points
FOREHEAD = [10, 338, 297, 332, 284]

# Chin
CHIN = [152]

# ──────────────────────────────────────────────────────────────────────────
# BLINK DETECTION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.21              # Eye Aspect Ratio threshold for blink
EAR_CONSEC_FRAMES = 2             # Consecutive frames for valid blink
NORMAL_BLINK_RATE = (15, 20)      # Normal blinks per minute range
STRESS_BLINK_RATE = 25            # Blinks/min indicating stress

# ──────────────────────────────────────────────────────────────────────────
# CALIBRATION SETTINGS
# ──────────────────────────────────────────────────────────────────────────
CALIBRATION_DURATION = 10         # Seconds for baseline calibration
CALIBRATION_SAMPLES = 300         # Minimum samples to collect

# ──────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────
FEATURE_WINDOW_SIZE = 30          # Rolling window for feature computation
MICRO_EXPRESSION_FRAMES = 8      # Frames to detect micro-expressions
MOVEMENT_SMOOTHING = 0.7          # Exponential smoothing factor

# ──────────────────────────────────────────────────────────────────────────
# ML MODEL SETTINGS
# ──────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/lie_detector_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
MODEL_TYPE = "random_forest"      # Options: random_forest, svm, logistic, neural_net
PREDICTION_SMOOTHING = 0.6        # Smoothing for prediction output
CONFIDENCE_THRESHOLD = 0.65       # Min confidence for positive detection

# Feature names for the ML model
FEATURE_NAMES = [
    "blink_rate",
    "blink_rate_change",
    "ear_mean",
    "ear_std",
    "gaze_horizontal_std",
    "gaze_vertical_std",
    "gaze_fixation_ratio",
    "head_pitch_std",
    "head_yaw_std",
    "head_roll_std",
    "head_movement_magnitude",
    "face_symmetry_score",
    "face_symmetry_change",
    "lip_movement_intensity",
    "lip_compression_ratio",
    "eyebrow_tension",
    "eyebrow_asymmetry",
    "micro_expression_count",
    "micro_expression_intensity",
    "overall_facial_tension",
    "nervous_movement_score",
    "jaw_clench_indicator",
]

# ──────────────────────────────────────────────────────────────────────────
# SESSION LOGGING
# ──────────────────────────────────────────────────────────────────────────
LOG_DIR = "logs"
LOG_INTERVAL = 0.5                # Seconds between log entries

# ──────────────────────────────────────────────────────────────────────────
# HUD UI DESIGN - FUTURISTIC DARK THEME
# ──────────────────────────────────────────────────────────────────────────

# Color palette (BGR format for OpenCV)
HUD_COLORS = {
    "background":       (15, 15, 20),
    "panel_bg":         (25, 25, 35),
    "panel_border":     (0, 255, 200),       # Cyan-green
    "primary":          (0, 255, 200),       # Neon cyan
    "secondary":        (255, 180, 0),       # Amber
    "accent":           (200, 50, 255),      # Magenta
    "danger":           (0, 0, 255),         # Red
    "warning":          (0, 200, 255),       # Orange-yellow
    "success":          (0, 255, 100),       # Green
    "text_primary":     (220, 230, 240),     # Light gray
    "text_secondary":   (140, 150, 170),     # Dim gray
    "text_bright":      (255, 255, 255),     # White
    "grid":             (30, 35, 45),        # Subtle grid lines
    "landmark_dot":     (0, 255, 200),       # Landmark color
    "landmark_mesh":    (40, 80, 60),        # Mesh lines
    "truth":            (0, 255, 100),       # Green for truth
    "deception":        (0, 50, 255),        # Red for deception
    "stress_low":       (0, 255, 100),       # Green
    "stress_medium":    (0, 200, 255),       # Yellow
    "stress_high":      (0, 50, 255),        # Red
    "calibration":      (255, 200, 0),       # Gold for calibration
    "scanline":         (0, 255, 200, 30),   # Translucent scanline
}

# Font settings
HUD_FONT_SCALE = 0.5
HUD_FONT_THICKNESS = 1
HUD_TITLE_SCALE = 0.7
HUD_TITLE_THICKNESS = 2

# Panel dimensions
SIDE_PANEL_WIDTH = 320
TOP_BAR_HEIGHT = 45
BOTTOM_BAR_HEIGHT = 60
METRIC_BAR_HEIGHT = 20
METRIC_BAR_WIDTH = 200
POLYGRAPH_HEIGHT = 150
POLYGRAPH_WIDTH = 600

# ──────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLD ADAPTATION
# ──────────────────────────────────────────────────────────────────────────
ADAPTATION_RATE = 0.02            # How fast thresholds adapt
BASELINE_WEIGHT = 0.7             # Weight of baseline in adapted threshold

# ──────────────────────────────────────────────────────────────────────────
# DISCLAIMER & RESEARCH ETHICS NOTICE
# ──────────────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "EDUCATIONAL & EXPERIMENTAL SYSTEM ONLY | "
    "Output is a behavioral-anomaly score, NOT a lie verdict | "
    "NOT scientifically validated for lie detection | "
    "Results are behavioral estimates, NOT definitive truth assessments | "
    "Lighting, movement, fatigue, stress & anxiety affect measured signals | "
    "Do NOT use to accuse, rank, or make consequential decisions about individuals"
)

