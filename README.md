# 🔍 Deception Analysis System v2.0

> **⚠️ EDUCATIONAL & EXPERIMENTAL PROJECT ONLY**
> This system does NOT provide scientifically validated lie detection.
> Results are behavioral estimates, not definitive truth assessments.
> Do NOT use this for any legal, employment, or judicial purposes.

A real-time facial behavioral analysis system using computer vision,
MediaPipe Face Mesh, and machine learning to estimate potential deception
indicators from facial behavior patterns.

---

## 🎯 Features

- **Real-time face detection** with 468+ facial landmarks (MediaPipe)
- **Multi-face support** (up to 3 simultaneous faces)
- **Eye blink detection** using Eye Aspect Ratio (EAR) algorithm
- **Eye gaze tracking** via iris landmark analysis
- **Head pose estimation** (pitch, yaw, roll)
- **Face symmetry analysis** for expression authenticity
- **Lip movement & compression** tracking
- **Micro-expression detection** via rapid landmark changes
- **Stress level computation** from weighted behavioral indicators
- **ML-based deception prediction** (Random Forest, SVM, Logistic Regression, MLP)
- **Calibration mode** for individual baseline establishment
- **Futuristic dark HUD UI** with neon-colored metrics and animated scanlines
- **CSV session logging** for post-analysis
- **Dynamic threshold adaptation** based on calibration baseline

---

## 📁 Project Structure

```
lie_detection_system/
├── main.py                         # Main application entry point
├── config.py                       # All configuration constants
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── modules/                        # Core analysis modules
│   ├── __init__.py                 # Package exports
│   ├── face_detector.py            # MediaPipe face detection & landmarks
│   ├── blink_detector.py           # Eye blink detection (EAR method)
│   ├── movement_tracker.py         # Head pose & gaze tracking
│   ├── micro_expression.py         # Micro-expression & symmetry analysis
│   ├── stress_analyzer.py          # Composite stress computation
│   ├── feature_extractor.py        # ML feature vector assembly
│   ├── ml_model.py                 # Deception prediction model
│   ├── hud_renderer.py             # Futuristic HUD overlay renderer
│   ├── calibration.py              # Baseline calibration manager
│   └── session_logger.py           # CSV logging
│
├── training/                       # ML model training
│   ├── train_model.py              # Training script (multiple models)
│   └── evaluate_model.py           # Model evaluation & metrics
│
├── models/                         # Saved ML models (.pkl)
├── logs/                           # Session CSV logs
├── data/                           # Training datasets
│   └── sample_features.csv         # Generated after training
│
└── docs/
    └── ml_pipeline.md              # ML pipeline documentation
```

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or USB)
- Windows / macOS / Linux

### Step 1: Clone or Download
```bash
cd lie_detection_system
```

### Step 2: Create Virtual Environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Train ML Model
```bash
# Train with synthetic data (Random Forest default)
python training/train_model.py

# Train with different model
python training/train_model.py --model svm
python training/train_model.py --model logistic
python training/train_model.py --model neural_net
python training/train_model.py --model gradient_boost
```

### Step 5: Run the System
```bash
python main.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save session logs to CSV |
| `C` | Start/restart calibration |
| `R` | Reset all analysis data |
| `M` | Toggle mesh visibility |

---

## 📊 Analyzed Metrics

| Metric | Description |
|--------|-------------|
| **Blink Rate** | Blinks per minute (normal: 15-20) |
| **Eye Aspect Ratio** | Eye openness measurement |
| **Gaze Stability** | How steady the eye gaze is |
| **Head Movement** | Overall head motion magnitude |
| **Nervous Score** | Composite fidgeting measurement |
| **Face Symmetry** | Left-right facial symmetry (%) |
| **Lip Compression** | Lip tightness indicator |
| **Brow Tension** | Eyebrow lowering/furrowing |
| **Micro-Expressions** | Count of rapid facial changes |
| **Jaw Clench** | Jaw tightness indicator |
| **Facial Tension** | Overall facial stress level |
| **Stress Level** | LOW / MEDIUM / HIGH composite |
| **Deception Probability** | Estimated deception likelihood |
| **Truth Probability** | Estimated truthfulness likelihood |
| **Confidence** | Prediction stability score |

---

## 🧠 How It Works

1. **Calibration** (10 seconds): Establishes your baseline behavior
2. **Detection**: Compares current behavior against your baseline
3. **Analysis**: 22 behavioral features extracted per frame
4. **Prediction**: ML model (or heuristic) estimates deception probability
5. **Display**: Real-time HUD shows all metrics with color-coded indicators

---

## 📈 Improving Accuracy

1. Collect real labeled truth/deception data for training
2. Extend calibration duration for better baselines
3. Add voice stress analysis (pitch, tempo, tremor)
4. Implement temporal models (LSTM/GRU) for sequence analysis
5. Use ensemble of multiple models
6. Add body language analysis via full pose estimation
7. Integrate speech content analysis
8. Fine-tune with individual-specific data

---

## ⚖️ Ethical Notice

- This project is for **learning and research only**
- Facial-behavior-based lie detection is **not scientifically reliable**
- Do NOT use results for any real-world decision making
- Always respect privacy and obtain consent before recording
- The system shows **behavioral indicators**, not definitive truth
