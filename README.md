# 🔍 Deception Analysis System v2.0 (Enhanced Edition)

> **⚠️ EDUCATIONAL & EXPERIMENTAL RESEARCH SYSTEM ONLY**
> This system does NOT provide scientifically validated lie detection.
> Results are behavioral estimates and stress anomalies, NOT definitive truth assessments.
> Do NOT use this for any legal, employment, or judicial purposes.

A real-time multimodal facial and physiological behavioral analysis system using computer vision,
MediaPipe Face Mesh (468+ landmarks), remote photoplethysmography (rPPG), and machine learning to estimate behavioral anomaly patterns.

---

## 🎯 Enhanced Features

- **Real-time face detection** with 468+ facial landmarks (MediaPipe Face Mesh)
- **Multi-channel behavioral explainability**:
  - *Ocular Channel*: Eye Aspect Ratio (EAR), blink anomalies, gaze saccades, fixation stability
  - *Myofacial Tension Channel*: Eyebrow furrowing, lip compression, jaw clench, facial tension
  - *Micro-Dynamics Channel*: Micro-expression burst detection (<200ms) and bilateral facial asymmetry deltas
  - *Autonomic Channel*: Remote photoplethysmography (forehead pulse rPPG estimation via optical FFT)
- **Structured Elicitation Protocols** (inspired by advanced deception elicitation benchmarks):
  - *Alibi Investigation Protocol*: Alibi declaration, witness corroboration, peripheral details, and reverse-chronology recall
  - *Standard Polygraph Protocol*: Baseline factual questions, control comparison questions, critical target probes
  - *Cognitive Load Protocol*: Dual-task load questions designed to elevate behavioral leakage
- **Scientific Evaluation Suite** (`training/evaluate_model.py`):
  - Balanced Accuracy, Cohen's Kappa, Brier Score Loss, Expected Calibration Error (ECE)
  - Wilson Score 95% Confidence Intervals for Accuracy, Precision, Recall, Specificity
  - Threshold sweep operating characteristics & automated JSON evaluation export
- **Automated Audit Reporting** (`modules/report_generator.py`):
  - Generates comprehensive post-session Markdown audit reports (`logs/session_report_*.md`) and structured JSON summaries (`logs/session_summary_*.json`)
- **Futuristic Dark HUD UI**:
  - Real-time animated polygraph wave, question status banner, neon-colored metrics, confidence intervals, and 4-channel anomaly mini-gauges
- **Comprehensive Automated Test Suite** (`tests/`):
  - Unit and integration tests covering model inference, explainability, elicitation, polygraph, and reporting

---

## 📁 Project Structure

```
lie_detection_system/
├── main.py                         # Main application entry point & camera HUD loop
├── config.py                       # System configuration constants & color palette
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive system documentation
│
├── modules/                        # Core analysis modules
│   ├── __init__.py                 # Package exports
│   ├── face_detector.py            # MediaPipe face detection & landmarks
│   ├── blink_detector.py           # Eye blink detection (EAR method)
│   ├── movement_tracker.py         # Head pose (pitch/yaw/roll) & gaze tracking
│   ├── micro_expression.py         # Micro-expression & symmetry analysis
│   ├── stress_analyzer.py          # Composite stress computation
│   ├── feature_extractor.py        # 22-dimensional feature vector assembly
│   ├── ml_model.py                 # Calibrated deception model & confidence intervals
│   ├── explainability.py           # 4-channel behavioral anomaly decomposition
│   ├── elicitation_protocol.py     # Structured interview protocols (Alibi, Polygraph)
│   ├── polygraph_analyzer.py       # Forehead optical rPPG pulse & waveform history
│   ├── voice_analyzer.py           # Background question speech recognition
│   ├── hud_renderer.py             # Futuristic HUD overlay renderer
│   ├── calibration.py              # Baseline calibration manager
│   ├── session_logger.py           # Session CSV logging & audit report dispatch
│   └── report_generator.py         # Markdown and JSON session audit generator
│
├── training/                       # ML model training & evaluation
│   ├── train_model.py              # ML model training script (Random Forest, SVM, etc.)
│   ├── evaluate_model.py           # Scientific evaluation suite with Wilson CIs & ECE
│   ├── collect_data.py             # Interactive webcam data collection script
│   └── process_videos.py           # Video file batch feature extraction
│
├── tests/                          # Automated test suite
│   ├── test_ml_model.py            # Model inference, smoothing & fallback tests
│   ├── test_explainability.py      # Channel attribution & evidence decomposition
│   ├── test_elicitation.py         # Question protocols & lifecycle tests
│   ├── test_polygraph.py           # rPPG pulse estimation tests
│   ├── test_report_generator.py    # Report synthesis & schema validation tests
│   └── test_scientific_evaluation.py # Wilson score & calibration error tests
│
├── models/                         # Saved ML models (.pkl) & scalers
├── logs/                           # Session CSV logs, Markdown & JSON reports
└── data/                           # Training datasets
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- Webcam and microphone
- Windows / macOS / Linux

### 2. Environment Setup
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Test Suite
Ensure all components are validated:
```powershell
pytest -v tests
```

### 4. Train the ML Model
```powershell
python training/train_model.py
```

### 5. Evaluate the Model Scientifically
```powershell
python training/evaluate_model.py
```
Outputs point estimates with 95% Wilson confidence intervals, Cohen's Kappa, Brier score, and exports `training/evaluation_report.json`.

### 6. Run the Application
```powershell
python main.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|:---:|:---|
| `Q` | **Quit** the application |
| `SPACE` | **Start / Advance Protocol Question** (Cycles from question to question) |
| `P` | **Cycle Elicitation Protocol** (*Alibi Investigation* ⇄ *Standard Polygraph* ⇄ *Cognitive Load*) |
| `S` | **Save Session Data** (Exports CSV log, Markdown audit report, and JSON summary) |
| `C` | **Start / Restart Calibration** (Neutral face baseline calibration) |
| `R` | **Reset** all analyzers, trackers, and histories |
| `M` | **Toggle Mesh** visibility on/off |
| `T` | **Record Question via Microphone** (Voice speech-to-text recognition) |

---

## 🔬 Scientific Evaluation & Limitations

1. **Behavioral Proxy Correlation**: This system measures physiological and behavioral reactivity (blinking frequency, saccades, facial asymmetry deltas, and optical pulse changes). Elevated stress or cognitive effort does **not** equal deception.
2. **Confidence Intervals**: All scientific evaluations report 95% Wilson score intervals to prevent reporting overconfident point estimates.
3. **Probability Calibration**: The model measures Expected Calibration Error (ECE) and Brier scores to ensure probabilities reflect empirical risk rather than uncalibrated classification margins.
4. **Independent Validation**: Without independent, ground-truth forensic datasets from real-world high-stakes scenarios, automated deception detection remains unvalidated for definitive assessments.

---

## ⚖️ Ethics & Privacy

- **Informed Consent**: Never record or analyze an individual without their explicit, prior consent.
- **Privacy Minimization**: Raw video streams are processed in-memory and not uploaded to remote servers.
- **Audit Trails**: Post-session reports (`logs/session_report_*.md`) preserve transparent disclaimers and multi-channel evidence rationale to ensure accountable usage.
