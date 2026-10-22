# ML Pipeline Explanation

## Overview

The Deception Analysis System uses a multi-stage pipeline to estimate behavioral
indicators that *may* correlate with deception. This is for **educational purposes
only** and does not provide scientifically validated lie detection.

## Pipeline Architecture

```
Camera Frame
     │
     ▼
┌─────────────┐
│ Face Detect  │  MediaPipe Face Mesh (468+ landmarks)
│ (face_detector)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│              Feature Extraction                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Blink   │ │Movement  │ │Micro-Expression  │ │
│  │ Detector │ │ Tracker  │ │   Analyzer       │ │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │
│       │             │                │            │
│       ▼             ▼                ▼            │
│  ┌────────────────────────────────────────────┐  │
│  │         Stress Analyzer                     │  │
│  │    (weighted composite of all signals)      │  │
│  └───────────────────┬────────────────────────┘  │
│                      │                            │
│  ┌───────────────────▼────────────────────────┐  │
│  │       Feature Vector (22 features)          │  │
│  └───────────────────┬────────────────────────┘  │
└──────────────────────┼───────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   ML Model     │  Random Forest / SVM / etc.
              │  (or Heuristic │  Falls back to rule-based
              │   Fallback)    │  when no model is trained
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Prediction    │  Deception probability
              │  + Confidence  │  Truth probability
              │  + Indicators  │  Component scores
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  HUD Renderer  │  Futuristic overlay display
              │  + CSV Logger  │  Session data export
              └────────────────┘
```

## Feature Descriptions (22 Features)

| # | Feature | Description | Deception Signal |
|---|---------|-------------|------------------|
| 1 | blink_rate | Blinks per minute | High rate = stress |
| 2 | blink_rate_change | Change from baseline | Increase = anxiety |
| 3 | ear_mean | Average Eye Aspect Ratio | Lower = squinting |
| 4 | ear_std | EAR variability | Higher = instability |
| 5 | gaze_horizontal_std | Horizontal gaze variation | Avoidance |
| 6 | gaze_vertical_std | Vertical gaze variation | Avoidance |
| 7 | gaze_fixation_ratio | How steady is gaze | Low = shifting eyes |
| 8 | head_pitch_std | Vertical head movement | Nervous nodding |
| 9 | head_yaw_std | Horizontal head movement | Head shaking |
| 10 | head_roll_std | Head tilt variation | Fidgeting |
| 11 | head_movement_magnitude | Overall head motion | Agitation |
| 12 | face_symmetry_score | Left-right symmetry | Low = fake expression |
| 13 | face_symmetry_change | Symmetry deviation | Sudden asymmetry |
| 14 | lip_movement_intensity | Lip motion amount | Compressed = stress |
| 15 | lip_compression_ratio | Lip tightness | Tight lips = tension |
| 16 | eyebrow_tension | Brow lowering | Furrowed = stress |
| 17 | eyebrow_asymmetry | Left vs right brow | Uneven = concealment |
| 18 | micro_expression_count | Rapid facial changes | More = suppression |
| 19 | micro_expression_intensity | Strength of micro-expr | Strong = leakage |
| 20 | overall_facial_tension | Composite tension | High = stress |
| 21 | nervous_movement_score | Fidgety behavior | High = anxiety |
| 22 | jaw_clench_indicator | Jaw tightness | Clenching = stress |

## Calibration Phase

Before detection begins, a 10-second calibration phase establishes the
subject's **baseline** behavior:
- Normal blink rate
- Resting EAR
- Natural face symmetry
- Baseline movement patterns

All subsequent predictions compare current behavior against this baseline,
making the system adaptive to individual differences.

## Model Options

| Model | Pros | Cons |
|-------|------|------|
| Random Forest | Fast, interpretable, robust | May miss subtle patterns |
| SVM | Good with small datasets | Slower at inference |
| Logistic Regression | Very fast, interpretable | Limited complexity |
| Neural Network (MLP) | Captures non-linear patterns | Needs more data |
| Gradient Boosting | Often best accuracy | Slower to train |

## How to Improve Accuracy

1. **Real labeled data**: Collect actual truth/deception samples
2. **More features**: Add voice analysis, body language, response timing
3. **Temporal models**: Use LSTM/GRU for sequence patterns
4. **Ensemble methods**: Combine multiple model predictions
5. **Individual calibration**: Longer baseline per subject
6. **Context awareness**: Account for question type and topic
7. **Multi-modal fusion**: Combine face + voice + text analysis
8. **Transfer learning**: Fine-tune pre-trained emotion models

## Future Upgrades

1. **Voice Stress Analysis**: Pitch, tempo, and tremor detection
2. **Speech-to-Text**: Analyze verbal content for inconsistencies
3. **Body Language**: Full upper-body pose estimation
4. **Emotion Recognition**: Deep learning emotion classification
5. **TensorFlow/PyTorch Models**: CNN-based micro-expression detection
6. **Streamlit Dashboard**: Web-based monitoring interface
7. **SQLite Logging**: Structured database storage
8. **AI Reports**: LLM-generated behavioral summaries
9. **Multi-camera**: Support for multiple viewing angles
10. **Thermal Integration**: Facial blood flow analysis
