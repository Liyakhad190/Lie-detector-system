# Reproducible Research Experiment Protocol & Label Documentation

## 1. Research Objective & Scientific Scope

This protocol defines the methodology for conducting controlled, reproducible research experiments to measure the behavior, signal stability, and limitations of the Deception Analysis System.

**Important Notice:**
> This system estimates behavioral sensor divergence (anomalies) from a calibrated baseline. It does **NOT** provide scientifically validated lie detection. It must never be used to accuse, judge, rank, or make consequential decisions about any individual.

---

## 2. Data Collection Protocol (`training/collect_data.py`)

### 2.1 Consent & Camera Gating
1. **Zero Pre-Consent Capture:** The physical webcam hardware is **never opened or queried** until after the participant reads and explicitly accepts the on-screen Informed Consent screen by pressing `Y`.
2. **Refusal Handling:** Pressing `N`, `Q`, or closing the consent window immediately terminates execution. No camera is initialized, no video buffer is allocated, and no participant records are created.
3. **Voluntary Withdrawal:** Participants may withdraw at any time during calibration or prompt recording by pressing `Q` or closing the camera window. All buffered frames in memory are immediately cleared via `.clear()` and no file is created or updated.

### 2.2 Pseudonymization & Privacy
- **No Personal Identifiers:** Raw participant names, contact details, raw video frames, and facial photographs are **never** stored to disk.
- **Pseudonym Hash:** Any input identifier is hashed using SHA-256: `P_<sha256(raw_id)[:10]>`.
- **Pseudonymization vs Anonymity Notice:** Hashed participant IDs prevent casual identification but represent *pseudonymization*, not absolute mathematical anonymity in small cohort studies.

### 2.3 Experimental Prompts & Task Structure
Data collection presents 10 structured prompts in randomized order:
- **5 Truthful Prompts:** Requesting factual recollections (e.g., breakfast, room layout, daily routine).
- **5 Deceptive Prompts:** Requesting invented fictional accounts (e.g., fake breakfast, imagined room, fictional weekend).

For each prompt:
1. `PROMPT` state displays the instruction.
2. Participant presses `SPACE` to transition to `RECORDING`.
3. The feature extraction pipeline computes frame-by-frame 22-dimensional behavioral features.
4. Participant presses `SPACE` to finish the response.

---

## 3. Dataset Schema & Label Definitions (`data/real_features.csv`)

Each row represents an aggregated response to a single prompt:

| Column | Type | Description | Scientific Meaning / Limitation |
|---|---|---|---|
| `participant_id` | `str` | Format `P_<hash>` | Pseudonymized participant cluster key |
| `session_id` | `str` | Format `S_YYYYMMDD_HHMMSS` | Unique recording session timestamp |
| `experiment_id` | `str` | e.g., `EXP_DEFAULT` | Configurable experiment cohort tag |
| `prompt_index` | `int` | 0 to 9 | Order index of prompt presented |
| `task_label_str` | `str` | `TRUTH` or `LIE` | Human-readable task instruction given to participant |
| `label` | `int` | `0` (Truth) or `1` (Deception) | Binary label assigned strictly by task instruction |
| `frame_count` | `int` | Number of tracked frames | Validated against `MIN_QUALITY_FRAMES` (10 frames) |
| `quality_flag` | `str` | Validation status | `OK` if passed; otherwise `LOW_FRAMES`, `CONTAINS_NAN`, `CONTAINS_INF`, `INVALID_LABEL`, etc. |
| `data_provenance` | `str` | `REAL_HUMAN` | Declares data origin to prevent silent mixing |
| `label_protocol` | `str` | `CONTROLLED_TASK_DIRECTED` | Protocol methodology tag |
| `label_verified` | `bool` | `False` | **CRITICAL:** Indicates that deceptive intent was NOT biologically or independently verified. Reflects task instruction compliance only. |
| `blink_rate` ... `jaw_clench_indicator` | `float` | 22 behavioral features | Mean feature values across all valid tracked frames |

### 3.1 Understanding `label_verified=False`
In controlled lab tasks, participants are instructed to tell truths or lies. However:
- A participant instructed to lie may feel no emotional arousal or cognitive conflict.
- A participant instructed to tell the truth may experience severe anxiety, camera discomfort, or cognitive hesitation.
- **Therefore, `label_verified=False` signifies that target labels represent experimental task instructions, NOT independently verified deception.** The training pipeline and evaluation suite explicitly check this flag and refuse to claim human deception detection accuracy.

---

## 4. Confounding Variables & Environmental Limitations

Measurements in this system are non-invasive computer vision approximations. They are sensitive to numerous non-deceptive factors:

1. **Illumination & Shadows:** Fluorescent flicker, side-lighting, and dim illumination degrade MediaPipe Face Mesh landmark tracking and rPPG blood-volume pulse signals.
2. **Head Movement & Speech Articulation:** Talking, smiling, shifting posture, or yawning alters lip compression, eye aspect ratio, and facial symmetry independently of stress or deception.
3. **Anxiety & Camera-Consciousness:** Being recorded by a webcam naturally induces elevated blink rates, facial tension, and tachycardia in many individuals (the "Hawthorne effect").
4. **Fatigue & Neurological Baselines:** Sleep deprivation, eye strain, neurodivergence, or personal expressive habits differ fundamentally across individuals, making population-level thresholds inaccurate without personalized baseline calibration.

---

## 5. Reproducible Training & Evaluation Instructions

### 5.1 Training the Model
```bash
# Train on collected human data with participant-independent splits
python training/train_model.py --data data/real_features.csv

# Train on synthetic baseline data (clearly flagged as SYNTHETIC)
python training/train_model.py --data data/sample_features.csv
```

### 5.2 Evaluating Model Performance
```bash
# Run scientific evaluation with participant-independent Leave-One-Group-Out CV
python training/evaluate_model.py --data data/real_features.csv --participant-col participant_id --out training/evaluation_report.json

# Evaluate synthetic benchmark (clearly notes validity CANNOT BE ESTABLISHED)
python training/evaluate_model.py --data data/sample_features.csv --out training/evaluation_report.json
```

### 5.3 Verifying Safeguards via Automated Tests
```bash
pytest tests/
```
