"""
============================================================================
 TRAINING SCRIPT - Deception Detection ML Model
============================================================================
 Trains a machine learning model on behavioral feature data.
 Supports Random Forest, SVM, Logistic Regression, and MLP.
 
 Usage:
   python training/train_model.py
   python training/train_model.py --model svm
   python training/train_model.py --generate-data
============================================================================
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic training data that simulates behavioral patterns.
    
    Truthful behavior: stable blink rate, steady gaze, symmetric face,
    relaxed features.
    
    Deceptive behavior: elevated blink rate, gaze aversion, asymmetric
    expressions, increased tension.
    """
    np.random.seed(42)
    n_truth = n_samples // 2
    n_deception = n_samples - n_truth

    features = config.FEATURE_NAMES
    data = []

    # Generate truthful samples
    for _ in range(n_truth):
        sample = {
            "blink_rate": np.random.normal(17, 4),
            "blink_rate_change": np.random.normal(1.0, 0.15),
            "ear_mean": np.random.normal(0.30, 0.03),
            "ear_std": np.random.normal(0.02, 0.008),
            "gaze_horizontal_std": np.random.normal(0.1, 0.05),
            "gaze_vertical_std": np.random.normal(0.08, 0.04),
            "gaze_fixation_ratio": np.random.normal(0.85, 0.08),
            "head_pitch_std": np.random.normal(1.5, 0.8),
            "head_yaw_std": np.random.normal(1.2, 0.6),
            "head_roll_std": np.random.normal(0.8, 0.4),
            "head_movement_magnitude": np.random.normal(0.5, 0.3),
            "face_symmetry_score": np.random.normal(0.92, 0.04),
            "face_symmetry_change": np.random.normal(0.02, 0.01),
            "lip_movement_intensity": np.random.normal(8.0, 3.0),
            "lip_compression_ratio": np.random.normal(0.4, 0.1),
            "eyebrow_tension": np.random.normal(0.2, 0.1),
            "eyebrow_asymmetry": np.random.normal(0.05, 0.03),
            "micro_expression_count": np.random.poisson(0.3),
            "micro_expression_intensity": np.random.normal(0.05, 0.03),
            "overall_facial_tension": np.random.normal(0.2, 0.1),
            "nervous_movement_score": np.random.normal(0.1, 0.07),
            "jaw_clench_indicator": np.random.normal(0.1, 0.05),
            "label": 0,  # Truthful
        }
        data.append(sample)

    # Generate deceptive samples (shifted distributions)
    for _ in range(n_deception):
        sample = {
            "blink_rate": np.random.normal(28, 8),
            "blink_rate_change": np.random.normal(1.6, 0.4),
            "ear_mean": np.random.normal(0.27, 0.04),
            "ear_std": np.random.normal(0.04, 0.015),
            "gaze_horizontal_std": np.random.normal(0.3, 0.12),
            "gaze_vertical_std": np.random.normal(0.2, 0.1),
            "gaze_fixation_ratio": np.random.normal(0.55, 0.15),
            "head_pitch_std": np.random.normal(3.5, 1.5),
            "head_yaw_std": np.random.normal(3.0, 1.2),
            "head_roll_std": np.random.normal(1.8, 0.8),
            "head_movement_magnitude": np.random.normal(1.5, 0.7),
            "face_symmetry_score": np.random.normal(0.78, 0.08),
            "face_symmetry_change": np.random.normal(0.08, 0.04),
            "lip_movement_intensity": np.random.normal(5.0, 2.5),
            "lip_compression_ratio": np.random.normal(0.25, 0.1),
            "eyebrow_tension": np.random.normal(0.5, 0.2),
            "eyebrow_asymmetry": np.random.normal(0.15, 0.08),
            "micro_expression_count": np.random.poisson(1.5),
            "micro_expression_intensity": np.random.normal(0.3, 0.15),
            "overall_facial_tension": np.random.normal(0.55, 0.18),
            "nervous_movement_score": np.random.normal(0.4, 0.15),
            "jaw_clench_indicator": np.random.normal(0.35, 0.12),
            "label": 1,  # Deceptive
        }
        data.append(sample)

    df = pd.DataFrame(data)
    # Clip to valid ranges
    for col in features:
        df[col] = df[col].clip(lower=0)

    return df


def train_model(model_type: str = "random_forest", data_path: str = None):
    """Train and save the deception detection model."""
    print(f"\n{'='*60}")
    print(f" Training {model_type.upper()} Model")
    print(f"{'='*60}\n")

    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    elif os.path.exists("data/real_features.csv"):
        print("Found real dataset! Loading data from data/real_features.csv...")
        df = pd.read_csv("data/real_features.csv")
    else:
        print("No real dataset found. Generating synthetic training data...")
        df = generate_synthetic_dataset(2000)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sample_features.csv", index=False)
        print(f"Saved synthetic dataset to data/sample_features.csv")

    # Filter low-quality flagged samples if quality_flag column exists
    if "quality_flag" in df.columns:
        n_before = len(df)
        df = df[df["quality_flag"] == "OK"]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            print(f"[FILTER] Excluded {n_filtered} low-quality/flagged samples from training.")

    # Validate binary labels (0=Truth, 1=Deception)
    if "label" not in df.columns:
        print("[ERROR] Dataset missing 'label' column.")
        return None
    valid_mask = df["label"].isin([0, 1])
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        print(f"[FILTER] Excluded {n_invalid} rows with invalid/unverified labels.")
        df = df[valid_mask]

    features = config.FEATURE_NAMES
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"[ERROR] Missing required features: {missing_feats}")
        return None

    # Filter NaNs or infinite values
    clean_mask = ~df[features].isna().any(axis=1) & ~np.isinf(df[features]).any(axis=1)
    if not clean_mask.all():
        n_nan = int((~clean_mask).sum())
        print(f"[FILTER] Excluded {n_nan} rows containing NaN or infinite feature values.")
        df = df[clean_mask]

    if len(df) == 0:
        print("[ERROR] No valid samples remaining for training.")
        return None

    X = df[features].values
    y = df["label"].values.astype(int)

    if len(np.unique(y)) < 2:
        print(f"[ERROR] Training requires both Truth (0) and Deception (1) classes. Found: {np.unique(y)}")
        return None

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: Truth={int(sum(y==0))}, Deception={int(sum(y==1))}")

    # ── PROVENANCE & MIXED DATA CHECK ───────────────────────────────────────
    if "data_provenance" in df.columns:
        provs = df["data_provenance"].dropna().unique()
        if len(provs) > 1:
            print(f"[ERROR] Mixed data provenance detected: {list(provs)}.")
            print("        Synthetic and human data CANNOT be silently mixed. Training aborted.")
            return None

    data_src = data_path or ("data/real_features.csv" if os.path.exists("data/real_features.csv") else "data/sample_features.csv")
    is_synthetic = "sample_features" in data_src.lower() or "data_provenance" not in df.columns or (df.get("data_provenance") == "SYNTHETIC").all()

    import hashlib
    dataset_sha256 = None
    if os.path.exists(data_src):
        try:
            with open(data_src, "rb") as f_hash:
                dataset_sha256 = hashlib.sha256(f_hash.read()).hexdigest()
        except Exception:
            dataset_sha256 = None

    label_protocol = str(df["label_protocol"].iloc[0]) if "label_protocol" in df.columns else (
        "SYNTHETIC_SIMULATION" if is_synthetic else "UNKNOWN"
    )

    label_verified_val = None
    if "label_verified" in df.columns:
        label_verified_val = bool(df["label_verified"].all())

    # ── TRAINING POLICY: LABEL VERIFICATION & MODEL CATEGORIZATION ─────────────
    # If label_verified=False, the pipeline MUST NOT describe the model as trained
    # on verified deception labels. It is strictly an experimental task-classification model.
    if is_synthetic:
        model_category = "SYNTHETIC_BENCHMARK"
        intended_interpretation = (
            "Measures mathematical separation of simulated synthetic Gaussian distributions. "
            "Cannot be interpreted as human behavior or deception detection."
        )
        limitations = [
            "Trained on simulated synthetic distributions; does not reflect real human behavioral variance",
            "Cannot be used to assess human honesty, truthfulness, or deception",
            "Prohibited for punitive, screening, ranking, or consequential decision-making",
            "Lighting, head movement, and camera angle introduce signal variance"
        ]
        print()
        print("!" * 70)
        print("  SYNTHETIC TRAINING DATA IN USE")
        print("  Labels were generated programmatically, NOT from observed deception.")
        print("  Model Category: SYNTHETIC_BENCHMARK")
        print("  This model cannot be claimed to detect deception in real humans.")
        print("  To train on real data: run  python training/collect_data.py")
        print("!" * 70)
        print()
    elif label_verified_val is False:
        model_category = "EXPERIMENTAL_TASK_CLASSIFICATION"
        intended_interpretation = (
            "Measures behavioral divergence between instructed experimental conditions (e.g., prompt compliance). "
            "Reflects task-directed compliance differences, NOT verified lie detection."
        )
        limitations = [
            "Labels represent instructed experimental tasks (label_verified=False), NOT verified deception",
            "Non-deceptive emotional stress, cognitive load, or camera anxiety can trigger behavioral anomalies",
            "Lighting, head movement, and camera angle introduce signal variance",
            "Prohibited for punitive, screening, ranking, or consequential decision-making"
        ]
        print()
        print("*" * 70)
        print("  POLICY: EXPERIMENTAL TASK-CLASSIFICATION MODEL")
        print("  Dataset observations have label_verified=False.")
        print("  Model learns behavioral differences between instructed task conditions.")
        print("  It will NOT be described or used as a verified deception-detection model.")
        print("*" * 70)
        print()
    elif label_verified_val is True:
        model_category = "INDEPENDENTLY_VERIFIED_DECEPTION_DETECTION"
        intended_interpretation = "Trained on independently verified deception observations."
        limitations = [
            "Requires continuous calibration to individual baseline",
            "Prohibited for punitive, screening, ranking, or consequential decision-making"
        ]
    else:
        model_category = "UNVERIFIED_EXPERIMENTAL"
        intended_interpretation = "Experimental model trained on data with unverified label provenance."
        limitations = [
            "Label verification provenance is unconfirmed",
            "Prohibited for punitive, screening, ranking, or consequential decision-making"
        ]
    # ────────────────────────────────────────────────────────────────────────

    # Split data — Subject-Independent when participant_id is available
    has_groups = "participant_id" in df.columns and df["participant_id"].nunique() >= 2
    if has_groups:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["participant_id"].values))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_p = sorted(list(df["participant_id"].iloc[train_idx].unique()))
        test_p = sorted(list(df["participant_id"].iloc[test_idx].unique()))
        split_type = "SUBJECT_INDEPENDENT_GROUP_SPLIT"
        print(f"[SPLIT] Grouped by participant ({len(train_p)} train, {len(test_p)} test) to prevent leakage:")
        print(f"        Train participants: {train_p}")
        print(f"        Test participants : {test_p}")
    else:
        split_type = "STRATIFIED_RANDOM"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_p, test_p = [], []

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select model
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "svm": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, random_state=42
        ),
        "logistic": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "neural_net": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=500, random_state=42, early_stopping=True
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
    }

    if model_type not in models:
        print(f"Unknown model type: {model_type}")
        print(f"Available: {list(models.keys())}")
        return

    model = models[model_type]
    print(f"\nTraining {model_type}...")

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print(f"\n{'='*40}")
    print("EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Truthful", "Deceptive"]))

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        print(f"\nTop 10 Feature Importances:")
        importances = list(zip(features, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for name, imp in importances[:10]:
            bar = "#" * int(imp * 50)
            print(f"  {name:30s} {imp:.4f} {bar}")

    # Save model, scaler, and metadata
    os.makedirs("models", exist_ok=True)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    from datetime import datetime
    import json
    sessions_count = int(df["session_id"].nunique()) if "session_id" in df.columns else 0
    participants_count = int(df["participant_id"].nunique()) if "participant_id" in df.columns else 0

    metadata = {
        "model_type": model_type,
        "model_category": model_category,
        "data_source": data_src,
        "data_provenance": "SYNTHETIC" if is_synthetic else "REAL_HUMAN",
        "dataset_sha256": dataset_sha256,
        "random_seed": 42,
        "split_type": split_type,
        "label_protocol": label_protocol,
        "label_verified": label_verified_val,
        "intended_interpretation": intended_interpretation,
        "limitations": limitations,
        "unique_participants_count": participants_count,
        "unique_sessions_count": sessions_count,
        "train_participants": train_p,
        "test_participants": test_p,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": datetime.now().isoformat()
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {config.MODEL_PATH}")
    print(f"Scaler saved to {config.SCALER_PATH}")
    print(f"Metadata saved to models/model_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deception detection model")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "svm", "logistic", "neural_net", "gradient_boost"],
                        help="Model type to train")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data CSV")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic dataset only")
    args = parser.parse_args()

    if args.generate_data:
        df = generate_synthetic_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sample_features.csv", index=False)
        print(f"Generated {len(df)} samples -> data/sample_features.csv")
    else:
        train_model(args.model, args.data)
