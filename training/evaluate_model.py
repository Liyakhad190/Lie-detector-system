"""
============================================================================
 TESTING / EVALUATION SCRIPT
============================================================================
 Evaluates the trained deception detection model with comprehensive metrics.
 
 Usage:
   python training/evaluate_model.py
   python training/evaluate_model.py --data data/sample_features.csv
============================================================================
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate(data_path: str = None):
    """Load model and run full evaluation."""
    print(f"\n{'='*60}")
    print(" MODEL EVALUATION")
    print(f"{'='*60}\n")

    # Load model
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Model not found at {config.MODEL_PATH}")
        print("Run: python training/train_model.py first")
        return

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"Model type: {type(model).__name__}")

    # Load data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    elif os.path.exists("data/sample_features.csv"):
        df = pd.read_csv("data/sample_features.csv")
    else:
        print("[ERROR] No evaluation data found. Generate data first.")
        return

    features = config.FEATURE_NAMES
    X = df[features].values
    y = df["label"].values

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Print results
    print(f"\nDataset: {len(X)} samples")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y, y_proba):.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Truthful", "Deceptive"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Per-threshold analysis
    print(f"\nThreshold Analysis:")
    print(f"{'Threshold':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_t = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y, y_t)
        tp = np.sum((y_t == 1) & (y == 1))
        fp = np.sum((y_t == 1) & (y == 0))
        fn = np.sum((y_t == 0) & (y == 1))
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        print(f"  {thresh:>10.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f}")

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deception model")
    parser.add_argument("--data", type=str, default=None, help="Path to test data CSV")
    args = parser.parse_args()
    evaluate(args.data)
