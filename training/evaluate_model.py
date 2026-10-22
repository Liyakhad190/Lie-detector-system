"""
============================================================================
 TESTING / EVALUATION SCRIPT (ADVANCED SCIENTIFIC METRICS)
============================================================================
 Evaluates the trained deception detection model with comprehensive metrics,
 statistical confidence intervals, calibration error (ECE), and Brier score.
 
 Supported Metrics:
   - Accuracy, Balanced Accuracy, Cohen's Kappa
   - Precision, Recall, F1-Score, Specificity
   - ROC-AUC and Precision-Recall AUC (PR-AUC)
   - Brier Score (probability accuracy) & Expected Calibration Error (ECE)
   - Wilson Score 95% Confidence Intervals
   - Per-Threshold Operating Characteristic Analysis
   - Structured JSON Report Export
============================================================================
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, brier_score_loss,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Compute Wilson score interval for binomial proportion.
    More reliable than normal approximation, especially near 0 or 1.
    """
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / total
    denominator = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denominator
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * total**2))) / denominator
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper


def compute_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures the difference between predicted probabilities and empirical accuracy.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_proba >= bin_lower) & (y_proba < bin_upper if i < n_bins - 1 else y_proba <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(data_path: str = None, output_report_path: str = "training/evaluation_report.json"):
    """Load model and run full scientific evaluation."""
    print(f"\n{'='*70}")
    print(" SCIENTIFIC DECEPTION MODEL EVALUATION SUITE")
    print(f"{'='*70}\n")

    # Load model and scaler
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Model not found at {config.MODEL_PATH}")
        print("Run: python training/train_model.py first")
        return None

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"Model Architecture: {type(model).__name__}")
    print(f"Feature Count:     {len(config.FEATURE_NAMES)}")

    # Load evaluation data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        data_source = data_path
    elif os.path.exists("data/real_features.csv"):
        df = pd.read_csv("data/real_features.csv")
        data_source = "data/real_features.csv (Real Human Dataset)"
    elif os.path.exists("data/sample_features.csv"):
        df = pd.read_csv("data/sample_features.csv")
        data_source = "data/sample_features.csv (Synthetic Behavioral Dataset)"
    else:
        print("[ERROR] No evaluation data found. Run training/train_model.py or training/collect_data.py first.")
        return None

    features = config.FEATURE_NAMES
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"[ERROR] Dataset missing required features: {missing_feats}")
        return None

    X = df[features].values
    y = df["label"].values.astype(int)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Core Metrics
    total_samples = len(y)
    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    kappa = float(cohen_kappa_score(y, y_pred))
    brier = float(brier_score_loss(y, y_proba))
    ece = compute_expected_calibration_error(y, y_proba)
    roc_auc = float(roc_auc_score(y, y_proba))
    pr_auc = float(average_precision_score(y, y_proba))

    # Confusion matrix elements
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0 # Sensitivity / Recall
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0 # Specificity
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0

    # Wilson Score Confidence Intervals (95%)
    acc_ci = wilson_score_interval(int(np.sum(y == y_pred)), total_samples)
    tpr_ci = wilson_score_interval(int(tp), int(tp + fn))
    tnr_ci = wilson_score_interval(int(tn), int(tn + fp))
    prec_ci = wilson_score_interval(int(tp), int(tp + fp))

    # Cross-Validation with Stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
    cv_bal_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="balanced_accuracy")

    # Output Presentation
    print(f"Data Source:       {data_source}")
    print(f"Sample Count:      {total_samples} (Truthful={np.sum(y==0)}, Deceptive={np.sum(y==1)})")
    print(f"-"*70)
    print("PRIMARY PERFORMANCE METRICS (Point Estimate & 95% Wilson CI):")
    print(f"  Accuracy:          {acc:.4f}  [95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f}]")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}  (Agreement beyond chance)")
    print(f"  Precision:         {precision:.4f}  [95% CI: {prec_ci[0]:.4f} - {prec_ci[1]:.4f}]")
    print(f"  Recall (TPR):      {tpr:.4f}  [95% CI: {tpr_ci[0]:.4f} - {tpr_ci[1]:.4f}]")
    print(f"  Specificity (TNR): {tnr:.4f}  [95% CI: {tnr_ci[0]:.4f} - {tnr_ci[1]:.4f}]")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"-"*70)
    print("CALIBRATION & PROBABILITY RELIABILITY:")
    print(f"  Brier Score:       {brier:.4f}  (Lower is better, 0.0 = perfect calibration)")
    print(f"  Expected Calib Err:{ece:.4f}  (ECE, average gap between proba and reality)")
    print(f"-"*70)
    print("CROSS-VALIDATION GENERALIZATION (5-Fold Stratified):")
    print(f"  Mean Accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Mean Bal. Accuracy:{cv_bal_scores.mean():.4f} (+/- {cv_bal_scores.std():.4f})")
    print(f"-"*70)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives (Truthful as Truthful):   {tn}")
    print(f"  False Positives (Truthful as Deceptive): {fp}  (FPR = {fpr:.4f})")
    print(f"  False Negatives (Deceptive as Truthful): {fn}  (FNR = {fnr:.4f})")
    print(f"  True Positives (Deceptive as Deceptive): {tp}")

    # Threshold Operating Sweep
    print(f"\nDECISION THRESHOLD OPERATING CHARACTERISTICS:")
    print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'FPR':>10} {'F1':>10}")
    threshold_results = []
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_t = (y_proba >= thresh).astype(int)
        t_acc = float(accuracy_score(y, y_t))
        t_tp = int(np.sum((y_t == 1) & (y == 1)))
        t_fp = int(np.sum((y_t == 1) & (y == 0)))
        t_fn = int(np.sum((y_t == 0) & (y == 1)))
        t_tn = int(np.sum((y_t == 0) & (y == 0)))
        t_prec = float(t_tp / (t_tp + t_fp)) if (t_tp + t_fp) > 0 else 0.0
        t_rec = float(t_tp / (t_tp + t_fn)) if (t_tp + t_fn) > 0 else 0.0
        t_fpr = float(t_fp / (t_fp + t_tn)) if (t_fp + t_tn) > 0 else 0.0
        t_f1 = float(2 * t_prec * t_rec / (t_prec + t_rec)) if (t_prec + t_rec) > 0 else 0.0
        print(f"  {thresh:>10.2f} {t_acc:>10.4f} {t_prec:>10.4f} {t_rec:>10.4f} {t_fpr:>10.4f} {t_f1:>10.4f}")
        threshold_results.append({
            "threshold": thresh,
            "accuracy": t_acc,
            "precision": t_prec,
            "recall": t_rec,
            "fpr": t_fpr,
            "f1": t_f1
        })

    # Prepare export dictionary
    report = {
        "metadata": {
            "model_type": type(model).__name__,
            "data_source": data_source,
            "sample_count": total_samples,
            "truth_count": int(np.sum(y == 0)),
            "deception_count": int(np.sum(y == 1)),
            "disclaimer": config.DISCLAIMER
        },
        "metrics": {
            "accuracy": acc,
            "accuracy_ci_95": acc_ci,
            "balanced_accuracy": bal_acc,
            "cohen_kappa": kappa,
            "precision": precision,
            "precision_ci_95": prec_ci,
            "recall_tpr": tpr,
            "recall_ci_95": tpr_ci,
            "specificity_tnr": tnr,
            "specificity_ci_95": tnr_ci,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "expected_calibration_error": ece,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_balanced_accuracy_mean": float(cv_bal_scores.mean()),
            "cv_balanced_accuracy_std": float(cv_bal_scores.std())
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "threshold_analysis": threshold_results
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVED] Comprehensive scientific evaluation exported to {output_report_path}")
    print("\n" + "="*70)
    print(" SCIENTIFIC VALIDATION NOTICE:")
    print(" " + config.DISCLAIMER)
    print("="*70 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientifically evaluate deception model")
    parser.add_argument("--data", type=str, default=None, help="Path to test data CSV")
    parser.add_argument("--out", type=str, default="training/evaluation_report.json", help="Report JSON destination")
    args = parser.parse_args()
    evaluate(args.data, args.out)
