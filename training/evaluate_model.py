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
import hashlib
from datetime import datetime
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


def compute_cluster_bootstrap_ci(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                                 cluster_col: str = "participant_id", n_bootstraps: int = 500,
                                 confidence: float = 0.95) -> dict:
    """
    Cluster bootstrap confidence intervals (resampling at participant/cluster level).
    Accounts for intra-cluster correlation across repeated observations from the same subject.
    """
    if cluster_col not in df.columns:
        return None
    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    if n_clusters < 3:
        return None

    rng = np.random.RandomState(42)
    accs, precs, recs, f1s = [], [], [], []

    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = []
        for c in sampled_clusters:
            sampled_indices.extend(np.where(clusters == c)[0])

        yb_true = y_true[sampled_indices]
        yb_pred = y_pred[sampled_indices]
        if len(yb_true) == 0:
            continue
        accs.append(float(accuracy_score(yb_true, yb_pred)))

        tp = int(np.sum((yb_true == 1) & (yb_pred == 1)))
        fp = int(np.sum((yb_true == 0) & (yb_pred == 1)))
        fn = int(np.sum((yb_true == 1) & (yb_pred == 0)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_val = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1_val)

    alpha = (1.0 - confidence) / 2.0
    def _pcts(vals):
        if not vals:
            return (0.0, 0.0)
        return (float(np.percentile(vals, 100 * alpha)), float(np.percentile(vals, 100 * (1 - alpha))))

    return {
        "resampling_unit": "participant_cluster",
        "cluster_column": cluster_col,
        "n_clusters": int(n_clusters),
        "n_bootstraps": n_bootstraps,
        "accuracy_ci_95": _pcts(accs),
        "precision_ci_95": _pcts(precs),
        "recall_ci_95": _pcts(recs),
        "f1_ci_95": _pcts(f1s)
    }

def _detect_data_provenance(df: pd.DataFrame, data_path: str) -> dict:
    """
    Determine whether the dataset is synthetic, real human, or unknown.
    Returns a provenance dict with 'provenance' key: SYNTHETIC | REAL_HUMAN | UNKNOWN
    """
    path_lower = (data_path or "").lower()
    is_synth_col = "data_provenance" in df.columns and (df["data_provenance"] == "SYNTHETIC").all()
    is_real_col = "data_provenance" in df.columns and (df["data_provenance"] == "REAL_HUMAN").all()

    if "sample_features" in path_lower or is_synth_col:
        return {
            "provenance": "SYNTHETIC",
            "warning": (
                "DATA IS SYNTHETIC. Metrics reflect separation of "
                "programmatically generated Gaussian distributions, "
                "NOT real-world deception detection performance. "
                "Do not interpret these results as evidence that this model "
                "can detect deception in human subjects."
            )
        }
    if "real_features" in path_lower or "participant_id" in df.columns or is_real_col:
        return {
            "provenance": "REAL_HUMAN",
            "warning": (
                "Ground-truth deception labels in this dataset were assigned by "
                "experimental protocol (self-report or induced-deception task). "
                "Labels may not correspond to verified deceptive intent."
            )
        }
    return {
        "provenance": "UNKNOWN",
        "warning": (
            "Dataset provenance could not be determined. "
            "Verify that labels represent verified deception events before "
            "reporting these metrics as deception detection performance."
        )
    }


def evaluate(data_path: str = None,
             output_report_path: str = "training/evaluation_report.json",
             participant_col: str = "participant_id"):
    """Load model and run full scientific evaluation with provenance-aware safeguards."""
    print(f"\n{'='*70}")
    print(" BEHAVIORAL MODEL EVALUATION SUITE")
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
        data_source = "data/real_features.csv"
    elif os.path.exists("data/sample_features.csv"):
        df = pd.read_csv("data/sample_features.csv")
        data_source = "data/sample_features.csv"
    else:
        print("[ERROR] No evaluation data found. Run training/train_model.py or training/collect_data.py first.")
        return None

    # ── PROVENANCE & MIXED DATA CHECK ─────────────────────────────────────────
    if "data_provenance" in df.columns:
        provs = df["data_provenance"].dropna().unique()
        if len(provs) > 1:
            print(f"[ERROR] Mixed data provenance detected: {list(provs)}.")
            print("        Synthetic and human data CANNOT be silently mixed. Evaluation aborted.")
            return None

    dataset_sha256 = None
    if data_source and os.path.exists(data_source):
        try:
            with open(data_source, "rb") as f_hash:
                dataset_sha256 = hashlib.sha256(f_hash.read()).hexdigest()
        except Exception:
            dataset_sha256 = None

    provenance_info = _detect_data_provenance(df, data_source)
    provenance = provenance_info["provenance"]
    provenance_warning = provenance_info["warning"]

    print(f"Data Source:       {data_source}")
    print(f"Data Provenance:   {provenance}")
    if dataset_sha256:
        print(f"Dataset SHA-256:   {dataset_sha256[:16]}...")
    if provenance == "SYNTHETIC":
        print()
        print("!" * 70)
        print("  WARNING: SYNTHETIC DATA DETECTED")
        print(f"  {provenance_warning}")
        print("!" * 70)
        print()
    else:
        print(f"  Note: {provenance_warning}")
        print()
    # ──────────────────────────────────────────────────────────────────────────

    label_protocol = str(df["label_protocol"].iloc[0]) if "label_protocol" in df.columns else (
        "SYNTHETIC_SIMULATION" if provenance == "SYNTHETIC" else "UNKNOWN"
    )

    print("PERFORMANCE EVALUATION SCOPES:")
    print(f"  Synthetic Benchmark:                {'EVALUATED' if provenance == 'SYNTHETIC' else 'NOT_APPLICABLE'}")
    print(f"  Controlled Task Classification:     {'EVALUATED' if provenance == 'REAL_HUMAN' else 'NOT_APPLICABLE'}")
    print("  Validated Deception Detection:      NOT ESTABLISHED")
    print()

    # Check label verification status
    label_verified_val = None
    label_verification_status = "NOT_SPECIFIED"
    if "label_verified" in df.columns:
        all_verified = bool(df["label_verified"].all())
        label_verified_val = all_verified
        if not all_verified:
            label_verification_status = "UNVERIFIED_TASK_DIRECTED_LABELS (INTENT_NOT_BIOLOGICALLY_VERIFIED)"
            print("[SAFEGUARD] Dataset observations have label_verified=False.")
            print("            Labels indicate task-prompt compliance only; intent is NOT verified.")
        else:
            label_verification_status = "INDEPENDENTLY_VERIFIED"

    features = config.FEATURE_NAMES
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"[ERROR] Dataset missing required features: {missing_feats}")
        return None

    # Track raw rows and quality exclusions
    total_raw_rows = len(df)
    n_quality_non_ok = int((df["quality_flag"] != "OK").sum()) if "quality_flag" in df.columns else 0
    nan_rows = int(df[features].isna().any(axis=1).sum())
    quality_dist = df["quality_flag"].value_counts().to_dict() if "quality_flag" in df.columns else {}
    n_unique_participants = int(df[participant_col].nunique()) if participant_col in df.columns else None
    n_unique_sessions = int(df["session_id"].nunique()) if "session_id" in df.columns else None

    # Validate labels
    if "label" not in df.columns:
        print("[ERROR] Dataset missing 'label' column. Evaluation aborted.")
        return None

    # Exclude NaN or invalid labels
    valid_mask = df["label"].isin([0, 1])
    if not valid_mask.any():
        print("[ERROR] No valid binary labels (0/1) found in dataset. Evaluation aborted.")
        return None
    n_invalid_labels = 0
    if not valid_mask.all():
        n_invalid_labels = int((~valid_mask).sum())
        print(f"[WARN] Excluding {n_invalid_labels} rows with invalid/unverified labels.")
        df = df[valid_mask].copy()

    # Clean NaNs in features for evaluation
    clean_mask = ~df[features].isna().any(axis=1) & ~np.isinf(df[features]).any(axis=1)
    n_nan_inf_dropped = 0
    if not clean_mask.all():
        n_nan_inf_dropped = int((~clean_mask).sum())
        print(f"[WARN] Dropping {n_nan_inf_dropped} rows containing NaN or infinite feature values.")
        df = df[clean_mask].copy()

    if len(df) == 0:
        print("[ERROR] No valid rows remaining for evaluation.")
        return None


    X = df[features].values
    y = df["label"].values.astype(int)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # ── SUBJECT-INDEPENDENT CV (if participant_id column present) ──────────────
    logo_cv_results = None
    has_participant_col = participant_col in df.columns
    if has_participant_col and provenance == "REAL_HUMAN":
        from sklearn.model_selection import LeaveOneGroupOut
        groups = df[participant_col].values
        n_participants = len(np.unique(groups))
        if n_participants >= 3:
            logo = LeaveOneGroupOut()
            logo_accs, logo_bal_accs = [], []
            for train_idx, test_idx in logo.split(X_scaled, y, groups):
                _yt, _yv = y[train_idx], y[test_idx]
                _Xt, _Xv = X_scaled[train_idx], X_scaled[test_idx]
                # Check that training fold has both classes
                if len(np.unique(_yt)) >= 2:
                    model.fit(_Xt, _yt)
                    _yp = model.predict(_Xv)
                    logo_accs.append(float(accuracy_score(_yv, _yp)))
                    logo_bal_accs.append(float(balanced_accuracy_score(_yv, _yp)))
            if logo_accs:
                logo_cv_results = {
                    "n_participants": n_participants,
                    "logo_accuracy_mean": float(np.mean(logo_accs)),
                    "logo_accuracy_std": float(np.std(logo_accs)),
                    "logo_balanced_accuracy_mean": float(np.mean(logo_bal_accs)),
                    "logo_balanced_accuracy_std": float(np.std(logo_bal_accs)),
                }
                print(f"Subject-Independent LOGO CV ({n_participants} participants):")
                print(f"  Accuracy     : {logo_cv_results['logo_accuracy_mean']:.4f} "
                      f"(+/- {logo_cv_results['logo_accuracy_std']:.4f})")
                print(f"  Balanced Acc : {logo_cv_results['logo_balanced_accuracy_mean']:.4f} "
                      f"(+/- {logo_cv_results['logo_balanced_accuracy_std']:.4f})")
                print()
                # Re-fit on full data for subsequent metrics
                model.fit(X_scaled, y)
        else:
            print(f"[WARN] Only {n_participants} unique participants — LOGO CV skipped (need >= 3).")
            logo_cv_results = {
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": f"Only {n_participants} unique participants (minimum 3 required for cross-validation)"
            }
    # ──────────────────────────────────────────────────────────────────────────

    # Cluster Bootstrap CI vs Wilson Score CI
    cluster_ci_results = None
    if has_participant_col and n_unique_participants and n_unique_participants >= 3:
        cluster_ci_results = compute_cluster_bootstrap_ci(df, y, y_pred, cluster_col=participant_col)

    # Core Metrics
    total_samples = len(y)
    acc = float(accuracy_score(y, y_pred))
    classes = np.unique(y)
    is_multi_class = len(classes) >= 2

    bal_acc = float(balanced_accuracy_score(y, y_pred)) if is_multi_class else 1.0
    kappa = float(cohen_kappa_score(y, y_pred)) if is_multi_class else 0.0

    # Calibration only when assumptions and sample sizes are adequate (N >= 50 and 2 classes)
    if total_samples >= 50 and is_multi_class:
        brier = float(brier_score_loss(y, y_proba))
        ece = compute_expected_calibration_error(y, y_proba)
        calibration_status = "VALID (N >= 50, binary classes)"
    else:
        brier = None
        ece = None
        calibration_status = "NOT_APPLICABLE (insufficient sample size N < 50 or single class present)"

    # Discrimination metrics
    if is_multi_class:
        roc_auc = float(roc_auc_score(y, y_proba))
        pr_auc = float(average_precision_score(y, y_proba))
    else:
        roc_auc = None
        pr_auc = None

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
    if is_multi_class and min(np.sum(y == 0), np.sum(y == 1)) >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
        cv_bal_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="balanced_accuracy")
        cv_acc_mean = float(cv_scores.mean())
        cv_acc_std = float(cv_scores.std())
        cv_bal_mean = float(cv_bal_scores.mean())
        cv_bal_std = float(cv_bal_scores.std())
    else:
        cv_acc_mean, cv_acc_std = None, None
        cv_bal_mean, cv_bal_std = None, None

    # Output Presentation
    print(f"Data Source:       {data_source}")
    print(f"Data Provenance:   {provenance}")
    print(f"Sample Count:      {total_samples} (Truthful={int(np.sum(y==0))}, Deceptive={int(np.sum(y==1))})")
    if n_unique_participants is not None:
        print(f"Unique Participants:{n_unique_participants}")
    if n_unique_sessions is not None:
        print(f"Unique Sessions:   {n_unique_sessions}")
    print(f"-"*70)

    if provenance == "SYNTHETIC":
        print("  NOTE: The metrics below measure synthetic-distribution separation ONLY.")
        print("  HUMAN DECEPTION DETECTION PERFORMANCE: NOT ESTABLISHED")
        print(f"-"*70)

    print("PRIMARY PERFORMANCE METRICS (Point Estimate & 95% Wilson CI):")
    print(f"  Accuracy:          {acc:.4f}  [95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f}]")
    if cluster_ci_results:
        c_acc = cluster_ci_results["accuracy_ci_95"]
        print(f"  Cluster Acc (95%): [{c_acc[0]:.4f} - {c_acc[1]:.4f}] (resampled by participant)")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"  Precision:         {precision:.4f}  [95% CI: {prec_ci[0]:.4f} - {prec_ci[1]:.4f}]")
    print(f"  Recall (TPR):      {tpr:.4f}  [95% CI: {tpr_ci[0]:.4f} - {tpr_ci[1]:.4f}]")
    print(f"  Specificity (TNR): {tnr:.4f}  [95% CI: {tnr_ci[0]:.4f} - {tnr_ci[1]:.4f}]")
    print(f"  F1-Score:          {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:           {roc_auc:.4f}")
        print(f"  PR-AUC:            {pr_auc:.4f}")
    else:
        print("  ROC-AUC / PR-AUC:  UNDEFINED (single class in dataset)")
    print(f"-"*70)
    print("CALIBRATION & PROBABILITY RELIABILITY:")
    if brier is not None:
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  Expected Calib Err:{ece:.4f}")
    else:
        print(f"  Calibration:       {calibration_status}")
    print(f"-"*70)
    if cv_acc_mean is not None:
        print("CROSS-VALIDATION GENERALIZATION (5-Fold Stratified):")
        print(f"  Mean Accuracy:     {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")
        print(f"  Mean Bal. Accuracy:{cv_bal_mean:.4f} (+/- {cv_bal_std:.4f})")
        print(f"-"*70)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives (Truthful as Truthful):   {tn}")
    print(f"  False Positives (Truthful as Deceptive): {fp}  (FPR = {fpr:.4f})")
    print(f"  False Negatives (Deceptive as Truthful): {fn}  (FNR = {fnr:.4f})")
    print(f"  True Positives (Deceptive as Deceptive): {tp}")

    # Threshold Operating Sweep
    threshold_results = []
    if is_multi_class:
        print(f"\nDECISION THRESHOLD OPERATING CHARACTERISTICS:")
        print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'FPR':>10} {'F1':>10}")
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
            "data_provenance": provenance,
            "dataset_sha256": dataset_sha256,
            "random_seed": 42,
            "evaluation_timestamp": datetime.now().isoformat(),
            "synthetic_warning": provenance_warning,
            "human_performance": "NOT ESTABLISHED" if provenance == "SYNTHETIC" else "MEASURED_ON_TASK",
            "human_deception_detection_accuracy": "NOT ESTABLISHED" if provenance == "SYNTHETIC" else "REQUIRES_INDEPENDENT_STUDY",
            "sample_count": total_samples,
            "truth_count": int(np.sum(y == 0)),
            "deception_count": int(np.sum(y == 1)),
            "unique_participants": n_unique_participants,
            "unique_sessions": n_unique_sessions,
            "nan_rows_count": nan_rows,
            "quality_flag_distribution": quality_dist,
            "has_participant_column": has_participant_col,
            "subject_independent_cv": logo_cv_results,
            "cluster_bootstrap": cluster_ci_results,
            "class_diversity_status": "SUFFICIENT (binary classes)" if is_multi_class else f"INSUFFICIENT (single class: {classes[0]})",
            "sample_size_status": "SUFFICIENT (N >= 50)" if total_samples >= 50 else f"INSUFFICIENT (N = {total_samples} < 50)",
            "label_verified": label_verified_val,
            "label_verification_status": label_verification_status,
            "data_exclusions": {
                "total_raw_rows": total_raw_rows,
                "quality_flag_non_ok": n_quality_non_ok,
                "invalid_labels_excluded": n_invalid_labels,
                "nan_inf_rows_excluded": n_nan_inf_dropped,
                "final_evaluated_rows": total_samples
            },
            "reproducibility": {
                "command": f"python training/evaluate_model.py --data {data_source} --out {output_report_path} --participant-col {participant_col}",
                "random_seed": 42,
                "dataset_sha256": dataset_sha256
            },
            "label_protocol": label_protocol,
            "synthetic_benchmark_performance": "EVALUATED" if provenance == "SYNTHETIC" else "NOT_APPLICABLE",
            "controlled_task_classification_performance": "EVALUATED" if provenance == "REAL_HUMAN" else "NOT_APPLICABLE",
            "independently_validated_deception_detection": "NOT ESTABLISHED",
            "performance_scopes": {
                "synthetic_benchmark": {
                    "status": "EVALUATED" if provenance == "SYNTHETIC" else "NOT_APPLICABLE",
                    "description": "Measures mathematical separation of simulated synthetic Gaussian distributions."
                },
                "controlled_task_classification": {
                    "status": "EVALUATED" if provenance == "REAL_HUMAN" else "NOT_APPLICABLE",
                    "description": "Measures behavioral divergence between instructed experimental conditions (e.g. prompt compliance)."
                },
                "independently_validated_deception_detection": {
                    "status": "NOT ESTABLISHED",
                    "reason": (
                        "Synthetic data cannot establish human deception detection"
                        if provenance == "SYNTHETIC"
                        else (
                            "Experimental labels are task-directed instructions (label_verified=False); ground-truth deception intent was not independently verified"
                            if label_verified_val is False
                            else "Requires multi-site independent study with biological/forensic ground truth"
                        )
                    ),
                    "description": "No claim of human lie detection accuracy is supported by this dataset or model."
                }
            },
            "disclaimer": config.DISCLAIMER,
            "deception_detection_validity": (
                "CANNOT BE ESTABLISHED — single class present"
                if not is_multi_class
                else (
                    "CANNOT BE ESTABLISHED — synthetic labels"
                    if provenance == "SYNTHETIC"
                    else (
                        "CANNOT BE ESTABLISHED — task instructions only (label_verified=False); intent not biologically verified"
                        if label_verified_val is False
                        else "REQUIRES INDEPENDENT VERIFICATION"
                    )
                )
            )
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
            "calibration_status": calibration_status,
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_balanced_accuracy_mean": cv_bal_mean,
            "cv_balanced_accuracy_std": cv_bal_std
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

    print(f"\n[SAVED] Evaluation report exported to {output_report_path}")
    if provenance == "SYNTHETIC":
        print("!" * 70)
        print("  SYNTHETIC DATA LIMITATION REMINDER")
        print("  These metrics measure synthetic-distribution separation ONLY.")
        print("  They do NOT establish deception detection performance in humans.")
        print("!" * 70)
    print("\n" + "="*70)
    print(" SCIENTIFIC VALIDATION NOTICE:")
    print(" " + config.DISCLAIMER)
    print("="*70 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate behavioral model with provenance-aware safeguards")
    parser.add_argument("--data", type=str, default=None, help="Path to evaluation data CSV")
    parser.add_argument("--out", type=str, default="training/evaluation_report.json", help="Report JSON destination")
    parser.add_argument("--participant-col", type=str, default="participant_id",
                        help="Column name for participant IDs (enables subject-independent LOGO CV)")
    args = parser.parse_args()
    evaluate(args.data, args.out, args.participant_col)
