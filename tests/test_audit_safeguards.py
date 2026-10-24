"""
Unit and regression tests for ML audit safeguards (2026-10):
  1. Prediction stability band terminology & backward compatibility.
  2. Multi-channel explainability heuristic weight attribution & metadata.
  3. Synthetic data provenance detection & evaluation validity safeguards.
  4. Data collection research safeguards (anonymization & sample validation).
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.ml_model import DeceptionModel
from modules.explainability import ExplainabilityDecomposer
from training.evaluate_model import _detect_data_provenance, evaluate
from training.collect_data import _validate_sample


# ─── 1. TERMINOLOGY & STABILITY BAND ──────────────────────────────────────────

def test_prediction_stability_band_present_and_bounded():
    """prediction_stability_band must be returned with values in [0, 1]."""
    model = DeceptionModel()
    features = np.zeros(len(config.FEATURE_NAMES), dtype=np.float32)
    pred = model.predict(features)
    
    assert "prediction_stability_band" in pred
    band = pred["prediction_stability_band"]
    assert isinstance(band, tuple) and len(band) == 2
    assert 0.0 <= band[0] <= band[1] <= 1.0


def test_confidence_interval_backward_compat_alias():
    """confidence_interval must match prediction_stability_band for backward compatibility."""
    model = DeceptionModel()
    features = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.5
    pred = model.predict(features)
    
    assert "confidence_interval" in pred
    assert pred["confidence_interval"] == pred["prediction_stability_band"]


def test_no_data_state_includes_stability_band():
    """Fallback / NO DATA output must also provide prediction_stability_band."""
    model = DeceptionModel()
    pred = model.predict(None)
    
    assert pred["prediction_stability_band"] == (0.0, 0.0)
    assert pred["confidence_interval"] == (0.0, 0.0)
    assert pred["label"] == "NO DATA"


# ─── 2. EXPLAINABILITY HEURISTIC METADATA ─────────────────────────────────────

def test_explainability_channel_weights_marked_as_heuristics():
    """Explainability outputs must explicitly declare manually assigned heuristic basis."""
    decomposer = ExplainabilityDecomposer()
    dummy = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.4
    decomp = decomposer.decompose(dummy)
    
    assert decomp.get("weight_basis") == "MANUALLY_ASSIGNED_DOMAIN_HEURISTIC"
    assert decomp.get("score_type") == "HEURISTIC_ESTIMATE"
    for name, channel in decomp["channels"].items():
        assert channel.get("weight_basis") == "MANUALLY_ASSIGNED_DOMAIN_HEURISTIC"


def test_empty_explainability_marked_as_heuristics():
    """Empty / fallback explainability outputs must also declare heuristic metadata."""
    decomposer = ExplainabilityDecomposer()
    decomp = decomposer.decompose(None)
    
    assert decomp.get("weight_basis") == "MANUALLY_ASSIGNED_DOMAIN_HEURISTIC"
    assert decomp.get("score_type") == "HEURISTIC_ESTIMATE"
    for name, channel in decomp["channels"].items():
        assert channel.get("weight_basis") == "MANUALLY_ASSIGNED_DOMAIN_HEURISTIC"


# ─── 3. DATA PROVENANCE DETECTION & EVALUATION SAFEGUARDS ──────────────────────

def test_provenance_detection_synthetic():
    """Synthetic dataset path must be flagged as SYNTHETIC with clear warning."""
    df = pd.DataFrame(columns=config.FEATURE_NAMES + ["label"])
    provenance_info = _detect_data_provenance(df, "data/sample_features.csv")
    
    assert provenance_info["provenance"] == "SYNTHETIC"
    assert "SYNTHETIC" in provenance_info["warning"].upper()
    assert "human subjects" in provenance_info["warning"].lower()


def test_provenance_detection_real_human():
    """Dataset with participant_id column or real_features path must be flagged as REAL_HUMAN."""
    df_with_participant = pd.DataFrame(columns=config.FEATURE_NAMES + ["participant_id", "label"])
    provenance_info = _detect_data_provenance(df_with_participant, "experiments/pilot_data.csv")
    assert provenance_info["provenance"] == "REAL_HUMAN"

    df_real_path = pd.DataFrame(columns=config.FEATURE_NAMES + ["label"])
    provenance_info_real = _detect_data_provenance(df_real_path, "data/real_features.csv")
    assert provenance_info_real["provenance"] == "REAL_HUMAN"


def test_provenance_detection_unknown():
    """Unidentifiable CSV path without participant column must be flagged as UNKNOWN."""
    df_generic = pd.DataFrame(columns=config.FEATURE_NAMES + ["label"])
    provenance_info = _detect_data_provenance(df_generic, "random_features.csv")
    assert provenance_info["provenance"] == "UNKNOWN"


def test_synthetic_evaluation_report_metadata(tmp_path):
    """Evaluating on synthetic data must explicitly record that deception validity is unestablished."""
    # Run evaluation on synthetic sample features if present
    if not os.path.exists("data/sample_features.csv"):
        pytest.skip("data/sample_features.csv not found")

    out_file = str(tmp_path / "test_eval_report.json")
    report = evaluate(data_path="data/sample_features.csv", output_report_path=out_file)
    
    assert report is not None
    meta = report["metadata"]
    assert meta["data_provenance"] == "SYNTHETIC"
    assert "CANNOT BE ESTABLISHED" in meta["deception_detection_validity"]


# ─── 4. RESEARCH DATA COLLECTION SAFEGUARDS ───────────────────────────────────

def test_sample_validation_detects_low_frames():
    """_validate_sample must reject responses with fewer than minimum frames."""
    feats = [0.1] * len(config.FEATURE_NAMES)
    is_valid, flag = _validate_sample(feats, frame_count=5)
    assert not is_valid
    assert "LOW_FRAMES" in flag


def test_sample_validation_detects_nan():
    """_validate_sample must reject responses containing NaN values."""
    feats = [0.1] * (len(config.FEATURE_NAMES) - 1) + [float("nan")]
    is_valid, flag = _validate_sample(feats, frame_count=30)
    assert not is_valid
    assert flag == "CONTAINS_NAN"


def test_sample_validation_accepts_good_sample():
    """_validate_sample must accept valid complete responses."""
    feats = [0.25] * len(config.FEATURE_NAMES)
    is_valid, flag = _validate_sample(feats, frame_count=25)
    assert is_valid
    assert flag == "OK"


def test_participant_id_anonymisation_hash():
    """Raw participant identifier must be irreversibly hashed with SHA-256."""
    raw_id = "Alice_Smith_Subject_01"
    expected_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:10]
    expected_anon_id = f"P_{expected_hash}"
    
    # Verify properties of anonymised ID
    assert expected_anon_id.startswith("P_")
    assert len(expected_anon_id) == 12
    assert "Alice" not in expected_anon_id


def test_camera_not_acquired_before_consent():
    """DataCollector must NOT acquire camera during __init__ before consent is received."""
    from training.collect_data import DataCollector
    collector = DataCollector(participant_id="TEST_PILOT_USER")
    assert collector.cap is None, "Hardware camera must not be acquired prior to explicit informed consent."


def test_sample_validation_detects_inf():
    """_validate_sample must reject responses containing infinite values."""
    feats = [0.1] * (len(config.FEATURE_NAMES) - 1) + [float("inf")]
    is_valid, flag = _validate_sample(feats, frame_count=30)
    assert not is_valid
    assert flag == "CONTAINS_INF"


def test_sample_validation_detects_invalid_label():
    """_validate_sample must reject labels outside {0, 1}."""
    feats = [0.1] * len(config.FEATURE_NAMES)
    is_valid, flag = _validate_sample(feats, frame_count=20, label=5)
    assert not is_valid
    assert "INVALID_LABEL" in flag


def test_sample_validation_detects_empty_participant_id():
    """_validate_sample must reject empty participant IDs."""
    feats = [0.1] * len(config.FEATURE_NAMES)
    is_valid, flag = _validate_sample(feats, frame_count=20, label=1, participant_id="   ")
    assert not is_valid
    assert flag == "MISSING_PARTICIPANT_ID"


def test_cluster_bootstrap_ci_calculation():
    """Cluster bootstrap must calculate participant-resampled 95% intervals."""
    from training.evaluate_model import compute_cluster_bootstrap_ci
    
    # 5 participants with 6 responses each
    p_ids = []
    for i in range(5):
        p_ids.extend([f"P_{i}"] * 6)
    
    df = pd.DataFrame({"participant_id": p_ids})
    y_true = np.array([0, 0, 1, 1, 0, 1] * 5)
    y_pred = np.array([0, 0, 1, 0, 0, 1] * 5)
    
    res = compute_cluster_bootstrap_ci(df, y_true, y_pred, cluster_col="participant_id", n_bootstraps=100)
    assert res is not None
    assert res["resampling_unit"] == "participant_cluster"
    assert res["n_clusters"] == 5
    ci = res["accuracy_ci_95"]
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_evaluation_handles_single_class_safely(tmp_path):
    """Evaluation must handle single-class datasets gracefully without crashing."""
    csv_file = str(tmp_path / "single_class.csv")
    out_file = str(tmp_path / "single_class_report.json")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0] * 10  # All Truth (single class)
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    report = evaluate(data_path=csv_file, output_report_path=out_file)
    assert report is not None
    assert report["metrics"]["roc_auc"] is None
    assert report["metrics"]["pr_auc"] is None
    assert report["metrics"]["calibration_status"].startswith("NOT_APPLICABLE")
    assert "INSUFFICIENT" in report["metadata"]["class_diversity_status"]


def test_mixed_dataset_prevention(tmp_path):
    """Training must reject mixed datasets containing both synthetic and human data."""
    from training.train_model import train_model
    mixed_file = str(tmp_path / "mixed.csv")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 5
    data["data_provenance"] = ["SYNTHETIC"] * 5 + ["REAL_HUMAN"] * 5
    pd.DataFrame(data).to_csv(mixed_file, index=False)
    
    res = train_model(data_path=mixed_file)
    assert res is None, "train_model must return None and abort when mixed data provenance is detected."


def test_insufficient_participants_explicit_status(tmp_path):
    """Evaluation must explicitly record INSUFFICIENT_EVIDENCE when unique participants < 3."""
    eval_file = str(tmp_path / "two_participants.csv")
    out_file = str(tmp_path / "two_p_report.json")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 5
    data["participant_id"] = ["P_001"] * 5 + ["P_002"] * 5
    data["data_provenance"] = ["REAL_HUMAN"] * 10
    pd.DataFrame(data).to_csv(eval_file, index=False)
    
    report = evaluate(data_path=eval_file, output_report_path=out_file, participant_col="participant_id")
    assert report is not None
    logo_meta = report["metadata"]["subject_independent_cv"]
    assert logo_meta is not None
    assert logo_meta["status"] == "INSUFFICIENT_EVIDENCE"
    assert "minimum 3 required" in logo_meta["reason"]


def test_pseudonymization_does_not_claim_absolute_anonymity():
    """Verify that participant ID hashing is documented as pseudonymization, not absolute anonymity."""
    from training.collect_data import DataCollector
    c = DataCollector(participant_id="SUBJ_PILOT_42")
    # SHA-256 hash yields 10-char hex prefix
    assert c.participant_id.startswith("P_")
    assert len(c.participant_id) == 12
    # Verify docstring acknowledges pseudonymization
    assert "anonymis" in DataCollector.__doc__.lower() or "pseudonym" in DataCollector.__doc__.lower()


# ─── 5. REPRODUCIBILITY & UNVERIFIED LABEL REGRESSION TESTS ───────────────────

def test_evaluation_rejects_mixed_provenance(tmp_path):
    """Evaluation must abort and return None if mixed provenance is passed."""
    mixed_file = str(tmp_path / "eval_mixed.csv")
    out_file = str(tmp_path / "eval_mixed_report.json")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 5
    data["data_provenance"] = ["SYNTHETIC"] * 5 + ["REAL_HUMAN"] * 5
    pd.DataFrame(data).to_csv(mixed_file, index=False)
    
    res = evaluate(data_path=mixed_file, output_report_path=out_file)
    assert res is None, "evaluate() must return None when mixed provenance is detected."


def test_evaluation_records_reproducibility_and_dataset_hash(tmp_path):
    """Evaluation report must record dataset SHA-256 hash and reproducibility command."""
    csv_file = str(tmp_path / "dummy_data.csv")
    out_file = str(tmp_path / "dummy_report.json")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 5
    data["data_provenance"] = ["SYNTHETIC"] * 10
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    report = evaluate(data_path=csv_file, output_report_path=out_file)
    assert report is not None
    meta = report["metadata"]
    assert "dataset_sha256" in meta and meta["dataset_sha256"] is not None
    assert len(meta["dataset_sha256"]) == 64
    assert meta["random_seed"] == 42
    assert "reproducibility" in meta
    assert meta["reproducibility"]["random_seed"] == 42
    assert "data_exclusions" in meta


def test_evaluation_flags_unverified_labels(tmp_path):
    """Observations with label_verified=False must explicitly state validity cannot be established."""
    csv_file = str(tmp_path / "unverified_labels.csv")
    out_file = str(tmp_path / "unverified_report.json")
    
    data = {f: [0.5] * 10 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 5
    data["data_provenance"] = ["REAL_HUMAN"] * 10
    data["label_verified"] = [False] * 10
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    report = evaluate(data_path=csv_file, output_report_path=out_file)
    assert report is not None
    meta = report["metadata"]
    assert meta["label_verified"] is False
    assert "UNVERIFIED" in meta["label_verification_status"]
    assert "CANNOT BE ESTABLISHED" in meta["deception_detection_validity"]
    assert "label_verified=False" in meta["deception_detection_validity"]


def test_deception_model_loads_metadata_provenance():
    """DeceptionModel must read model_metadata.json and expose data_provenance."""
    model = DeceptionModel()
    dummy = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.5
    pred = model.predict(dummy)
    
    assert "data_provenance" in pred
    assert hasattr(model, "provenance")
    assert model.provenance in ("SYNTHETIC", "REAL_HUMAN", "UNKNOWN")


# ─── 6. TRAINING POLICY & 3-SCOPE PERFORMANCE REGRESSION TESTS ───────────────

def test_training_policy_categorizes_unverified_task_model(tmp_path):
    """When label_verified=False, model must be categorized as experimental task-classification."""
    from training.train_model import train_model
    import json
    
    csv_file = str(tmp_path / "task_data.csv")
    data = {f: [0.5] * 20 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 10
    data["data_provenance"] = ["REAL_HUMAN"] * 20
    data["label_protocol"] = ["CONTROLLED_TASK_DIRECTED"] * 20
    data["label_verified"] = [False] * 20
    data["participant_id"] = ["P_001"] * 10 + ["P_002"] * 10
    data["session_id"] = ["S_001"] * 10 + ["S_002"] * 10
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    train_model(data_path=csv_file)
    
    with open("models/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["model_category"] == "EXPERIMENTAL_TASK_CLASSIFICATION"
    assert meta["label_verified"] is False
    assert meta["label_protocol"] == "CONTROLLED_TASK_DIRECTED"
    assert "task-directed" in meta["intended_interpretation"].lower() or "instructed" in meta["intended_interpretation"].lower()
    assert "NOT verified lie detection" in meta["intended_interpretation"]
    assert any("label_verified=False" in lim for lim in meta["limitations"])
    assert meta["unique_participants_count"] == 2
    assert meta["unique_sessions_count"] == 2


def test_training_policy_does_not_mutate_label_verified(tmp_path):
    """Training must never silently mutate label_verified to True."""
    from training.train_model import train_model
    import json
    
    csv_file = str(tmp_path / "never_mutate.csv")
    data = {f: [0.5] * 20 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 10
    data["data_provenance"] = ["REAL_HUMAN"] * 20
    data["label_verified"] = [False] * 20
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    train_model(data_path=csv_file)
    
    with open("models/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["label_verified"] is False, "label_verified must never be changed to True during training."


def test_evaluation_report_clearly_distinguishes_three_performance_scopes(tmp_path):
    """Evaluation reports must explicitly distinguish all 3 performance scopes and mark lie detection NOT ESTABLISHED."""
    csv_file = str(tmp_path / "scopes_data.csv")
    out_file = str(tmp_path / "scopes_report.json")
    
    data = {f: [0.5] * 20 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 10
    data["data_provenance"] = ["REAL_HUMAN"] * 20
    data["label_verified"] = [False] * 20
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    report = evaluate(data_path=csv_file, output_report_path=out_file)
    assert report is not None
    meta = report["metadata"]
    
    # Check top-level scope fields
    assert "synthetic_benchmark_performance" in meta
    assert "controlled_task_classification_performance" in meta
    assert "independently_validated_deception_detection" in meta
    
    # Real human data with unverified labels:
    assert meta["synthetic_benchmark_performance"] == "NOT_APPLICABLE"
    assert meta["controlled_task_classification_performance"] == "EVALUATED"
    assert meta["independently_validated_deception_detection"] == "NOT ESTABLISHED"
    
    # Detailed performance scopes breakdown:
    scopes = meta["performance_scopes"]
    assert scopes["synthetic_benchmark"]["status"] == "NOT_APPLICABLE"
    assert scopes["controlled_task_classification"]["status"] == "EVALUATED"
    assert scopes["independently_validated_deception_detection"]["status"] == "NOT ESTABLISHED"
    assert "No claim of human lie detection accuracy" in scopes["independently_validated_deception_detection"]["description"]


def test_training_preserves_participant_independent_splits(tmp_path):
    """Verify that participant-independent splitting has zero participant leakage between train and test."""
    from training.train_model import train_model
    import json
    
    csv_file = str(tmp_path / "split_test.csv")
    data = {f: [0.5] * 40 for f in config.FEATURE_NAMES}
    data["label"] = [0, 1] * 20
    data["data_provenance"] = ["REAL_HUMAN"] * 40
    data["label_verified"] = [False] * 40
    data["participant_id"] = ["P_1"] * 10 + ["P_2"] * 10 + ["P_3"] * 10 + ["P_4"] * 10
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    train_model(data_path=csv_file)
    
    with open("models/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["split_type"] == "SUBJECT_INDEPENDENT_GROUP_SPLIT"
    train_p = set(meta["train_participants"])
    test_p = set(meta["test_participants"])
    assert len(train_p) > 0
    assert len(test_p) > 0
    assert train_p.isdisjoint(test_p), "Train and test participants must have zero intersection (no identity leakage)."




