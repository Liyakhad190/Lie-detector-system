"""
Unit tests for ExplainabilityDecomposer and 4-channel anomaly attribution.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.explainability import ExplainabilityDecomposer


def test_explainability_channel_structure():
    """Verify all 4 channels (ocular, myofacial, micro_dynamics, autonomic) are produced."""
    decomposer = ExplainabilityDecomposer()
    dummy = np.ones(len(config.FEATURE_NAMES), dtype=np.float32) * 0.3
    
    decomp = decomposer.decompose(dummy, polygraph_data={"estimated_hr": 78.0})
    assert "composite_anomaly_score" in decomp
    assert "channels" in decomp
    channels = decomp["channels"]
    
    assert "ocular" in channels
    assert "myofacial" in channels
    assert "micro_dynamics" in channels
    assert "autonomic" in channels
    
    assert 0.0 <= decomp["composite_anomaly_score"] <= 1.0
    for ch_name, ch_dict in channels.items():
        assert 0.0 <= ch_dict["score"] <= 1.0
        assert ch_dict["weight"] > 0.0


def test_explainability_evidence_attribution():
    """Verify evidence factors are generated when specific channels spike."""
    decomposer = ExplainabilityDecomposer()
    dummy = np.zeros(len(config.FEATURE_NAMES), dtype=np.float32)
    # Simulate high ocular stress: high blink rate and gaze variance
    dummy[0] = 35.0 # blink rate
    dummy[4] = 0.45 # gaze horizontal std
    dummy[5] = 0.40 # gaze vertical std
    dummy[6] = 0.30 # fixation ratio dropped
    
    decomp = decomposer.decompose(dummy)
    assert decomp["channels"]["ocular"]["score"] > 0.45
    assert len(decomp["evidence_factors"]) > 0
    assert any("Ocular" in factor["channel"] for factor in decomp["evidence_factors"])


def test_empty_or_invalid_decomposition():
    """Verify safe fallback for None or malformed inputs."""
    decomposer = ExplainabilityDecomposer()
    res = decomposer.decompose(None)
    assert res["composite_anomaly_score"] == 0.0
    assert res["primary_contributor"] == "None"
