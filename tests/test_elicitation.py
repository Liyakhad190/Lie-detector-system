"""
Unit tests for ElicitationProtocolManager and structured questioning protocols.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.elicitation_protocol import (
    ElicitationProtocolManager, ElicitationQuestion, QuestionCategory, QuestionState
)


def test_protocol_initialization():
    """Verify loading default and alternate protocols."""
    manager = ElicitationProtocolManager("ALIBI_INVESTIGATION")
    assert manager.protocol_name == "ALIBI_INVESTIGATION"
    assert len(manager.questions) == 5
    assert not manager.is_active

    manager.load_protocol("STANDARD_POLYGRAPH")
    assert manager.protocol_name == "STANDARD_POLYGRAPH"
    assert len(manager.questions) == 6


def test_protocol_lifecycle():
    """Verify start, advance, record, and summary lifecycle."""
    manager = ElicitationProtocolManager("ALIBI_INVESTIGATION")
    manager.start_protocol()
    assert manager.is_active
    
    q1 = manager.get_current_question()
    assert q1 is not None
    assert q1.qid == 1
    assert q1.category == QuestionCategory.BASELINE
    
    # Simulate recording frames for question 1
    manager.record_frame({"deception_probability": 0.15, "raw_score": 0.12})
    manager.record_frame({"deception_probability": 0.20, "raw_score": 0.18})
    
    # Advance to question 2
    summary1 = manager.advance_question()
    assert summary1 is not None
    assert summary1["qid"] == 1
    assert summary1["sample_count"] == 2
    assert summary1["mean_deception_prob"] == pytest.approx(0.175, abs=0.01)
    
    q2 = manager.get_current_question()
    assert q2.qid == 2
    assert q2.category == QuestionCategory.CRITICAL


def test_protocol_summary_verdict():
    """Verify multi-question differential analysis computes correctly."""
    manager = ElicitationProtocolManager("ALIBI_INVESTIGATION")
    manager.start_protocol()
    
    # Q1: Baseline (calm)
    manager.record_frame({"deception_probability": 0.10})
    manager.advance_question()
    
    # Q2: Critical (high reactivity)
    manager.record_frame({"deception_probability": 0.85})
    manager.advance_question()
    
    # Remaining questions
    while manager.is_active:
        manager.record_frame({"deception_probability": 0.50})
        manager.advance_question()
        
    summary = manager.get_protocol_summary()
    assert "protocol_name" in summary
    assert "probe_reactivity_delta" in summary
    assert "overall_verdict" in summary
    assert summary["probe_reactivity_delta"] > 0.30
    assert "ELEVATED" in summary["overall_verdict"]
