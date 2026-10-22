"""
Unit tests for SessionReportGenerator (Markdown & JSON report exports).
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.report_generator import SessionReportGenerator


def test_report_generation(tmp_path):
    """Verify Markdown and JSON session audit files are created with valid schemas."""
    output_dir = str(tmp_path / "test_logs")
    generator = SessionReportGenerator(output_dir=output_dir)
    
    dummy_entries = [
        {"deception_probability": 0.25, "stress_score": 0.30, "blink_rate": 18.0, "heart_rate": 72.0},
        {"deception_probability": 0.65, "stress_score": 0.60, "blink_rate": 28.0, "heart_rate": 92.0},
    ]
    
    dummy_protocol_summary = {
        "protocol_name": "ALIBI_INVESTIGATION",
        "overall_verdict": "ELEVATED BEHAVIORAL REACTIVITY OBSERVED",
        "probe_reactivity_delta": 0.35,
        "question_summaries": [
            {"qid": 1, "category": "BASELINE", "text": "State your name.", "duration": 4.5, "mean_deception_prob": 0.22, "risk_level": "LOW"},
            {"qid": 2, "category": "CRITICAL", "text": "Where were you?", "duration": 7.2, "mean_deception_prob": 0.68, "risk_level": "HIGH"}
        ]
    }
    
    paths = generator.generate_report(
        session_id="test_session_001",
        log_entries=dummy_entries,
        protocol_summary=dummy_protocol_summary,
        baseline={"avg_ear": 0.31, "blink_rate": 17.5}
    )
    
    assert os.path.exists(paths["markdown_path"])
    assert os.path.exists(paths["json_path"])
    
    # Verify JSON content
    with open(paths["json_path"], "r", encoding="utf-8") as jf:
        data = json.load(jf)
        assert data["session_id"] == "test_session_001"
        assert data["protocol_name"] == "ALIBI_INVESTIGATION"
        assert len(data["question_evaluations"]) == 2
        assert "disclaimer" in data
        
    # Verify Markdown content
    with open(paths["markdown_path"], "r", encoding="utf-8") as mf:
        content = mf.read()
        assert "Behavioral Analysis Session Report" in content
        assert "SCIENTIFIC & LEGAL DISCLAIMER" in content
        assert "ALIBI_INVESTIGATION" in content
