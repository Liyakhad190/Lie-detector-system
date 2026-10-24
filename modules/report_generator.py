"""
============================================================================
 REPORT GENERATOR MODULE - Comprehensive Session Analysis & Audit
============================================================================
 Produces human-readable Markdown audit reports and structured JSON exports
 from recorded session logs and protocol executions.
============================================================================
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import config


class SessionReportGenerator:
    """Generates structured Markdown and JSON reports for deception analysis sessions."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self,
                        session_id: str,
                        log_entries: List[Dict[str, Any]],
                        protocol_summary: Optional[Dict[str, Any]] = None,
                        baseline: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Synthesize session metrics into Markdown and JSON reports.
        
        Returns:
            Dictionary with paths to generated 'markdown_path' and 'json_path'.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = os.path.join(self.output_dir, f"session_report_{timestamp}.md")
        json_filename = os.path.join(self.output_dir, f"session_summary_{timestamp}.json")

        total_frames = len(log_entries)
        if total_frames == 0:
            print("[REPORT] Warning: No log entries recorded in session.")

        # Compute aggregate statistics
        # Support both in-memory dict keys and CSV-serialised key names (deception_prob vs deception_probability)
        def _getf(entry, *keys):
            for k in keys:
                v = entry.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return None

        probs = [_getf(e, "deception_probability", "deception_prob") for e in log_entries]
        probs = [v for v in probs if v is not None]
        stress_scores = [_getf(e, "stress_score") for e in log_entries]
        stress_scores = [v for v in stress_scores if v is not None]
        blink_rates = [_getf(e, "blink_rate") for e in log_entries]
        blink_rates = [v for v in blink_rates if v is not None]
        hrs = [_getf(e, "heart_rate", "estimated_hr") for e in log_entries]
        hrs = [v for v in hrs if v is not None]

        mean_prob = float(np.mean(probs)) if probs else 0.0
        max_prob = float(np.max(probs)) if probs else 0.0
        mean_stress = float(np.mean(stress_scores)) if stress_scores else 0.0
        mean_blink = float(np.mean(blink_rates)) if blink_rates else 0.0
        mean_hr = float(np.mean(hrs)) if hrs else 75.0

        # Protocol data
        protocol_name = protocol_summary.get("protocol_name", "AD-HOC SESSION") if protocol_summary else "AD-HOC SESSION"
        overall_verdict = protocol_summary.get("overall_verdict", "STANDARD SESSION COMPLETED") if protocol_summary else "SESSION COMPLETED"
        q_summaries = protocol_summary.get("question_summaries", []) if protocol_summary else []

        # Build JSON Summary
        summary_data = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "protocol_name": protocol_name,
            "overall_verdict": overall_verdict,
            "sample_count": total_frames,
            "aggregate_metrics": {
                "mean_behavioral_anomaly_score": round(mean_prob, 3),
                "peak_behavioral_anomaly_score": round(max_prob, 3),
                "mean_deception_probability": round(mean_prob, 3),  # backward-compat alias
                "peak_deception_probability": round(max_prob, 3),  # backward-compat alias
                "mean_stress_score": round(mean_stress, 3),
                "mean_blink_rate_bpm": round(mean_blink, 1),
                "mean_heart_rate_bpm": round(mean_hr, 1)
            },
            "baseline": baseline or {},
            "question_evaluations": q_summaries,
            "disclaimer": config.DISCLAIMER
        }

        with open(json_filename, "w", encoding="utf-8") as jf:
            json.dump(summary_data, jf, indent=2)

        # Build Markdown Document
        md_lines = [
            "# 📋 Behavioral Analysis Session Report",
            f"**Session ID:** `{session_id}`  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Protocol:** `{protocol_name}`  ",
            "",
            "> ⚠️ **SCIENTIFIC & LEGAL DISCLAIMER**",
            f"> {config.DISCLAIMER}",
            "> This system is an experimental research tool. Measured facial, ocular, and physiological signals indicate autonomic reactivity and cognitive load, NOT definitive truth or deception. Results must never be used as evidence of lying.",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            f"- **Overall Assessment:** **{overall_verdict}**",
            f"- **Average Behavioral Anomaly Score:** `{mean_prob:.3f}` (Scale: 0.0 - 1.0)",
            f"- **Peak Behavioral Anomaly Score:** `{max_prob:.3f}`",
            f"- **Mean Stress Score:** `{mean_stress:.3f}`",
            f"- **Recorded Frames:** `{total_frames}`",
            "",
            "## 2. Physiological & Behavioral Baselines",
            "| Parameter | Session Mean | Baseline Reference | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Blink Rate** | {mean_blink:.1f} bpm | 15.0 - 20.0 bpm | {'Elevated' if mean_blink > 25 else 'Normal'} |",
            f"| **Forehead Pulse (rPPG)** | {mean_hr:.1f} BPM | 60.0 - 85.0 BPM | {'Elevated' if mean_hr > 85 else 'Resting'} |",
            f"| **Composite Stress** | {mean_stress:.2f} | 0.20 - 0.35 | {'Elevated' if mean_stress > 0.45 else 'Nominal'} |",
            "",
        ]

        if q_summaries:
            md_lines.extend([
                "## 3. Structured Protocol Question Breakdown",
                "| # | Category | Question Text | Duration | Anomaly Score | Risk Level |",
                "| :-: | :--- | :--- | :-: | :-: | :-: |"
            ])
            for q in q_summaries:
                md_lines.append(
                    f"| {q.get('qid', '-')} | `{q.get('category', '-')}` | {q.get('text', '-')} | "
                    f"{q.get('duration', 0):.1f}s | `{q.get('mean_deception_prob', 0.0):.3f}` | **{q.get('risk_level', 'LOW')}** |"
                )
            md_lines.append("")

        md_lines.extend([
            "## 4. Multi-Channel Diagnostic Explanation",
            "The system deconstructs behavioral evidence across four independent observation channels:",
            "1. **Ocular Channel:** Tracks rapid micro-blinks, gaze aversion saccades, and eye fixation stability.",
            "2. **Myofacial Channel:** Measures brow furrowing, lip compression ratios, and jaw clenching tension.",
            "3. **Micro-Dynamics Channel:** Captures involuntary micro-facial changes (<200ms) and bilateral asymmetry deltas.",
            "4. **Autonomic Pulse Channel:** Estimates cardiovascular pulse variation via remote photoplethysmography (rPPG).",
            "",
            "---",
            "*Report automatically generated by Deception Analysis System v2.0.*"
        ])

        with open(md_filename, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines))

        print(f"[REPORT] Exported Markdown report to: {md_filename}")
        print(f"[REPORT] Exported JSON summary to:    {json_filename}")

        return {"markdown_path": md_filename, "json_path": json_filename}
