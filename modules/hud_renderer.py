"""
============================================================================
 HUD RENDERER MODULE - Futuristic AI Surveillance UI
============================================================================
 Renders a dark, futuristic Head-Up Display overlay on the webcam feed
 with neon-colored metrics, animated bars, scanlines, and panels.
============================================================================
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, Tuple, List, Optional
import config


class HUDRenderer:
    """Futuristic HUD overlay renderer for the lie detection system."""

    def __init__(self, width: int, height: int):
        self._w = width
        self._h = height
        self._frame_count = 0
        self._start_time = time.time()
        self._colors = config.HUD_COLORS
        self._scanline_offset = 0

    def render(self, frame: np.ndarray, faces_data: List[Dict],
               calibration_progress: float = 1.0,
               is_calibrating: bool = False,
               calibration_remaining: float = 0.0,
               fps: float = 0.0,
               log_count: int = 0,
               interview_state: int = 0,
               final_verdict: Dict = None,
               current_question: str = "") -> np.ndarray:
        """Render the complete HUD overlay on the frame."""
        self._frame_count += 1
        h, w = frame.shape[:2]
        self._w = w
        self._h = h

        # Create overlay for transparency effects
        overlay = frame.copy()

        # Draw dark border panels
        self._draw_panels(overlay)

        # Top bar
        self._draw_top_bar(overlay, fps, len(faces_data), log_count)

        # Bottom disclaimer bar
        self._draw_bottom_bar(overlay)

        # Animated scanlines
        self._draw_scanlines(overlay)

        # Corner decorations
        self._draw_corner_brackets(overlay)

        if is_calibrating:
            self._draw_calibration_overlay(overlay, calibration_progress,
                                           calibration_remaining)
        elif not faces_data:
            self._draw_no_face_warning(overlay)
        else:
            # Draw metrics for primary face
            primary = faces_data[0]
            self._draw_side_panel(overlay, primary)
            self._draw_face_bbox(overlay, primary)
            self._draw_prediction_display(overlay, primary)
            
            if "polygraph_data" in primary:
                self._draw_polygraph_chart(overlay, primary["polygraph_data"])

        # Draw Interview Mode Status
        self._draw_interview_status(overlay, interview_state, current_question)

        # Draw Final Verdict if available
        if interview_state == 2 and final_verdict:
            self._draw_final_verdict_overlay(overlay, final_verdict, current_question)

        # Blend overlay
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def _draw_panels(self, frame: np.ndarray):
        """Draw semi-transparent dark panels."""
        h, w = frame.shape[:2]
        pw = config.SIDE_PANEL_WIDTH

        # Right side panel background
        panel = frame[config.TOP_BAR_HEIGHT:h - config.BOTTOM_BAR_HEIGHT,
                       w - pw:w]
        dark = np.full_like(panel, self._colors["panel_bg"], dtype=np.uint8)
        cv2.addWeighted(dark, 0.7, panel, 0.3, 0, panel)

        # Top bar background
        top = frame[0:config.TOP_BAR_HEIGHT, 0:w]
        dark_top = np.full_like(top, (10, 10, 15), dtype=np.uint8)
        cv2.addWeighted(dark_top, 0.8, top, 0.2, 0, top)

        # Bottom bar background
        bot = frame[h - config.BOTTOM_BAR_HEIGHT:h, 0:w]
        dark_bot = np.full_like(bot, (10, 10, 15), dtype=np.uint8)
        cv2.addWeighted(dark_bot, 0.8, bot, 0.2, 0, bot)

    def _draw_top_bar(self, frame: np.ndarray, fps: float,
                      face_count: int, log_count: int):
        """Draw top status bar with system info."""
        h, w = frame.shape[:2]
        y = 30
        color = self._colors["primary"]
        dim = self._colors["text_secondary"]

        # System title
        cv2.putText(frame, "DECEPTION ANALYSIS SYSTEM v2.0",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Animated pulse dot
        pulse = abs(math.sin(self._frame_count * 0.1))
        pulse_color = (0, int(255 * pulse), int(200 * pulse))
        cv2.circle(frame, (w - 320, y - 5), 5, pulse_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ACTIVE", (w - 310, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 230, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)

        # Face count
        cv2.putText(frame, f"FACES: {face_count}", (w - 150, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)

        # Log entries
        cv2.putText(frame, f"LOG: {log_count}", (w - 70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(frame, (0, config.TOP_BAR_HEIGHT),
                 (w, config.TOP_BAR_HEIGHT), color, 1, cv2.LINE_AA)

    def _draw_bottom_bar(self, frame: np.ndarray):
        """Draw bottom disclaimer bar."""
        h, w = frame.shape[:2]
        y = h - 20
        cv2.line(frame, (0, h - config.BOTTOM_BAR_HEIGHT),
                 (w, h - config.BOTTOM_BAR_HEIGHT),
                 self._colors["primary"], 1, cv2.LINE_AA)

        # Scrolling disclaimer
        elapsed = time.time() - self._start_time
        text = config.DISCLAIMER
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0]
        scroll_x = int(w - (elapsed * 50) % (text_width + w))

        cv2.putText(frame, text, (scroll_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self._colors["warning"], 1, cv2.LINE_AA)

        # Controls hint
        cv2.putText(frame, "[Q] Quit  [S] Save  [C] Calibrate  [R] Reset",
                    (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self._colors["text_secondary"], 1, cv2.LINE_AA)

    def _draw_scanlines(self, frame: np.ndarray):
        """Draw animated scanning lines for futuristic effect."""
        h, w = frame.shape[:2]
        self._scanline_offset = (self._scanline_offset + 2) % h

        # Horizontal scanline
        y = self._scanline_offset
        if 0 < y < h:
            line_overlay = frame[max(0, y-1):min(h, y+1), :].copy()
            bright = np.clip(line_overlay.astype(np.int16) + 20, 0, 255).astype(np.uint8)
            frame[max(0, y-1):min(h, y+1), :] = bright

    def _draw_corner_brackets(self, frame: np.ndarray):
        """Draw decorative corner brackets."""
        h, w = frame.shape[:2]
        color = self._colors["primary"]
        length = 30
        thickness = 1
        margin = 5

        corners = [
            (margin, margin),
            (w - margin, margin),
            (margin, h - margin),
            (w - margin, h - margin)
        ]

        for cx, cy in corners:
            dx = length if cx < w // 2 else -length
            dy = length if cy < h // 2 else -length
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, thickness, cv2.LINE_AA)

    def _draw_side_panel(self, frame: np.ndarray, data: Dict):
        """Draw the right-side metrics panel."""
        h, w = frame.shape[:2]
        pw = config.SIDE_PANEL_WIDTH
        x_start = w - pw + 15
        y = config.TOP_BAR_HEIGHT + 25
        color = self._colors["primary"]
        dim = self._colors["text_secondary"]
        bright = self._colors["text_bright"]

        # Panel title
        cv2.putText(frame, "BEHAVIORAL ANALYSIS",
                    (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 5
        cv2.line(frame, (x_start, y), (w - 15, y), color, 1, cv2.LINE_AA)
        y += 25

        blink = data.get("blink_data", {})
        movement = data.get("movement_data", {})
        micro = data.get("micro_data", {})
        stress = data.get("stress_data", {})
        pred = data.get("prediction", {})

        # Metrics to display
        metrics = [
            ("BLINK RATE", f"{blink.get('blink_rate_per_min', 0):.0f}/min",
             self._blink_color(blink.get('blink_rate_per_min', 15))),
            ("EYE ASPECT RATIO", f"{blink.get('avg_ear', 0):.3f}", dim),
            ("TOTAL BLINKS", f"{blink.get('total_blinks', 0)}", dim),
            None,  # Spacer
            ("GAZE STABILITY", f"{movement.get('gaze_fixation_ratio', 1):.0%}",
             self._stability_color(movement.get('gaze_fixation_ratio', 1))),
            ("HEAD MOVEMENT", f"{movement.get('movement_magnitude', 0):.2f}",
             dim),
            ("NERVOUS SCORE", f"{movement.get('nervous_score', 0):.0%}",
             self._stress_color(movement.get('nervous_score', 0))),
            None,
            ("FACE SYMMETRY", f"{micro.get('face_symmetry', 1):.0%}",
             self._symmetry_color(micro.get('face_symmetry', 1))),
            ("LIP COMPRESSION", f"{micro.get('lip_compression', 0):.3f}", dim),
            ("BROW TENSION", f"{micro.get('eyebrow_tension', 0):.0%}",
             self._stress_color(micro.get('eyebrow_tension', 0))),
            ("MICRO-EXPR", f"{micro.get('micro_expression_count', 0)}",
             self._colors["accent"] if micro.get('micro_expression_count', 0) > 0 else dim),
            ("JAW CLENCH", f"{micro.get('jaw_clench', 0):.0%}", dim),
            None,
            ("FACIAL TENSION", f"{micro.get('facial_tension', 0):.0%}",
             self._stress_color(micro.get('facial_tension', 0))),
        ]

        for item in metrics:
            if item is None:
                y += 10
                continue
            label, value, val_color = item
            cv2.putText(frame, label, (x_start, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (x_start + 160, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, val_color, 1, cv2.LINE_AA)
            y += 22

        # Stress bar
        y += 10
        stress_score = stress.get("stress_score", 0)
        stress_level = stress.get("stress_level", "LOW")
        stress_color = self._colors.get(stress.get("stress_color_key", "stress_low"),
                                         self._colors["success"])

        cv2.putText(frame, "STRESS LEVEL", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(frame, stress_level, (x_start + 160, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stress_color, 1, cv2.LINE_AA)
        y += 15

        # Stress bar visualization
        bar_w = pw - 40
        bar_h = 8
        cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + bar_h),
                       self._colors["grid"], -1)
        fill_w = int(bar_w * stress_score)
        if fill_w > 0:
            cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + bar_h),
                           stress_color, -1)
        cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + bar_h),
                       color, 1)

    def _draw_face_bbox(self, frame: np.ndarray, data: Dict):
        """Draw a futuristic bounding box around the detected face."""
        bbox = data.get("bbox", (0, 0, 0, 0))
        x, y, bw, bh = bbox
        if bw < 10 or bh < 10:
            return

        pred = data.get("prediction", {})
        label_color_key = pred.get("label_color", "truth")
        color = self._colors.get(label_color_key, self._colors["primary"])

        # Corner brackets on face
        bracket_len = min(bw, bh) // 5
        thickness = 2

        corners = [
            (x, y), (x + bw, y),
            (x, y + bh), (x + bw, y + bh)
        ]
        for cx, cy in corners:
            dx = bracket_len if cx == x else -bracket_len
            dy = bracket_len if cy == y else -bracket_len
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, thickness, cv2.LINE_AA)

        # Face ID label
        face_id = data.get("face_id", 0)
        cv2.putText(frame, f"SUBJECT {face_id}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)

    def _draw_prediction_display(self, frame: np.ndarray, data: Dict):
        """Draw the main prediction result display."""
        h, w = frame.shape[:2]
        pred = data.get("prediction", {})
        if not pred:
            return

        deception = pred.get("deception_probability", 0)
        truth = pred.get("truth_probability", 1)
        confidence = pred.get("confidence", 0.5)
        label = pred.get("label", "ANALYZING...")
        label_color_key = pred.get("label_color", "truth")
        color = self._colors.get(label_color_key, self._colors["primary"])

        # Main label display (left side)
        x = 20
        y = config.TOP_BAR_HEIGHT + 40

        # Prediction label with pulsing effect
        pulse = 0.7 + 0.3 * abs(math.sin(self._frame_count * 0.05))
        label_color_val = tuple(int(c * pulse) for c in color)
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color_val, 2, cv2.LINE_AA)

        y += 35
        # Truth bar
        cv2.putText(frame, "TRUTH", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    self._colors["text_secondary"], 1, cv2.LINE_AA)
        bar_x = x + 80
        bar_w = 200
        bar_h = 14
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y - 12 + bar_h),
                       self._colors["grid"], -1)
        fill = int(bar_w * truth)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y - 12), (bar_x + fill, y - 12 + bar_h),
                           self._colors["truth"], -1)
        cv2.putText(frame, f"{truth:.0%}", (bar_x + bar_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._colors["truth"], 1, cv2.LINE_AA)

        y += 25
        # Deception bar
        cv2.putText(frame, "DECEPT", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    self._colors["text_secondary"], 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y - 12 + bar_h),
                       self._colors["grid"], -1)
        fill = int(bar_w * deception)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y - 12), (bar_x + fill, y - 12 + bar_h),
                           self._colors["deception"], -1)
        cv2.putText(frame, f"{deception:.0%}", (bar_x + bar_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._colors["deception"], 1, cv2.LINE_AA)

        y += 25
        # Confidence and CI
        ci = pred.get("confidence_interval", (0.0, 0.0))
        cv2.putText(frame, f"CONFIDENCE: {confidence:.0%} [CI: {ci[0]:.0%}-{ci[1]:.0%}]", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    self._colors["secondary"], 1, cv2.LINE_AA)

        # Method indicator
        method = pred.get("method", "HEURISTIC")
        cv2.putText(frame, f"[{method}]", (x + 230, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self._colors["text_secondary"], 1, cv2.LINE_AA)

        # Multi-Channel Anomaly Attribution Mini-Gauges
        exp = pred.get("explainability", {})
        channels = exp.get("channels", {})
        if channels:
            y += 22
            cv2.putText(frame, "ANOMALY CHANNELS:", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._colors["text_secondary"], 1, cv2.LINE_AA)
            channel_items = [
                ("OCULAR", channels.get("ocular", {}).get("score", 0.0)),
                ("TENSION", channels.get("myofacial", {}).get("score", 0.0)),
                ("MICRO", channels.get("micro_dynamics", {}).get("score", 0.0)),
                ("PULSE", channels.get("autonomic", {}).get("score", 0.0)),
            ]
            cx_pos = x
            for ch_name, ch_score in channel_items:
                y_ch = y + 15
                ch_color = self._colors["danger"] if ch_score > 0.50 else (
                    self._colors["warning"] if ch_score > 0.30 else self._colors["success"]
                )
                cv2.putText(frame, f"{ch_name[:3]}:{ch_score:.0%}", (cx_pos, y_ch),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, ch_color, 1, cv2.LINE_AA)
                cx_pos += 65

    def _draw_calibration_overlay(self, frame: np.ndarray,
                                   progress: float, remaining: float):
        """Draw calibration mode overlay."""
        h, w = frame.shape[:2]
        color = self._colors["calibration"]

        # Center text
        cx, cy = w // 2, h // 2

        cv2.putText(frame, "CALIBRATION MODE",
                    (cx - 150, cy - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

        cv2.putText(frame, "Look at the camera naturally",
                    (cx - 160, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._colors["text_primary"], 1, cv2.LINE_AA)

        cv2.putText(frame, f"Establishing baseline... {remaining:.0f}s",
                    (cx - 150, cy + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._colors["text_secondary"], 1, cv2.LINE_AA)

        # Progress bar
        bar_w = 300
        bar_h = 12
        bx = cx - bar_w // 2
        by = cy + 50
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h),
                       self._colors["grid"], -1)
        fill = int(bar_w * progress)
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h),
                           color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h),
                       color, 1)

        cv2.putText(frame, f"{progress:.0%}",
                    (cx - 15, by + bar_h + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    def _draw_no_face_warning(self, frame: np.ndarray):
        """Draw warning when no face is detected."""
        h, w = frame.shape[:2]
        pulse = abs(math.sin(self._frame_count * 0.1))
        color = (0, 0, int(255 * pulse))

        cx, cy = w // 2, h // 2
        cv2.putText(frame, "! NO FACE DETECTED !",
                    (cx - 160, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Position your face in front of the camera",
                    (cx - 210, cy + 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, self._colors["text_secondary"], 1, cv2.LINE_AA)

    def _draw_polygraph_chart(self, frame: np.ndarray, polygraph_data: Dict):
        """Draw a real-time scrolling polygraph chart at the bottom left."""
        h, w = frame.shape[:2]
        
        chart_w = getattr(config, 'POLYGRAPH_WIDTH', 500)
        chart_h = getattr(config, 'POLYGRAPH_HEIGHT', 150)
        
        # Position (bottom left, above the bottom bar)
        x = 20
        y = h - config.BOTTOM_BAR_HEIGHT - chart_h - 20
        
        # Prevent drawing if screen is too small
        if y < config.TOP_BAR_HEIGHT + 100:
            return
            
        # Draw background panel
        panel = frame[y:y+chart_h, x:x+chart_w]
        dark = np.full_like(panel, self._colors["panel_bg"], dtype=np.uint8)
        cv2.addWeighted(dark, 0.7, panel, 0.3, 0, panel)
        cv2.rectangle(frame, (x, y), (x + chart_w, y + chart_h), self._colors["grid"], 1)
        
        # Title
        cv2.putText(frame, "PHYSIOLOGICAL & STRESS POLYGRAPH", (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._colors["primary"], 1, cv2.LINE_AA)
                    
        # Heart rate value
        hr = polygraph_data.get("hr_bpm", 0)
        cv2.putText(frame, f"HR: {hr:.1f} BPM", (x + chart_w - 120, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._colors["success"], 1, cv2.LINE_AA)
        
        history = polygraph_data.get("history", {})
        if not history:
            return
            
        # Lines config: (key, color)
        lines = [
            ("hr", self._colors["success"]),      # Green
            ("stress", self._colors["danger"]),   # Red
            ("deception", self._colors["warning"]) # Amber
        ]
        
        for key, color in lines:
            data = list(history.get(key, []))
            if len(data) < 2:
                continue
                
            dx = chart_w / (len(data) - 1)
            pts = []
            for i, val in enumerate(data):
                px = int(x + i * dx)
                # Map value [0, 1] to chart height (leaving margins)
                margin = 35
                usable_h = chart_h - margin - 15
                # Ensure val is clamped between 0 and 1 just in case
                val_clamped = max(0.0, min(1.0, float(val)))
                py = int(y + chart_h - 15 - (val_clamped * usable_h))
                pts.append((px, py))
                
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            
        # Draw legend
        cv2.putText(frame, "HR", (x + 10, y + chart_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._colors["success"], 1, cv2.LINE_AA)
        cv2.putText(frame, "STRESS", (x + 50, y + chart_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._colors["danger"], 1, cv2.LINE_AA)
        cv2.putText(frame, "DECEPTION", (x + 110, y + chart_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._colors["warning"], 1, cv2.LINE_AA)

    def _draw_interview_status(self, frame: np.ndarray, state: int, current_question: str):
        """Draw the current interview mode state."""
        h, w = frame.shape[:2]
        
        y = config.TOP_BAR_HEIGHT + 25
        x = 20
        
        if state == 0:
            text = "INTERVIEW: IDLE (Press T to ask question, SPACE to record answer)"
            color = self._colors["text_primary"]
            pulse = 1.0
        elif state == 1:
            text = "INTERVIEW: RECORDING ANSWER (Press SPACE to end and analyze)"
            pulse = 0.5 + 0.5 * abs(math.sin(self._frame_count * 0.1))
            color = tuple(int(c * pulse) for c in self._colors["danger"])
        else: # state == 2
            text = "INTERVIEW: VERDICT RENDERED (Press SPACE to reset)"
            color = self._colors["primary"]
            pulse = 1.0
            
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        if current_question:
            # Draw question text below status
            q_text = f"Q: {current_question}"
            # Ensure it fits on screen (truncate if too long)
            if len(q_text) > 80:
                q_text = q_text[:77] + "..."
            cv2.putText(frame, q_text, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._colors["secondary"], 1, cv2.LINE_AA)

    def _draw_final_verdict_overlay(self, frame: np.ndarray, verdict: Dict, question: str):
        """Draw the large final verdict overlay in the center."""
        h, w = frame.shape[:2]
        
        vw, vh = 600, 300
        x = (w - vw) // 2
        y = (h - vh) // 2
        
        color_key = verdict.get("label_color", "primary")
        color = self._colors.get(color_key, self._colors["primary"])
        label = verdict.get("label", "UNKNOWN")
        conf = verdict.get("confidence", 0.0)
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + vw, y + vh), (10, 10, 15), -1)
        cv2.rectangle(overlay, (x, y), (x + vw, y + vh), color, 2)
        
        # Blend
        alpha = 0.95
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Text
        cv2.putText(frame, "FINAL ANSWER VERDICT", (x + 20, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self._colors["text_secondary"], 1, cv2.LINE_AA)
        
        if question:
            q_text = question if len(question) < 55 else question[:52] + "..."
            cv2.putText(frame, f'"{q_text}"', (x + 20, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._colors["secondary"], 1, cv2.LINE_AA)
        
        cv2.line(frame, (x + 20, y + 85), (x + vw - 20, y + 85), color, 1)
        
        # Verdict Label (Large)
        font_scale = 1.8
        thickness = 3
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        tx = x + (vw - text_size[0]) // 2
        ty = y + 160
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # Confidence
        conf_text = f"CONFIDENCE: {conf:.0%}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = x + (vw - text_size[0]) // 2
        ty = y + 230
        cv2.putText(frame, conf_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self._colors["text_primary"], 2, cv2.LINE_AA)
        
        # Details
        dec_prob = verdict.get("deception_probability", 0)
        tru_prob = verdict.get("truth_probability", 0)
        details = f"Deception Score: {dec_prob:.2f} | Truth Score: {tru_prob:.2f}"
        cv2.putText(frame, details, (x + 20, y + vh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._colors["text_secondary"], 1, cv2.LINE_AA)

    # ── Color helper methods ──────────────────────────────────────────

    def _blink_color(self, rate: float) -> Tuple[int, int, int]:
        if config.NORMAL_BLINK_RATE[0] <= rate <= config.NORMAL_BLINK_RATE[1]:
            return self._colors["success"]
        elif rate > config.STRESS_BLINK_RATE:
            return self._colors["danger"]
        return self._colors["warning"]

    def _stability_color(self, ratio: float) -> Tuple[int, int, int]:
        if ratio > 0.7:
            return self._colors["success"]
        elif ratio > 0.4:
            return self._colors["warning"]
        return self._colors["danger"]

    def _symmetry_color(self, score: float) -> Tuple[int, int, int]:
        if score > 0.85:
            return self._colors["success"]
        elif score > 0.65:
            return self._colors["warning"]
        return self._colors["danger"]

    def _stress_color(self, score: float) -> Tuple[int, int, int]:
        if score < 0.3:
            return self._colors["success"]
        elif score < 0.6:
            return self._colors["warning"]
        return self._colors["danger"]
