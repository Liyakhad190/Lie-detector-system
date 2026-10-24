import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt


class PolygraphAnalyzer:
    """
    Real-time physiological estimation (rPPG) and polygraph history tracking.
    Extracts blood volume pulse from forehead green channel photoplethysmography.
    """
    def __init__(self, history_len=150, rppg_buffer_len=150):
        self.history_len = history_len
        self.rppg_buffer_len = rppg_buffer_len

        # Physiological data (rPPG)
        self.green_signal = deque(maxlen=rppg_buffer_len)
        self.times = deque(maxlen=rppg_buffer_len)
        self.current_hr = 75.0
        self.is_valid_hr = False
        self.latest_pulse_val = 0.5  # Real-time cardiac pulse waveform sample [0, 1]

        # Display history buffers (for the HUD Polygraph chart)
        self.history = {
            "stress": deque(maxlen=history_len),
            "hr": deque(maxlen=history_len),
            "deception": deque(maxlen=history_len)
        }

        # Pre-fill with baseline values so the graph draws immediately
        for _ in range(history_len):
            self.history["stress"].append(0.0)
            self.history["hr"].append(0.5)
            self.history["deception"].append(0.0)

    def _butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = max(0.01, min(0.99, lowcut / nyq))
        high = max(0.02, min(0.99, highcut / nyq))
        if low >= high:
            low = high * 0.5
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _estimate_heart_rate(self):
        """Estimate Heart Rate from the buffered green channel signal using FFT."""
        # Require at least 40 frames (~1.5-2 seconds) to compute real-time HR
        if len(self.green_signal) < 40:
            return self.current_hr

        signal = np.array(self.green_signal, dtype=np.float64)
        t = np.array(self.times, dtype=np.float64)

        dt = np.mean(np.diff(t))
        if dt <= 0:
            return self.current_hr

        fs = 1.0 / dt
        if fs < 5.0:
            return self.current_hr

        # Detrend the signal
        signal = signal - np.mean(signal)

        # Bandpass filter (45 to 180 bpm -> 0.75 to 3.0 Hz)
        try:
            highcut = min(3.0, 0.45 * fs)
            if highcut <= 0.75:
                return self.current_hr

            b, a = self._butter_bandpass(0.75, highcut, fs, order=2)
            filtered_signal = filtfilt(b, a, signal)

            # Extract real-time pulse waveform sample for polygraph green curve
            if len(filtered_signal) > 0:
                p_std = float(np.std(filtered_signal))
                if p_std > 1e-4:
                    self.latest_pulse_val = float(np.clip(0.5 + (filtered_signal[-1] / (3.0 * p_std)) * 0.4, 0.0, 1.0))

            # Use FFT to find dominant frequency
            fft = np.abs(np.fft.rfft(filtered_signal))
            freqs = np.fft.rfftfreq(len(filtered_signal), 1.0 / fs)

            # Find dominant frequency within human HR range (0.75 Hz to 3.0 Hz)
            valid_idx = np.where((freqs >= 0.75) & (freqs <= highcut))[0]
            if len(valid_idx) > 0:
                best_freq = freqs[valid_idx[np.argmax(fft[valid_idx])]]
                new_hr = float(best_freq * 60.0)
                if 45.0 <= new_hr <= 180.0:
                    self.current_hr = (self.current_hr * 0.85) + (new_hr * 0.15)
                    self.is_valid_hr = True
        except Exception:
            pass

        return self.current_hr

    def add_deception(self, deception_prob: float):
        """Append deception score to history once per frame."""
        self.history["deception"].append(float(np.clip(deception_prob, 0.0, 1.0)))

    def update(self, frame, landmarks, stress_score, deception_prob=None):
        """Update the polygraph history with new frame and metrics."""
        # 1. Extract rPPG signal (Green channel of forehead)
        if landmarks is not None and frame is not None:
            h, w = frame.shape[:2]

            # Use forehead center (landmark 10 in MediaPipe face mesh)
            lm = landmarks[10]
            # Handle both normalized [0, 1] (unit tests) and pixel coords (FaceData)
            cx = int(lm[0] * w) if float(lm[0]) <= 1.0 else int(lm[0])
            cy = int(lm[1] * h) if float(lm[1]) <= 1.0 else int(lm[1])

            roi_size = 20
            # Ensure ROI is within bounds
            x1 = max(0, cx - roi_size)
            y1 = max(0, cy - roi_size)
            x2 = min(w, cx + roi_size)
            y2 = min(h, cy + roi_size)

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                mean_green = float(np.mean(roi[:, :, 1]))
                self.green_signal.append(mean_green)
                self.times.append(time.time())

                # Estimate HR and pulse waveform
                self._estimate_heart_rate()

        # 2. Real-time pulse waveform plotted on the green HR curve
        hr_plot_val = self.latest_pulse_val

        # 3. Update history buffers
        self.history["stress"].append(float(np.clip(stress_score, 0.0, 1.0)))
        self.history["hr"].append(hr_plot_val)
        if deception_prob is not None:
            self.history["deception"].append(float(np.clip(deception_prob, 0.0, 1.0)))

        return {
            "hr_bpm": self.current_hr,
            "hr_norm": hr_plot_val,
            "is_valid_hr": self.is_valid_hr,
            "history": self.history
        }
