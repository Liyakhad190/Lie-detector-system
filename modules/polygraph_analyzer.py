import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt

class PolygraphAnalyzer:
    """
    Real-time physiological estimation (rPPG) and polygraph history tracking.
    """
    def __init__(self, history_len=150, rppg_buffer_len=150):
        self.history_len = history_len
        self.rppg_buffer_len = rppg_buffer_len
        
        # Physiological data (rPPG)
        self.green_signal = deque(maxlen=rppg_buffer_len)
        self.times = deque(maxlen=rppg_buffer_len)
        self.current_hr = 75.0  # Default resting HR
        
        # Display history buffers (for the HUD Polygraph chart)
        # Stored as normalized values [0, 1] for easy plotting
        self.history = {
            "stress": deque(maxlen=history_len),
            "hr": deque(maxlen=history_len),
            "deception": deque(maxlen=history_len)
        }
        
        # Pre-fill with zeros/defaults so the graph draws immediately
        for _ in range(history_len):
            self.history["stress"].append(0.0)
            self.history["hr"].append(0.25)  # 0.25 = ~75 bpm
            self.history["deception"].append(0.0)
            
    def _butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _estimate_heart_rate(self):
        """Estimate Heart Rate from the buffered green channel signal using FFT."""
        if len(self.green_signal) < self.rppg_buffer_len:
            return self.current_hr
            
        signal = np.array(self.green_signal)
        t = np.array(self.times)
        
        dt = np.mean(np.diff(t))
        if dt <= 0:
            return self.current_hr
            
        fs = 1.0 / dt
        
        # Detrend the signal
        signal = signal - np.mean(signal)
        
        # Bandpass filter (45 to 180 bpm -> 0.75 to 3.0 Hz)
        try:
            b, a = self._butter_bandpass(0.75, 3.0, fs, order=3)
            filtered_signal = filtfilt(b, a, signal)
            
            # Use FFT to find dominant frequency
            fft = np.abs(np.fft.rfft(filtered_signal))
            freqs = np.fft.rfftfreq(len(filtered_signal), 1.0/fs)
            
            # Find dominant frequency within human HR range
            valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))[0]
            if len(valid_idx) > 0:
                best_freq = freqs[valid_idx[np.argmax(fft[valid_idx])]]
                # Smooth HR changes
                new_hr = best_freq * 60.0
                self.current_hr = (self.current_hr * 0.8) + (new_hr * 0.2)
                
        except Exception:
            pass # Keep previous HR if filtering fails
            
        return self.current_hr

    def update(self, frame, landmarks, stress_score, deception_prob):
        """Update the polygraph history with new frame and metrics."""
        
        # 1. Extract rPPG signal (Green channel of forehead)
        if landmarks is not None and frame is not None:
            h, w = frame.shape[:2]
            
            # Use forehead center (landmark 10 in MediaPipe face mesh)
            lm = landmarks[10]
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            
            roi_size = 15
            # Ensure ROI is within bounds
            x1 = max(0, cx - roi_size)
            y1 = max(0, cy - roi_size)
            x2 = min(w, cx + roi_size)
            y2 = min(h, cy + roi_size)
            
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                # BGR format, Green is index 1
                mean_green = np.mean(roi[:, :, 1])
                self.green_signal.append(mean_green)
                self.times.append(time.time())
                
                # Estimate HR
                self._estimate_heart_rate()
        
        # 2. Normalize HR for graph (e.g., 60-140 bpm mapped to 0-1)
        # 75 bpm -> 15/80 = 0.1875
        hr_norm = (self.current_hr - 60.0) / 80.0
        hr_norm = max(0.0, min(1.0, hr_norm))
        
        # 3. Update history buffers
        self.history["stress"].append(stress_score)
        self.history["hr"].append(hr_norm)
        self.history["deception"].append(deception_prob)
        
        return {
            "hr_bpm": self.current_hr,
            "hr_norm": hr_norm,
            "history": self.history
        }
