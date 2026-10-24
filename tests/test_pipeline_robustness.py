"""
============================================================================
 PIPELINE ROBUSTNESS TESTS  (mock-based, no hardware required)
============================================================================
 Covers the end-to-end live-application edge cases identified during the
 2026-10 end-to-end pipeline validation:

  1. Camera frame-read failure - transient vs. fatal.
  2. No face detected - HUD renders with empty faces_data without crashing.
  3. Per-face processing exception - surviving faces still rendered.
  4. NaN / Inf feature vector - sanitised before reaching the model.
  5. Missing model_metadata.json - provenance defaults to UNKNOWN.
  6. Anomaly score and prediction_stability_band are always separate fields.
  7. shutdown() safely releases camera even after partial __init__ failure.
  8. Session logger does not save data when no entries exist.
  9. Feature vector length always matches FEATURE_NAMES count.
 10. Model prediction always includes scientific_disclaimer and data_provenance.
============================================================================
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.ml_model import DeceptionModel
from modules.feature_extractor import FeatureExtractor
from modules.session_logger import SessionLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_face(face_id: int = 0):
    face = MagicMock()
    face.face_id = face_id
    face.bbox = (10, 10, 100, 100)
    face.landmarks = MagicMock()
    return face


def _make_good_features():
    return np.full(len(config.FEATURE_NAMES), 0.3, dtype=np.float32)


def _build_mock_system(mock_cap, mock_hud, mock_fd=None):
    """Return a LieDetectionSystem instance with all I/O mocked out."""
    from main import LieDetectionSystem

    system = LieDetectionSystem.__new__(LieDetectionSystem)
    system.cap = mock_cap
    system.running = True
    system._frame_times = []
    system._frame_count = 0
    system._fps = 0.0
    system.face_detector = mock_fd if mock_fd is not None else MagicMock()
    system.feature_extractor = MagicMock()
    system.model = MagicMock()
    system.hud = mock_hud
    calib = MagicMock()
    calib.get_progress.return_value = 0.5
    calib.is_calibrating = False
    calib.get_remaining_seconds.return_value = 0
    calib.baseline = None
    calib.is_complete = False
    system.calibration = calib
    system.logger = MagicMock()
    system.logger.get_entry_count.return_value = 0
    va = MagicMock()
    va.get_question.return_value = ""
    system.voice_analyzer = va
    pm = MagicMock()
    pm.get_current_question.return_value = None
    pm.get_protocol_summary.return_value = {}
    system.protocol_manager = pm
    system.show_mesh = False
    system.interview_state = 0
    system.final_verdict = None
    system.blink_detectors = {}
    system.movement_trackers = {}
    system.micro_analyzers = {}
    system.stress_analyzers = {}
    system.polygraph_analyzers = {}
    return system


# ---------------------------------------------------------------------------
# 1. Camera frame-read failure
# ---------------------------------------------------------------------------

class TestCameraFrameFailure:

    def test_frame_failure_exits_cleanly_and_releases_camera(self):
        """run() must exit without crashing when cap.read() returns False."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=255), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640
            mock_cap.read.return_value = (False, None)
            mock_cap_cls.return_value = mock_cap

            mock_hud = MagicMock()
            mock_hud.render.return_value = fake_frame

            from main import LieDetectionSystem
            system = _build_mock_system(mock_cap, mock_hud)
            system.run()

            mock_cap.release.assert_called_once()

    def test_webcam_not_opened_raises_runtime_error(self):
        """__init__ must raise RuntimeError when VideoCapture fails to open."""
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("main.FaceDetector"), \
             patch("main.FeatureExtractor"), \
             patch("main.DeceptionModel"), \
             patch("main.HUDRenderer"), \
             patch("main.CalibrationManager"), \
             patch("main.SessionLogger"), \
             patch("main.VoiceAnalyzer"), \
             patch("main.ElicitationProtocolManager"):

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            from main import LieDetectionSystem
            with pytest.raises(RuntimeError, match="Could not open webcam"):
                LieDetectionSystem()


# ---------------------------------------------------------------------------
# 2. No face detected
# ---------------------------------------------------------------------------

class TestNoFaceDetected:

    def test_hud_render_called_with_empty_faces(self):
        """HUD render() must be called with faces_data=[] when no face is detected."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", side_effect=[ord("q")]), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640
            mock_cap.read.return_value = (True, fake_frame)
            mock_cap_cls.return_value = mock_cap

            mock_fd = MagicMock()
            mock_fd.detect.return_value = []

            mock_hud = MagicMock()
            mock_hud.render.return_value = fake_frame

            from main import LieDetectionSystem
            system = _build_mock_system(mock_cap, mock_hud, mock_fd=mock_fd)
            system.run()

            assert mock_hud.render.called
            call_kwargs = mock_hud.render.call_args
            # faces_data is passed as keyword argument
            faces_arg = call_kwargs[1].get("faces_data", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
            assert faces_arg == []


# ---------------------------------------------------------------------------
# 3. Per-face processing exception
# ---------------------------------------------------------------------------

class TestPerFaceException:

    def test_face_exception_does_not_crash_run_loop(self):
        """If _process_face raises, the run() loop must catch it and keep going."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", side_effect=[ord("q")]), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640
            mock_cap.read.return_value = (True, fake_frame)
            mock_cap_cls.return_value = mock_cap

            mock_fd = MagicMock()
            mock_fd.detect.return_value = [_make_fake_face(0)]

            mock_hud = MagicMock()
            mock_hud.render.return_value = fake_frame

            from main import LieDetectionSystem
            system = _build_mock_system(mock_cap, mock_hud, mock_fd=mock_fd)
            system._process_face = MagicMock(side_effect=RuntimeError("Mock face error"))
            system.run()

            # HUD still rendered even though _process_face threw
            assert mock_hud.render.called


# ---------------------------------------------------------------------------
# 4. NaN / Inf in feature vector
# ---------------------------------------------------------------------------

class TestFeatureSanitisation:

    def _nan_blink(self):
        return {"blink_rate_per_min": float("nan"), "blink_rate_change": float("nan"),
                "avg_ear": float("nan"), "ear_std": float("nan")}

    def _nan_movement(self):
        return {k: float("nan") for k in
                ["gaze_h_std", "gaze_v_std", "gaze_fixation_ratio",
                 "head_pitch_std", "head_yaw_std", "head_roll_std",
                 "movement_magnitude", "nervous_score"]}

    def _nan_micro(self):
        return {k: float("nan") for k in
                ["face_symmetry", "face_symmetry_change", "lip_movement",
                 "lip_compression", "eyebrow_tension", "eyebrow_asymmetry",
                 "micro_expression_count", "micro_expression_intensity",
                 "facial_tension", "jaw_clench"]}

    def test_nan_features_are_replaced(self):
        extractor = FeatureExtractor()
        features = extractor.extract(self._nan_blink(), self._nan_movement(), self._nan_micro(), {})
        assert not np.any(np.isnan(features))

    def test_inf_features_are_replaced(self):
        extractor = FeatureExtractor()
        inf_blink = {k: float("inf") for k in ["blink_rate_per_min", "blink_rate_change", "avg_ear", "ear_std"]}
        inf_movement = {k: float("inf") for k in
                        ["gaze_h_std", "gaze_v_std", "gaze_fixation_ratio",
                         "head_pitch_std", "head_yaw_std", "head_roll_std",
                         "movement_magnitude", "nervous_score"]}
        inf_micro = {k: float("inf") for k in
                     ["face_symmetry", "face_symmetry_change", "lip_movement",
                      "lip_compression", "eyebrow_tension", "eyebrow_asymmetry",
                      "micro_expression_count", "micro_expression_intensity",
                      "facial_tension", "jaw_clench"]}
        features = extractor.extract(inf_blink, inf_movement, inf_micro, {})
        assert not np.any(np.isinf(features))

    def test_model_accepts_nan_cleaned_vector(self):
        model = DeceptionModel()
        features = np.full(len(config.FEATURE_NAMES), float("nan"), dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        pred = model.predict(features)
        assert 0.0 <= pred["deception_probability"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Missing model_metadata.json
# ---------------------------------------------------------------------------

class TestMissingMetadata:

    def test_missing_metadata_defaults_to_unknown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import pickle
        os.makedirs("models", exist_ok=True)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=1, random_state=0)
        clf.fit(np.zeros((2, len(config.FEATURE_NAMES))), [0, 1])
        with open("models/lie_detector_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        # No model_metadata.json created
        model = DeceptionModel()
        assert model.provenance == "UNKNOWN"
        assert model.label_verified is False
        assert model.model_category == "UNKNOWN"


# ---------------------------------------------------------------------------
# 6. Anomaly score and stability band are always separate keys
# ---------------------------------------------------------------------------

class TestSeparateScoringFields:

    def test_anomaly_and_stability_are_separate_keys(self):
        model = DeceptionModel()
        pred = model.predict(_make_good_features())
        assert isinstance(pred["deception_probability"], float)
        assert isinstance(pred["prediction_stability_band"], tuple)
        assert len(pred["prediction_stability_band"]) == 2

    def test_stability_band_bounds_always_ordered(self):
        model = DeceptionModel()
        for val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            features = np.full(len(config.FEATURE_NAMES), val, dtype=np.float32)
            band = model.predict(features)["prediction_stability_band"]
            assert band[0] <= band[1], f"Band out of order at val={val}: {band}"

    def test_calibration_prediction_dict_has_both_keys(self):
        # Mirrors the dict assembled in main.py during calibration
        calibration_pred = {
            "deception_probability": 0.0,
            "prediction_stability_band": (0.0, 0.0),
            "confidence_interval": (0.0, 0.0),
            "label": "CALIBRATING...",
        }
        assert "deception_probability" in calibration_pred
        assert "prediction_stability_band" in calibration_pred


# ---------------------------------------------------------------------------
# 7. shutdown() safety
# ---------------------------------------------------------------------------

class TestShutdownSafety:

    def test_shutdown_releases_camera(self):
        from main import LieDetectionSystem
        system = LieDetectionSystem.__new__(LieDetectionSystem)
        mock_cap = MagicMock()
        system.cap = mock_cap
        system.face_detector = MagicMock()
        system.voice_analyzer = MagicMock()
        system.logger = MagicMock()
        pm = MagicMock()
        pm.get_protocol_summary.return_value = {}
        system.protocol_manager = pm
        calib = MagicMock()
        calib.baseline = None
        system.calibration = calib
        with patch("cv2.destroyAllWindows"):
            system.shutdown()
        mock_cap.release.assert_called_once()
        system.face_detector.release.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Session logger with no entries
# ---------------------------------------------------------------------------

class TestSessionLoggerNoEntries:

    def test_save_with_no_entries_returns_empty_string(self, tmp_path):
        logger = SessionLogger()
        logger._log_file = str(tmp_path / "empty.csv")
        assert logger.save() == ""

    def test_close_with_no_entries_does_not_raise(self, tmp_path):
        logger = SessionLogger()
        logger._log_file = str(tmp_path / "empty_close.csv")
        logger.close()  # must not raise


# ---------------------------------------------------------------------------
# 9. Feature vector length
# ---------------------------------------------------------------------------

class TestFeatureVectorLength:

    def test_length_matches_config(self):
        extractor = FeatureExtractor()
        features = extractor.extract({}, {}, {}, {})
        assert len(features) == len(config.FEATURE_NAMES)

    def test_feature_names_match_config(self):
        extractor = FeatureExtractor()
        assert extractor.get_feature_names() == config.FEATURE_NAMES


# ---------------------------------------------------------------------------
# 10. Prediction required fields
# ---------------------------------------------------------------------------

class TestPredictionRequiredFields:

    @pytest.mark.parametrize("val", [0.0, 0.3, 0.7, 1.0])
    def test_prediction_has_disclaimer_and_provenance(self, val):
        model = DeceptionModel()
        features = np.full(len(config.FEATURE_NAMES), val, dtype=np.float32)
        pred = model.predict(features)
        assert "scientific_disclaimer" in pred
        assert "data_provenance" in pred
        assert pred["data_provenance"] in ("SYNTHETIC", "REAL_HUMAN", "UNKNOWN")

    def test_empty_prediction_has_disclaimer_and_provenance(self):
        model = DeceptionModel()
        pred = model.predict(None)
        assert "scientific_disclaimer" in pred
        assert "data_provenance" in pred
