"""
============================================================================
 CAMERA RETRY TESTS  (mock-based, no hardware required)
============================================================================
 Verifies the bounded consecutive-failure retry logic in main.py:run().

 Policy under test (configurable via config.py):
  - On cap.read() failure: increment consecutive counter,
    sleep CAMERA_READ_RETRY_DELAY_MS / 1000.0 s via time.sleep(), continue.
  - After CAMERA_READ_RETRY_LIMIT consecutive failures: log and exit cleanly.
  - After a successful read: reset the counter to 0.
  - Invalid / None frames are never passed downstream.

 Retry sleep is patched via unittest.mock so tests run instantly.
 cv2.waitKey() is only used in the normal GUI frame loop (keyboard handling)
 and is patched separately.

 NOTE: These tests use mocks only.  Actual webcam recovery behaviour depends
 on hardware and OS driver support and cannot be verified by unit tests.
============================================================================
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _build_system(mock_cap, mock_hud, retry_limit: int = 3):
    """
    Build a LieDetectionSystem with all I/O mocked and the retry limit
    patched in config.  setup_method / teardown_method must restore config.
    """
    config.CAMERA_READ_RETRY_LIMIT = retry_limit
    config.CAMERA_READ_RETRY_DELAY_MS = 50     # non-zero; sleep is patched out

    from main import LieDetectionSystem

    sys_ = LieDetectionSystem.__new__(LieDetectionSystem)
    sys_.cap = mock_cap
    sys_.running = True
    sys_._frame_times = []
    sys_._frame_count = 0
    sys_._fps = 0.0

    fd = MagicMock()
    fd.detect.return_value = []
    sys_.face_detector = fd

    sys_.feature_extractor = MagicMock()
    sys_.model = MagicMock()
    sys_.hud = mock_hud

    calib = MagicMock()
    calib.get_progress.return_value = 0.0
    calib.is_calibrating = False
    calib.get_remaining_seconds.return_value = 0
    calib.baseline = None
    calib.is_complete = False
    sys_.calibration = calib

    logger = MagicMock()
    logger.get_entry_count.return_value = 0
    sys_.logger = logger

    va = MagicMock()
    va.get_question.return_value = ""
    sys_.voice_analyzer = va

    pm = MagicMock()
    pm.get_current_question.return_value = None
    pm.get_protocol_summary.return_value = {}
    sys_.protocol_manager = pm

    sys_.show_mesh = False
    sys_.interview_state = 0
    sys_.final_verdict = None
    sys_.blink_detectors = {}
    sys_.movement_trackers = {}
    sys_.micro_analyzers = {}
    sys_.stress_analyzers = {}
    sys_.polygraph_analyzers = {}
    return sys_


class TestCameraReadRetry:
    """
    Regression tests for the bounded camera-read retry loop.
    All assertions use mock objects; no physical webcam is involved.

    time.sleep is patched in every test to prevent real delays.
    cv2.waitKey is patched only for the normal GUI loop (keyboard handling).
    """

    def setup_method(self):
        self._orig_limit = config.CAMERA_READ_RETRY_LIMIT
        self._orig_delay = config.CAMERA_READ_RETRY_DELAY_MS

    def teardown_method(self):
        config.CAMERA_READ_RETRY_LIMIT = self._orig_limit
        config.CAMERA_READ_RETRY_DELAY_MS = self._orig_delay

    # ── 1. One failure then success ──────────────────────────────────────────

    def test_one_failure_then_success_continues_processing(self):
        """
        A single failed read followed by a good frame must NOT exit the loop;
        HUD must be called at least once proving the recovered frame was processed.
        """
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (False, None),          # failure  → time.sleep() called
            (True, fake_frame),     # recovery → cv2.waitKey(1) called for 'q'
        ]

        mock_hud = MagicMock()
        mock_hud.render.return_value = fake_frame

        with patch("time.sleep") as mock_sleep, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            system = _build_system(mock_cap, mock_hud, retry_limit=3)
            system.run()

        assert mock_hud.render.call_count >= 1, (
            "HUD should have rendered the recovered frame"
        )
        # sleep must have been called exactly once (one failure)
        mock_sleep.assert_called_once_with(50 / 1000.0)

    # ── 2. Several consecutive failures below limit ──────────────────────────

    def test_consecutive_failures_below_limit_do_not_terminate(self):
        """
        Two consecutive failures with limit=3 must survive and render the
        subsequent recovered frame.
        """
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (False, None),           # fail 1 → sleep
            (False, None),           # fail 2 → sleep
            (True, fake_frame),      # recovered
        ]

        mock_hud = MagicMock()
        mock_hud.render.return_value = fake_frame

        with patch("time.sleep") as mock_sleep, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            system = _build_system(mock_cap, mock_hud, retry_limit=3)
            system.run()

        assert mock_hud.render.call_count >= 1
        # sleep called once per failure that was below the limit
        assert mock_sleep.call_count == 2

    # ── 3. Exceeding retry limit exits cleanly ───────────────────────────────

    def test_exceeding_retry_limit_exits_cleanly(self):
        """
        CAMERA_READ_RETRY_LIMIT consecutive failures must cause the loop to
        exit via shutdown() releasing camera resources.
        """
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)   # always fails

        mock_hud = MagicMock()
        mock_hud.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("time.sleep"), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=0), \
             patch("cv2.destroyAllWindows"):

            system = _build_system(mock_cap, mock_hud, retry_limit=3)
            system.run()

        mock_cap.release.assert_called_once()

    # ── 4. Success resets counter ────────────────────────────────────────────

    def test_success_resets_consecutive_failure_counter(self):
        """
        A successful read must reset the counter to 0 so that a subsequent
        failure episode starts fresh.
        Pattern [fail, ok, fail, ok, q] with limit=2 must stay alive.
        """
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (False, None),          # episode 1 – failure 1  → sleep
            (True, fake_frame),     # episode 1 – recovered  → waitKey → '?'
            (False, None),          # episode 2 – failure 1  → sleep (NOT cumulative)
            (True, fake_frame),     # episode 2 – recovered  → waitKey → 'q'
        ]

        mock_hud = MagicMock()
        mock_hud.render.return_value = fake_frame

        # waitKey returns non-exit on first good frame, 'q' on second
        with patch("time.sleep") as mock_sleep, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", side_effect=[255, ord("q")]), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            system = _build_system(mock_cap, mock_hud, retry_limit=2)
            system.run()

        # Both recovered frames must have been rendered
        assert mock_hud.render.call_count >= 2
        # sleep called once per failure (one per episode)
        assert mock_sleep.call_count == 2

    # ── 5. Camera released on every termination path ─────────────────────────

    def test_camera_released_on_retry_limit_exceeded(self):
        """
        cap.release() and face_detector.release() must both be called when
        the retry limit is exceeded.
        """
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)

        mock_hud = MagicMock()
        mock_hud.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("time.sleep"), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=0), \
             patch("cv2.destroyAllWindows"):

            system = _build_system(mock_cap, mock_hud, retry_limit=2)
            system.run()

        mock_cap.release.assert_called_once()
        system.face_detector.release.assert_called_once()

    # ── 6. No-face frame safe during recovery ────────────────────────────────

    def test_no_face_frame_safe_during_recovery(self):
        """
        A frame recovered after a failure with no detected face must render
        the HUD with faces_data=[] without crashing.
        """
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (False, None),           # failure  → sleep
            (True, fake_frame),      # recovered, no face
        ]

        mock_hud = MagicMock()
        mock_hud.render.return_value = fake_frame

        with patch("time.sleep"), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            system = _build_system(mock_cap, mock_hud, retry_limit=3)
            system.run()

        assert mock_hud.render.called
        call_args = mock_hud.render.call_args
        # hud.render(frame, faces_data, **kwargs) — faces_data is positional arg[1]
        # or passed as keyword; handle both.
        kwarg_faces = call_args.kwargs.get("faces_data")
        if kwarg_faces is not None:
            faces_arg = kwarg_faces
        elif len(call_args.args) > 1:
            faces_arg = call_args.args[1]
        else:
            faces_arg = None
        assert faces_arg == [], f"Expected empty faces_data, got: {faces_arg}"

    # ── 7. time.sleep receives correct duration ──────────────────────────────

    def test_sleep_receives_correct_duration(self):
        """
        time.sleep() must be called with CAMERA_READ_RETRY_DELAY_MS / 1000.0
        — not with the raw millisecond value.
        """
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Set a recognisable non-zero delay
        config.CAMERA_READ_RETRY_DELAY_MS = 80

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (False, None),
            (True, fake_frame),
        ]

        mock_hud = MagicMock()
        mock_hud.render.return_value = fake_frame

        with patch("time.sleep") as mock_sleep, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.flip", side_effect=lambda f, _: f):

            system = _build_system(mock_cap, mock_hud, retry_limit=3)
            # Override the delay that _build_system set
            import config as cfg
            cfg.CAMERA_READ_RETRY_DELAY_MS = 80
            system.run()

        mock_sleep.assert_called_once_with(80 / 1000.0)

    # ── 8. cv2.waitKey NOT called during retry sleep ─────────────────────────

    def test_cv2_waitkey_not_used_for_retry_sleep(self):
        """
        cv2.waitKey() must NOT be called during the retry-sleep path;
        it is only used in the normal frame loop for keyboard handling.
        With a read sequence of [fail, limit-exceeded], the GUI frame loop
        is never reached so cv2.waitKey must not be called at all.
        """
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)   # always fails

        mock_hud = MagicMock()
        mock_hud.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("time.sleep"), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey") as mock_waitkey, \
             patch("cv2.destroyAllWindows"):

            system = _build_system(mock_cap, mock_hud, retry_limit=2)
            system.run()

        mock_waitkey.assert_not_called()
