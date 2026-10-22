"""
============================================================================
 FACE DETECTOR MODULE
============================================================================
 Handles face detection and facial landmark extraction using MediaPipe
 Face Mesh. Supports multi-face detection and provides 468+ landmark points
 for each detected face.
============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import config


@dataclass
class FaceData:
    """Container for detected face data."""
    face_id: int = 0
    landmarks: np.ndarray = field(default_factory=lambda: np.array([]))
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 0.0
    raw_landmarks: object = None  # MediaPipe NormalizedLandmarkList

    def get_landmark(self, idx: int) -> np.ndarray:
        """Get a specific landmark point by index."""
        if len(self.landmarks) > idx:
            return self.landmarks[idx]
        return np.array([0.0, 0.0, 0.0])


class FaceDetector:
    """
    Real-time face detection and landmark extraction using MediaPipe Face Mesh.
    
    Provides 468 facial landmark points per face with optional iris tracking
    (478 points when refine_landmarks is enabled).
    """

    def __init__(self):
        """Initialize MediaPipe Face Mesh with configuration parameters."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize face mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

        self._prev_faces: List[FaceData] = []
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> List[FaceData]:
        """
        Detect faces and extract landmarks from a video frame.

        Args:
            frame: BGR image from OpenCV capture.

        Returns:
            List of FaceData objects for each detected face.
        """
        self._frame_count += 1
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Improve performance by marking image as not writeable
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        faces: List[FaceData] = []

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                # Convert landmarks to numpy array of (x, y, z) in pixel coords
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in face_landmarks.landmark
                ])

                # Compute bounding box from landmarks
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x_min = int(max(0, np.min(x_coords) - 10))
                y_min = int(max(0, np.min(y_coords) - 10))
                x_max = int(min(w, np.max(x_coords) + 10))
                y_max = int(min(h, np.max(y_coords) + 10))
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                face_data = FaceData(
                    face_id=face_id,
                    landmarks=landmarks,
                    bbox=bbox,
                    confidence=1.0,  # MediaPipe doesn't expose per-face confidence
                    raw_landmarks=face_landmarks,
                )
                faces.append(face_data)

        self._prev_faces = faces
        return faces

    def draw_landmarks(self, frame: np.ndarray, face: FaceData,
                       draw_mesh: bool = True, draw_contour: bool = True,
                       draw_iris: bool = True) -> np.ndarray:
        """
        Draw facial landmarks on the frame with customizable visualization.

        Args:
            frame: Image to draw on.
            face: FaceData containing landmark information.
            draw_mesh: Whether to draw the full face mesh.
            draw_contour: Whether to draw face contour outline.
            draw_iris: Whether to draw iris landmarks.

        Returns:
            Frame with landmarks drawn.
        """
        if face.raw_landmarks is None:
            return frame

        # Create custom drawing specs to match the HUD theme
        mesh_spec = self.mp_drawing.DrawingSpec(color=config.HUD_COLORS["landmark_mesh"], thickness=1, circle_radius=1)
        contour_spec = self.mp_drawing.DrawingSpec(color=config.HUD_COLORS["primary"], thickness=1, circle_radius=1)

        # Draw the full mesh tessellation (subtle appearance)
        if draw_mesh:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face.raw_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_spec,
            )

        # Draw face contour
        if draw_contour:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face.raw_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=contour_spec,
            )

        # Draw iris landmarks
        if draw_iris and config.REFINE_LANDMARKS:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face.raw_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=contour_spec,
            )

        return frame

    def draw_key_landmarks(self, frame: np.ndarray, face: FaceData) -> np.ndarray:
        """
        Draw only key analysis landmarks (eyes, lips, eyebrows) with
        color-coded dots for the HUD display.
        """
        color = config.HUD_COLORS["landmark_dot"]
        radius = 1

        # Draw key landmark groups
        key_indices = (
            config.LEFT_EYE_LANDMARKS + config.RIGHT_EYE_LANDMARKS +
            config.LEFT_EYEBROW + config.RIGHT_EYEBROW +
            [config.NOSE_TIP] + config.CHIN
        )

        for idx in key_indices:
            if idx < len(face.landmarks):
                pt = face.landmarks[idx][:2].astype(int)
                cv2.circle(frame, tuple(pt), radius, color, -1, cv2.LINE_AA)

        return frame

    def get_face_region(self, frame: np.ndarray, face: FaceData,
                        padding: int = 20) -> Optional[np.ndarray]:
        """Extract the face region from the frame with optional padding."""
        x, y, w, h = face.bbox
        h_img, w_img = frame.shape[:2]

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        if x2 - x1 > 0 and y2 - y1 > 0:
            return frame[y1:y2, x1:x2]
        return None

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
