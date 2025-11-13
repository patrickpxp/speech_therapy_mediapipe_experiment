"""Vision utilities for lip landmark extraction using MediaPipe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np

LIP_LANDMARKS = {
    "upper_inner": 13,
    "lower_inner": 14,
    "left_corner": 61,
    "right_corner": 291,
}


@dataclass
class LipMetrics:
    lip_width: float
    lip_height: float
    lip_aspect_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "lip_width": self.lip_width,
            "lip_height": self.lip_height,
            "lip_aspect_ratio": self.lip_aspect_ratio,
        }


class VisionProcessor:
    """Wraps MediaPipe Face Mesh to compute lip metrics and overlays."""

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def _extract_lip_landmarks(self, frame: np.ndarray) -> Dict[str, Tuple[float, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected")
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        coords = {}
        for name, idx in LIP_LANDMARKS.items():
            landmark = landmarks[idx]
            coords[name] = (landmark.x * w, landmark.y * h)
        return coords

    def _compute_metrics(self, coords: Dict[str, Tuple[float, float]]) -> LipMetrics:
        left = np.array(coords["left_corner"])
        right = np.array(coords["right_corner"])
        upper = np.array(coords["upper_inner"])
        lower = np.array(coords["lower_inner"])

        lip_width = float(np.linalg.norm(right - left))
        lip_height = float(np.linalg.norm(lower - upper))
        lip_aspect_ratio = lip_height / (lip_width + 1e-6)
        return LipMetrics(lip_width=lip_width, lip_height=lip_height, lip_aspect_ratio=lip_aspect_ratio)

    def annotate_frame(self, frame: np.ndarray, metrics: LipMetrics, feedback: str) -> np.ndarray:
        annotated = frame.copy()
        h, w, _ = annotated.shape
        cv2.putText(
            annotated,
            feedback,
            (int(0.05 * w), int(0.1 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        overlay_lines = [
            f"Width: {metrics.lip_width:.1f}",
            f"Height: {metrics.lip_height:.1f}",
            f"Aspect: {metrics.lip_aspect_ratio:.2f}",
        ]
        for idx, line in enumerate(overlay_lines):
            cv2.putText(
                annotated,
                line,
                (int(0.05 * w), int(0.15 * h) + 25 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_frame(self, frame: np.ndarray) -> LipMetrics:
        coords = self._extract_lip_landmarks(frame)
        return self._compute_metrics(coords)
