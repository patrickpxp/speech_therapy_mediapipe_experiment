"""Utilities around MediaPipe Face Mesh to extract lip metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


@dataclass
class LipMetrics:
    lip_opening: float
    aspect_ratio: float
    roundness: float
    smile: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "lip_opening": float(self.lip_opening),
            "aspect_ratio": float(self.aspect_ratio),
            "roundness": float(self.roundness),
            "smile": float(self.smile),
        }


class LipFeatureExtractor:
    """Thin wrapper around MediaPipe's FaceMesh."""

    def __init__(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Annotate frame and return computed metrics."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            annotated = frame.copy()
            cv2.putText(annotated, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return annotated, {}

        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        lip_points_outer = np.array(
            [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIPS_OUTER],
            dtype=np.float32,
        )
        lip_points_inner = np.array(
            [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIPS_INNER],
            dtype=np.float32,
        )

        metrics = self._compute_metrics(lip_points_outer, lip_points_inner)
        annotated = self._draw_overlay(frame.copy(), lip_points_outer, lip_points_inner, metrics)
        return annotated, metrics.to_dict()

    @staticmethod
    def _compute_metrics(outer: np.ndarray, inner: np.ndarray) -> LipMetrics:
        vertical_pairs = [(0, 5), (1, 4), (2, 3), (6, 9), (7, 8)]
        vertical_distances = [np.linalg.norm(outer[a] - outer[b]) for a, b in vertical_pairs]
        lip_opening = float(np.mean(vertical_distances))

        lip_width = float(np.linalg.norm(outer[0] - outer[6]))
        aspect_ratio = lip_opening / (lip_width + 1e-6)

        area_outer = cv2.contourArea(outer)
        perimeter_outer = float(cv2.arcLength(outer, True))
        roundness = 4 * np.pi * area_outer / (perimeter_outer ** 2 + 1e-6)

        mouth_corners = outer[[0, 6]]
        smile = float(mouth_corners[1][1] - mouth_corners[0][1])

        return LipMetrics(
            lip_opening=lip_opening,
            aspect_ratio=aspect_ratio,
            roundness=roundness,
            smile=smile,
        )

    @staticmethod
    def _draw_overlay(frame: np.ndarray, outer: np.ndarray, inner: np.ndarray, metrics: LipMetrics) -> np.ndarray:
        for point in outer.astype(int):
            cv2.circle(frame, tuple(point), 1, (0, 255, 0), 2)
        for point in inner.astype(int):
            cv2.circle(frame, tuple(point), 1, (255, 0, 0), 2)

        info_lines = [
            f"Lip opening: {metrics.lip_opening:.1f}",
            f"Aspect ratio: {metrics.aspect_ratio:.2f}",
            f"Roundness: {metrics.roundness:.2f}",
        ]
        y = 30
        for line in info_lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30

        return frame
