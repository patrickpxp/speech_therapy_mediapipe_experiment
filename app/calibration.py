"""Calibration helpers for lip metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .session import CalibrationSnapshot


@dataclass
class CalibrationManager:
    """Collects metrics until calibration converges."""

    required_frames: int = 30
    lip_widths: List[float] = None
    lip_heights: List[float] = None
    lip_aspect_ratios: List[float] = None

    def __post_init__(self) -> None:
        self.lip_widths = []
        self.lip_heights = []
        self.lip_aspect_ratios = []

    def add_measurement(self, lip_width: float, lip_height: float, lip_aspect_ratio: float) -> None:
        self.lip_widths.append(lip_width)
        self.lip_heights.append(lip_height)
        self.lip_aspect_ratios.append(lip_aspect_ratio)

    @property
    def is_complete(self) -> bool:
        return len(self.lip_widths) >= self.required_frames

    def snapshot(self) -> CalibrationSnapshot:
        if not self.is_complete:
            raise ValueError("Calibration not complete")
        return CalibrationSnapshot(
            lip_width=sum(self.lip_widths) / len(self.lip_widths),
            lip_height=sum(self.lip_heights) / len(self.lip_heights),
            lip_aspect_ratio=sum(self.lip_aspect_ratios) / len(self.lip_aspect_ratios),
        )

    def reset(self) -> None:
        self.lip_widths.clear()
        self.lip_heights.clear()
        self.lip_aspect_ratios.clear()
