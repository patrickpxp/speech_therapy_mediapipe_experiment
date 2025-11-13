"""Session-level calibration helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional


@dataclass
class CalibrationSample:
    lip_opening: float
    aspect_ratio: float
    roundness: float


@dataclass
class CalibrationProfile:
    """Aggregated calibration statistics for a learner."""

    samples: List[CalibrationSample] = field(default_factory=list)

    def add_sample(self, sample: CalibrationSample) -> None:
        self.samples.append(sample)

    def is_ready(self, min_samples: int = 15) -> bool:
        return len(self.samples) >= min_samples

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"lip_opening": 0.0, "aspect_ratio": 0.0, "roundness": 0.0}

        return {
            "lip_opening": mean(s.lip_opening for s in self.samples),
            "aspect_ratio": mean(s.aspect_ratio for s in self.samples),
            "roundness": mean(s.roundness for s in self.samples),
        }


@dataclass
class SessionState:
    """Mutable session container shared between video/audio pipelines."""

    target_phoneme: str = "A"
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
    audio_history: List[Dict[str, float]] = field(default_factory=list)
    latest_metrics: Optional[Dict[str, float]] = None
    latest_feedback: str = ""

    def register_metrics(self, metrics: Dict[str, float]) -> None:
        self.latest_metrics = metrics

    def register_feedback(self, feedback: str) -> None:
        self.latest_feedback = feedback

    def store_audio_likelihoods(self, likelihoods: Dict[str, float]) -> None:
        self.audio_history.append(likelihoods)


def apply_calibration(metrics: Dict[str, float], profile: CalibrationProfile) -> Dict[str, float]:
    """Normalize metrics using calibration profile means."""

    if not profile.samples:
        return metrics

    summary = profile.summary()
    normalized = {
        "lip_opening": metrics["lip_opening"] - summary["lip_opening"],
        "aspect_ratio": metrics["aspect_ratio"] - summary["aspect_ratio"],
        "roundness": metrics["roundness"] - summary["roundness"],
    }
    normalized.update({k: v for k, v in metrics.items() if k not in normalized})
    return normalized
