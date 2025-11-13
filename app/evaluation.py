"""Rule-based phoneme evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .session import CalibrationSnapshot


@dataclass
class PhonemeRule:
    """Defines geometric thresholds for a phoneme."""

    min_height_scale: float = 1.0
    min_width_scale: float = 1.0
    max_height_scale: float = 10.0
    description: str = ""

    def validate(self, metrics: Dict[str, float], baseline: CalibrationSnapshot) -> Tuple[bool, str]:
        width_ratio = metrics["lip_width"] / (baseline.lip_width + 1e-6)
        height_ratio = metrics["lip_height"] / (baseline.lip_height + 1e-6)
        aspect_ratio = metrics["lip_aspect_ratio"] / (baseline.lip_aspect_ratio + 1e-6)

        if height_ratio < self.min_height_scale:
            return False, "Try opening your mouth wider."
        if width_ratio < self.min_width_scale:
            return False, "Stretch your lips sideways a bit more."
        if height_ratio > self.max_height_scale:
            return False, "Relax your mouth slightly."
        return True, self.description or "Great job!"


class PhonemeEvaluationEngine:
    """Evaluates phonemes using rule thresholds and calibration."""

    DEFAULT_RULES: Dict[str, PhonemeRule] = {
        "AA": PhonemeRule(
            min_height_scale=1.2,
            min_width_scale=0.9,
            max_height_scale=2.5,
            description="Nice open 'ah' sound!",
        ),
        "EE": PhonemeRule(
            min_height_scale=0.8,
            min_width_scale=1.1,
            max_height_scale=1.6,
            description="Great smile shape for 'ee'!",
        ),
        "MM": PhonemeRule(
            min_height_scale=0.4,
            min_width_scale=0.8,
            max_height_scale=1.1,
            description="Good closed lips for 'mm'.",
        ),
    }

    def __init__(self, rules: Dict[str, PhonemeRule] | None = None) -> None:
        self.rules = rules or self.DEFAULT_RULES

    def evaluate(
        self, phoneme: str, metrics: Dict[str, float], baseline: CalibrationSnapshot
    ) -> Tuple[bool, str]:
        rule = self.rules.get(phoneme, PhonemeRule(description="Looks good!"))
        return rule.validate(metrics, baseline)
