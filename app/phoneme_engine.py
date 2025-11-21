"""Phoneme evaluation that combines vision metrics and audio likelihoods."""
from __future__ import annotations

from typing import Dict, Tuple

from .calibration import CalibrationProfile, apply_calibration
from .phoneme_rules import get_rule


def evaluate_metrics(
    phoneme: str,
    metrics: Dict[str, float],
    calibration: CalibrationProfile,
) -> Tuple[str, Dict[str, float]]:
    """Compare calibrated metrics against the rule thresholds."""

    if not metrics:
        return "No face detected. Try adjusting lighting and camera position.", {}

    calibrated = apply_calibration(metrics, calibration)
    rule = get_rule(phoneme)

    feedback_parts = []
    scores: Dict[str, float] = {}

    lip_opening = calibrated["lip_opening"]
    aspect_ratio = calibrated["aspect_ratio"]
    roundness = calibrated["roundness"]

    def within(value: float, min_value: float, max_value: float | None) -> float:
        if max_value is None:
            return 1.0 if value >= min_value else value / (min_value + 1e-6)
        if value < min_value:
            return value / (min_value + 1e-6)
        if value > max_value:
            return max_value / (value + 1e-6)
        return 1.0

    opening_score = within(lip_opening, rule.min_lip_opening, rule.max_lip_opening)
    aspect_score = within(aspect_ratio, rule.min_aspect_ratio, rule.max_aspect_ratio)
    roundness_score = within(roundness, rule.min_roundness, rule.max_roundness)

    scores.update(
        lip_opening=opening_score,
        aspect_ratio=aspect_score,
        roundness=roundness_score,
    )

    if opening_score < 0.8:
        feedback_parts.append("Open your mouth wider." if lip_opening < rule.min_lip_opening else "Relax your mouth slightly.")
    if aspect_score < 0.8:
        feedback_parts.append("Adjust jaw to match the target shape (height/width mismatch).")
    if roundness_score < 0.8:
        feedback_parts.append("Purse your lips more for a rounded shape." if roundness < rule.min_roundness else "Relax lip rounding.")

    if not feedback_parts:
        feedback = f"Great job! That looks like the phoneme {phoneme.upper()}"
    else:
        feedback = " ".join(feedback_parts)

    return feedback, scores


def fuse_audio_visual(
    visual_scores: Dict[str, float],
    audio_likelihoods: Dict[str, float],
    phoneme: str,
) -> float:
    """Return a simple fused score between audio and visual evidence."""

    if not visual_scores:
        return 0.0

    audio_score = audio_likelihoods.get(phoneme.upper(), 0.0)
    visual_score = sum(visual_scores.values()) / len(visual_scores)
    return float(0.6 * visual_score + 0.4 * audio_score)
