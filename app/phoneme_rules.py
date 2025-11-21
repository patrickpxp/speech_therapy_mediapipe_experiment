"""Rule-based phoneme definitions for lip geometry thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class PhonemeRule:
    """Container for the heuristic thresholds tied to a phoneme.

    Attributes:
        name: Human readable name of the phoneme or viseme.
        min_lip_opening: Minimal average vertical distance between lip landmarks
            required to consider the mouth ``open`` enough for the phoneme.
        max_lip_opening: Maximum acceptable opening. ``None`` disables the upper
            bound.
        min_aspect_ratio: Minimal mouth aspect ratio (height / width).
        max_aspect_ratio: Optional upper bound for the aspect ratio.
        min_roundness: Minimal roundedness (0..1) for rounded vowels such as ``O``.
        max_roundness: Optional upper bound for roundedness.
        audio_hint: Short description of the expected audio profile that can help
            the audio scoring layer assign likelihoods.
    """

    name: str
    min_lip_opening: float = 0.0
    max_lip_opening: float | None = None
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float | None = None
    min_roundness: float = 0.0
    max_roundness: float | None = None
    audio_hint: str = ""

    def to_dict(self) -> Dict[str, float | str]:
        """Serialize the rule to a dictionary for UI consumption."""
        return {
            "name": self.name,
            "min_lip_opening": self.min_lip_opening,
            "max_lip_opening": self.max_lip_opening,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_roundness": self.min_roundness,
            "max_roundness": self.max_roundness,
            "audio_hint": self.audio_hint,
        }


_PHONEME_RULES: Mapping[str, PhonemeRule] = {
    "A": PhonemeRule(
        name="Open vowel A",
        min_lip_opening=12.0,
        max_lip_opening=None,
        min_aspect_ratio=0.50,
        max_aspect_ratio=0.85,
        min_roundness=0.1,
        max_roundness=0.5,
        audio_hint="Strong first formant energy between 700-1100 Hz.",
    ),
    "O": PhonemeRule(
        name="Rounded vowel O",
        min_lip_opening=8.0,
        max_lip_opening=18.0,
        min_aspect_ratio=0.35,
        max_aspect_ratio=0.70,
        min_roundness=0.55,
        max_roundness=0.95,
        audio_hint="Lower first formant (400-600 Hz) with visible lip rounding.",
    ),
    "M": PhonemeRule(
        name="Bilabial consonant M",
        min_lip_opening=1.0,
        max_lip_opening=5.0,
        min_aspect_ratio=0.20,
        max_aspect_ratio=0.50,
        min_roundness=0.2,
        max_roundness=0.6,
        audio_hint="Low-frequency nasal energy with small mouth opening.",
    ),
    "F": PhonemeRule(
        name="Labiodental consonant F/V",
        min_lip_opening=2.0,
        max_lip_opening=10.0,
        min_aspect_ratio=0.30,
        max_aspect_ratio=0.70,
        min_roundness=0.1,
        max_roundness=0.5,
        audio_hint="Mid-frequency fricative noise above 1 kHz.",
    ),
}


def get_rule(phoneme: str) -> PhonemeRule:
    """Return the rule associated with ``phoneme``.

    Raises:
        KeyError: If no rule is registered for the phoneme.
    """

    normalized = phoneme.strip().upper()
    if normalized not in _PHONEME_RULES:
        raise KeyError(f"Unknown phoneme '{phoneme}'.")
    return _PHONEME_RULES[normalized]


def list_rules() -> Dict[str, Dict[str, float | str]]:
    """Expose all phoneme rules as serializable dictionaries."""

    return {phoneme: rule.to_dict() for phoneme, rule in _PHONEME_RULES.items()}
