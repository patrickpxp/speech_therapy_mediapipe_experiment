"""Session management primitives for the speech therapy prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CalibrationSnapshot:
    """Baseline lip metrics captured during calibration."""

    lip_width: float
    lip_height: float
    lip_aspect_ratio: float


@dataclass
class PhonemeAttempt:
    """Result of a single phoneme attempt."""

    phoneme: str
    lip_metrics: Dict[str, float]
    audio_likelihoods: Dict[str, float]
    passed: bool
    feedback: str


@dataclass
class SpeechTherapySession:
    """In-memory representation of a learner session."""

    session_id: str
    target_phoneme: str = "AA"
    calibration: Optional[CalibrationSnapshot] = None
    attempts: List[PhonemeAttempt] = field(default_factory=list)
    latest_audio_likelihoods: Dict[str, float] = field(default_factory=dict)

    def record_calibration(self, snapshot: CalibrationSnapshot) -> None:
        self.calibration = snapshot

    def record_attempt(
        self,
        phoneme: str,
        lip_metrics: Dict[str, float],
        audio_likelihoods: Dict[str, float],
        passed: bool,
        feedback: str,
    ) -> None:
        self.attempts.append(
            PhonemeAttempt(
                phoneme=phoneme,
                lip_metrics=lip_metrics,
                audio_likelihoods=audio_likelihoods,
                passed=passed,
                feedback=feedback,
            )
        )
        self.latest_audio_likelihoods = audio_likelihoods


class SessionStore:
    """Naive in-memory store for active learner sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SpeechTherapySession] = {}

    def get(self, session_id: str) -> SpeechTherapySession:
        if session_id not in self._sessions:
            self._sessions[session_id] = SpeechTherapySession(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
