"""Audio phoneme heuristics for the prototype."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class AudioPhonemeEstimator:
    """Produces lightweight phoneme likelihoods from audio clips."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def _ensure_mono(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return data.mean(axis=1)
        return data

    def _extract_features(self, audio: Tuple[int, np.ndarray]) -> Dict[str, float]:
        sr, data = audio
        if data.size == 0:
            return {"energy": 0.0, "zcr": 0.0, "dominant_freq": 0.0}
        data = self._ensure_mono(data.astype(np.float32))
        energy = float(np.sqrt(np.mean(np.square(data))))
        zero_crossings = np.mean(np.abs(np.diff(np.sign(data)))) / 2.0
        spectrum = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        dominant_freq = float(freqs[np.argmax(np.abs(spectrum))])
        return {"energy": energy, "zcr": float(zero_crossings), "dominant_freq": dominant_freq}

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def estimate_likelihoods(self, audio: Tuple[int, np.ndarray]) -> Dict[str, float]:
        features = self._extract_features(audio)
        energy = features["energy"]
        zcr = features["zcr"]
        dominant_freq = features["dominant_freq"]

        aa_score = self._sigmoid(5 * energy - 1.0)
        ee_score = self._sigmoid(4 * zcr + 0.5)
        mm_score = self._sigmoid(-5 * zcr + 1.0 - 0.002 * dominant_freq)
        return {
            "AA": float(aa_score),
            "EE": float(ee_score),
            "MM": float(mm_score),
        }
