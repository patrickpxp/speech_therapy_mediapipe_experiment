"""Audio heuristics for phoneme likelihoods."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _rms_energy(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def _zero_crossing_rate(signal: np.ndarray) -> float:
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    return float(len(zero_crossings)) / len(signal)


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    if signal.size == 0:
        return 0.0
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window)
    magnitudes = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    numerator = np.sum(frequencies * magnitudes)
    denominator = np.sum(magnitudes) + 1e-6
    return float(numerator / denominator)


def compute_audio_descriptors(audio: Tuple[int, np.ndarray]) -> Dict[str, float]:
    """Return lightweight descriptors from the microphone buffer."""

    sample_rate, samples = audio
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    normalized = mono / (np.max(np.abs(mono)) + 1e-6)

    return {
        "rms": _rms_energy(normalized),
        "zcr": _zero_crossing_rate(normalized),
        "centroid": _spectral_centroid(normalized, sample_rate),
    }


def infer_phoneme_likelihoods(descriptors: Dict[str, float]) -> Dict[str, float]:
    """Map descriptors to heuristic phoneme likelihoods."""

    rms = descriptors.get("rms", 0.0)
    centroid = descriptors.get("centroid", 0.0)
    zcr = descriptors.get("zcr", 0.0)

    return {
        "A": float(min(1.0, rms * 2.5) * (1.0 if 700 <= centroid <= 1200 else 0.4)),
        "O": float(min(1.0, rms * 2.0) * (1.0 if 400 <= centroid <= 700 else 0.5)),
        "M": float(min(1.0, rms * 1.5) * (1.0 if centroid <= 500 and zcr < 0.05 else 0.3)),
        "F": float(min(1.0, rms * 3.0) * (1.0 if centroid >= 1000 and zcr > 0.08 else 0.3)),
    }
