"""Audio phoneme heuristics for the prototype."""
from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np


AudioInput = Union[
    Tuple[int, np.ndarray],
    Tuple[int, np.ndarray, Any],
    Sequence[Any],
    Dict[str, Any],
    str,
    Path,
]


class AudioPhonemeEstimator:
    """Produces lightweight phoneme likelihoods from audio clips."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def _ensure_mono(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return data.mean(axis=1)
        return data

    def _load_audio_file(self, path: str | Path) -> Tuple[int, np.ndarray]:
        """Lightweight WAV reader for filepath inputs."""
        try:
            with wave.open(str(path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.getnframes()
                raw = wav_file.readframes(frames)
        except (wave.Error, FileNotFoundError, OSError):
            return self.sample_rate, np.empty(0, dtype=np.float32)

        if sample_width not in (1, 2, 4):
            return self.sample_rate, np.empty(0, dtype=np.float32)

        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        data = np.frombuffer(raw, dtype=dtype_map[sample_width])
        if sample_width == 1:
            data = data.astype(np.int16) - 128

        if channels > 1:
            data = data.reshape(-1, channels)
        return sample_rate, data.astype(np.float32)

    def _normalize_audio(self, audio: AudioInput | None) -> Tuple[int, np.ndarray]:
        """Accept multiple gradio audio payload shapes and emit (sr, data)."""
        if audio is None:
            return self.sample_rate, np.empty(0, dtype=np.float32)

        sample_rate = self.sample_rate
        data: Any = None

        if isinstance(audio, (str, Path)):
            return self._load_audio_file(audio)
        if isinstance(audio, dict):
            sample_rate = int(audio.get("sample_rate") or sample_rate)
            data = audio.get("data")
        elif isinstance(audio, tuple):
            if len(audio) >= 2:
                sample_rate = int(audio[0] or sample_rate)
                data = audio[1]
        elif isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, bytearray)):
            if len(audio) >= 2:
                sample_rate = int(audio[0] or sample_rate)
                data = audio[1]

        if data is None:
            return sample_rate, np.empty(0, dtype=np.float32)

        return sample_rate, np.asarray(data, dtype=np.float32)

    def _extract_features(self, audio: AudioInput | None) -> Dict[str, float]:
        sr, data = self._normalize_audio(audio)
        if data.size == 0:
            return {"energy": 0.0, "zcr": 0.0, "dominant_freq": 0.0}
        data = self._ensure_mono(data)
        energy = float(np.sqrt(np.mean(np.square(data))))
        zero_crossings = np.mean(np.abs(np.diff(np.sign(data)))) / 2.0
        spectrum = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        dominant_freq = float(freqs[np.argmax(np.abs(spectrum))])
        return {"energy": energy, "zcr": float(zero_crossings), "dominant_freq": dominant_freq}

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def estimate_likelihoods(self, audio: AudioInput | None) -> Dict[str, float]:
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
