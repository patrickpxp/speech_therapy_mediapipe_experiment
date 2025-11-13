# Speech Therapy Mediapipe Experiment

This repository hosts a pre-MVP prototype that validates the speech therapy concept using a Python-only stack powered by Gradio, MediaPipe, and lightweight audio heuristics.

## Features

- **Guided practice loop** – A Gradio interface streams webcam frames and microphone audio while providing real-time feedback text overlays.
- **Personalized calibration** – Learners capture ~30 baseline frames to normalize lip width/height before practicing phonemes.
- **Rule-based phoneme checks** – Simple thresholds derived from the calibration snapshot detect whether the current mouth shape matches the selected phoneme.
- **Audio likelihoods** – A tiny audio pipeline computes energy, zero-crossing rate, and dominant frequency to estimate heuristic phoneme scores.
- **Session tracking** – In-memory session store captures attempts so the backend logic can be reused by future clients.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Gradio demo:
   ```bash
   python -m app.gradio_interface
   ```
3. Click **Start new session**, keep the calibration checkbox enabled, and hold a neutral mouth for the first few seconds to build the baseline. Once calibration finishes, practice the target phoneme and observe the feedback updates.

> **Note**: This prototype favors readability over production hardening. It keeps state in memory and relies on webcam/microphone access provided by Gradio.

## Project Layout

- `app/vision.py` – MediaPipe face mesh integration with lip metrics and on-frame annotation helpers.
- `app/audio.py` – Minimal audio feature extraction and heuristic phoneme likelihood estimation.
- `app/evaluation.py` – Rule-based phoneme thresholds with friendly feedback strings.
- `app/calibration.py` – Collector that aggregates lip measurements into a baseline snapshot.
- `app/session.py` – Dataclasses for session state, calibration snapshots, and attempt history.
- `app/gradio_interface.py` – End-to-end Gradio Blocks app tying vision, audio, and evaluation together.

These modules are designed so the backend logic can be mounted behind FastAPI or another service while experimenting with different clients.
