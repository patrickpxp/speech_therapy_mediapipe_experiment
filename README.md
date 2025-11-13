# Speech Therapy Pre-MVP

This repository contains a Gradio-based pre-MVP experience for validating the
core pieces of a speech therapy assistant for children. The app focuses on
combining MediaPipe lip landmark tracking with simple audio heuristics and a
rule-based phoneme evaluator.

## Features

- **Real-time guidance** – live webcam feed with overlayed lip landmarks and
  instructions driven by heuristic thresholds.
- **Calibration flow** – capture a neutral mouth baseline so that subsequent
  metrics are normalized per learner.
- **Rule-based phoneme engine** – configurable thresholds for vowels and
  consonants such as A, O, M, and F.
- **Audio heuristics** – lightweight descriptors (RMS energy, spectral centroid,
  zero crossing rate) converted into phoneme likelihoods that can be fused with
  the visual scores.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.gradio_app
```

Once the app launches, follow the calibration instructions, select a phoneme,
and observe the real-time feedback while practicing. Capture microphone samples
whenever you want to compare audio cues with the mouth shape.
