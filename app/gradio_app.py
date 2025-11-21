"""Gradio client that orchestrates calibration, vision, and audio pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import gradio as gr
import numpy as np

from . import audio_processing
from .calibration import CalibrationSample, SessionState
from .phoneme_engine import evaluate_metrics, fuse_audio_visual
from .phoneme_rules import list_rules
from .vision import LipFeatureExtractor


EXTRACTOR = LipFeatureExtractor()
INSTRUCTIONS = Path(__file__).with_suffix(".instructions.txt")

if not INSTRUCTIONS.exists():
    INSTRUCTIONS.write_text(
        "\n".join(
            [
                "1. Position your face inside the camera frame with good lighting.",
                "2. Press 'Start calibration' and hold a neutral mouth for 5 seconds.",
                "3. Select a phoneme and practice while watching the feedback.",
                "4. Record audio samples to compare the sound to the mouth shape.",
            ]
        )
    )


def process_frame(
    frame: np.ndarray,
    state: SessionState,
    calibrating: bool,
) -> Tuple[np.ndarray, str, Dict[str, float], SessionState]:
    if frame is None:
        return frame, "Waiting for camera frame...", {}, state

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated, metrics = EXTRACTOR.process_frame(bgr_frame)

    if bool(calibrating) and metrics:
        state.calibration.add_sample(
            CalibrationSample(
                lip_opening=metrics["lip_opening"],
                aspect_ratio=metrics["aspect_ratio"],
                roundness=metrics["roundness"],
            )
        )
        feedback = "Calibrating... Keep a relaxed, neutral mouth."
        state.register_metrics(metrics)
        state.register_feedback(feedback)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), feedback, metrics, state

    feedback, visual_scores = evaluate_metrics(state.target_phoneme, metrics, state.calibration)
    state.register_metrics(metrics)
    state.register_feedback(feedback)

    fused = 0.0
    if state.audio_history:
        fused = fuse_audio_visual(visual_scores, state.audio_history[-1], state.target_phoneme)
        feedback += f" | Combined audio-visual score: {fused:.2f}"

    metrics_view = dict(metrics)
    metrics_view.update({"fused_score": fused})

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), feedback, metrics_view, state


def update_phoneme(state: SessionState, phoneme: str) -> Tuple[SessionState, str]:
    state.target_phoneme = phoneme
    return state, f"Target phoneme set to {phoneme}. Follow the on-screen guidance."


def reset_calibration(state: SessionState) -> Tuple[SessionState, str]:
    state.calibration.samples.clear()
    return state, "Calibration cleared. Press 'Start calibration' to record a new baseline."


def ingest_audio(audio: Tuple[int, np.ndarray], state: SessionState) -> Tuple[Dict[str, float], SessionState]:
    if audio is None:
        return {}, state
    descriptors = audio_processing.compute_audio_descriptors(audio)
    likelihoods = audio_processing.infer_phoneme_likelihoods(descriptors)
    state.store_audio_likelihoods(likelihoods)
    return likelihoods, state


def build_interface() -> gr.Blocks:
    state = SessionState()
    rules = list_rules()

    with gr.Blocks(title="Speech Therapy Pre-MVP") as demo:
        gr.Markdown("# Speech Therapy Coach (Pre-MVP)\n" + INSTRUCTIONS.read_text())

        session_state = gr.State(state)
        calibrating_flag = gr.State(False)

        with gr.Row():
            webcam = gr.Image(sources=["webcam"], streaming=True, mirror=True, shape=(480, 640), label="Webcam")
            annotated = gr.Image(label="Live feedback", interactive=False)

        feedback_box = gr.Textbox(label="Real-time feedback", interactive=False)
        metrics_json = gr.JSON(label="Metrics & Scores")

        with gr.Row():
            phoneme_selector = gr.Radio(list(rules.keys()), value="A", label="Target phoneme")
            start_calib = gr.Button("Start calibration", variant="primary")
            stop_calib = gr.Button("Stop calibration")
            reset_calib_btn = gr.Button("Reset calibration")

        with gr.Row():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Audio sample")
            audio_scores = gr.JSON(label="Audio phoneme likelihoods")

        def start_calibration(state: SessionState):
            return state, True, "Calibration started. Keep a neutral mouth and hold still."

        def stop_calibration(state: SessionState):
            return state, False, "Calibration stopped. Proceed with practice."

        phoneme_selector.change(update_phoneme, [session_state, phoneme_selector], [session_state, feedback_box])
        start_calib.click(start_calibration, [session_state], [session_state, calibrating_flag, feedback_box])
        stop_calib.click(stop_calibration, [session_state], [session_state, calibrating_flag, feedback_box])
        reset_calib_btn.click(reset_calibration, [session_state], [session_state, feedback_box])

        webcam.stream(process_frame, [webcam, session_state, calibrating_flag], [annotated, feedback_box, metrics_json, session_state])
        audio_input.change(ingest_audio, [audio_input, session_state], [audio_scores, session_state])

    return demo


if __name__ == "__main__":
    build_interface().launch()
