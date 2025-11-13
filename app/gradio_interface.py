"""Gradio interface for the speech therapy experiment."""
from __future__ import annotations

import uuid
from typing import Dict, Tuple

import cv2
import gradio as gr
import numpy as np

from .audio import AudioPhonemeEstimator
from .calibration import CalibrationManager
from .evaluation import PhonemeEvaluationEngine
from .session import SessionStore
from .vision import VisionProcessor


session_store = SessionStore()
vision_processor = VisionProcessor()
audio_estimator = AudioPhonemeEstimator()
evaluation_engine = PhonemeEvaluationEngine()
calibration_managers: Dict[str, CalibrationManager] = {}

def _get_calibration_manager(session_id: str) -> CalibrationManager:
    manager = calibration_managers.get(session_id)
    if manager is None:
        manager = CalibrationManager()
        calibration_managers[session_id] = manager
    return manager

def start_session(_: str, phoneme: str) -> Tuple[str, str]:
    session_id = str(uuid.uuid4())
    session = session_store.get(session_id)
    session.target_phoneme = phoneme
    calibration_managers[session_id] = CalibrationManager()
    return session_id, f"Session created for phoneme {phoneme}."  

def process_frame(
    frame: np.ndarray,
    session_id: str,
    phoneme: str,
    calibrating: bool,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    if frame is None or session_id is None:
        return frame, "Waiting for video...", {}

    session = session_store.get(session_id)
    session.target_phoneme = phoneme
    try:
        metrics = vision_processor.process_frame(frame)
    except ValueError as err:
        return frame, f"{err}. Please align your face.", {}

    feedback = ""
    manager = _get_calibration_manager(session_id)
    if calibrating and not manager.is_complete:
        manager.add_measurement(
            lip_width=metrics.lip_width,
            lip_height=metrics.lip_height,
            lip_aspect_ratio=metrics.lip_aspect_ratio,
        )
        progress = len(manager.lip_widths)
        if manager.is_complete:
            snapshot = manager.snapshot()
            session.record_calibration(snapshot)
            feedback = "Calibration complete! Let's practice."
        else:
            feedback = f"Hold still... capturing baseline ({progress}/{manager.required_frames})."
    elif session.calibration:
        passed, rule_feedback = evaluation_engine.evaluate(
            phoneme=session.target_phoneme,
            metrics=metrics.to_dict(),
            baseline=session.calibration,
        )
        audio_likelihoods = session.latest_audio_likelihoods or {}
        if audio_likelihoods:
            audio_score = audio_likelihoods.get(session.target_phoneme, 0.0)
            audio_message = f"Audio confidence: {audio_score:.2f}."
        else:
            audio_message = "Capture audio to combine feedback."
        if passed:
            feedback = f"Great shape! {rule_feedback} {audio_message}"
        else:
            feedback = f"Keep trying: {rule_feedback} {audio_message}"
        session.record_attempt(
            phoneme=session.target_phoneme,
            lip_metrics=metrics.to_dict(),
            audio_likelihoods=audio_likelihoods,
            passed=passed,
            feedback=feedback,
        )
    else:
        feedback = "Start calibration to personalize feedback."

    annotated = vision_processor.annotate_frame(frame, metrics, feedback)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, feedback, metrics.to_dict()

def process_audio(audio: Tuple[int, np.ndarray], session_id: str, phoneme: str) -> Tuple[str, Dict[str, float]]:
    if audio is None or session_id is None:
        return "Waiting for audio...", {}

    session = session_store.get(session_id)
    session.target_phoneme = phoneme
    likelihoods = audio_estimator.estimate_likelihoods(audio)
    session.latest_audio_likelihoods = likelihoods
    target_score = likelihoods.get(phoneme, 0.0)
    if target_score > 0.7:
        guidance = "Great sound!"
    elif target_score > 0.4:
        guidance = "Almost there—focus on clarity."
    else:
        guidance = "Let's try that sound again." 
    message = f"{guidance} Target {phoneme}: {target_score:.2f}."
    return message, likelihoods

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Speech Therapy Pre-MVP") as demo:
        gr.Markdown(
            """
            # Speech Therapy Pre-MVP
            1. Create a session and calibrate your neutral mouth position.
            2. Practice phonemes with live visual and audio feedback.
            """
        )

        session_id_state = gr.State(value=None)

        with gr.Row():
            phoneme_selector = gr.Dropdown(
                choices=["AA", "EE", "MM"],
                value="AA",
                label="Target phoneme",
            )
            calibrate_toggle = gr.Checkbox(label="Calibrating", value=True)
            session_message = gr.Textbox(label="Session status", interactive=False)

        start_btn = gr.Button("Start new session")

        def _start(phoneme: str):
            session_id, message = start_session("", phoneme)
            return session_id, message, True

        start_btn.click(
            _start,
            inputs=[phoneme_selector],
            outputs=[session_id_state, session_message, calibrate_toggle],
        )

        with gr.Row():
            video_stream = gr.Video(source="webcam", streaming=True, label="Practice feed")
            annotated_view = gr.Image(label="Annotated feedback", type="numpy")
            audio_stream = gr.Audio(source="microphone", streaming=True, label="Audio stream")

        feedback_box = gr.Textbox(label="Visual feedback", interactive=False)
        audio_feedback_box = gr.Textbox(label="Audio feedback", interactive=False)
        metrics_display = gr.JSON(label="Lip metrics")
        audio_display = gr.JSON(label="Audio likelihoods")

        video_stream.stream(
            process_frame,
            inputs=[session_id_state, phoneme_selector, calibrate_toggle],
            outputs=[annotated_view, feedback_box, metrics_display],
        )

        audio_stream.stream(
            process_audio,
            inputs=[session_id_state, phoneme_selector],
            outputs=[audio_feedback_box, audio_display],
        )

    return demo

if __name__ == "__main__":
    build_interface().launch()
