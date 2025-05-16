"""Retrieve the text transcript from a local media file."""

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import ffmpeg
import numpy as np
import torch.cuda
import torchaudio
import tqdm
import whisper
from dotenv import load_dotenv

# from huggingface_hub import login
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

# from typing import List, Tuple
from my_logger import my_logger


class TqdmProgressBar:
    """Replacement for whisper.utils.ProgressBar that uses tqdm."""

    def __init__(self, total: int) -> None:
        """Initialize the progress bar."""
        self._bar = tqdm(total=total, unit="segment")

    def update(self, n: int = 1) -> None:
        """Update the progress bar by n segments."""
        self._bar.update(n)

    def close(self) -> None:
        """Close the progress bar."""
        self._bar.close()


def patch_whisper_progress_bar() -> None:
    """Monkey-patch whisper's ProgressBar with tqdm-based one."""
    whisper.utils.ProgressBar = TqdmProgressBar


def extract_audio(input_file: str, output_format: str = "wav") -> str:
    """Extract audio from a video file and returns the path to the audio file."""
    if not Path(input_file).exists():
        err_msg = f"File not found: {input_file}"
        raise FileNotFoundError(err_msg)

    # file_ext = Path(input_file).suffix.lower()
    # if file_ext not in [".mp4", ".webm"]:
    #     err_msg = f"Unsupported file format: {file_ext}"
    #     raise ValueError(err_msg)

    my_logger.info(f"Extracting audio from {input_file} to {output_format} format")
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp_audio_file:
        tmp_audio_path = tmp_audio_file.name

    ffmpeg.input(input_file).output(
        tmp_audio_path,
        format=output_format,
        ac=1,
        ar="16000",
        ).run(quiet=True, overwrite_output=True)
    return tmp_audio_path


def get_device() -> str:
    """Return 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_language(audio_file: str, model: whisper.Whisper, device: str) -> str:
    """Detect the language of the audio file using Whisper."""
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def transcribe_audio(audio_file: str, model_size: str = "base") -> tuple[str,str]:
    """Transcribe audio using Whisper and returns the transcribed text and the detected language."""
    my_logger.info(f"Transcribing audio file: {audio_file}")

    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    patch_whisper_progress_bar()
    model = whisper.load_model(model_size, device=device)

    # Detect language
    detected_lang = detect_language(audio_file, model, device)
    my_logger.info(f"\tDetected language: {detected_lang}")

    result = model.transcribe(
        audio_file, fp16=(device == "cuda"), language=detected_lang,
    )
    return result["text"], detected_lang


def transcribe_video_file(video_file: str, model_size: str = "base") -> tuple[str, str]:
    """Full pipeline: Extract audio from video, transcribe it, return text."""
    my_logger.info(f"Processing: {video_file}")
    audio_path = extract_audio(video_file)
    try:
        transcription, language = transcribe_audio(audio_path, model_size=model_size)
    finally:
        Path(audio_path).unlink()  # Cleanup temp audio file
    return transcription, language


def diarize_speakers(audio_file: str) -> list[tuple[str, Segment]]:
    """Diarize speakers in the audio file using PyAnnote."""
    my_logger.info(f"Diarizing speakers in: {audio_file}")
    load_dotenv()  # Load environment variables from .env file
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # noqa: N806
    if not HUGGINGFACE_TOKEN:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)
    # Needs token with access to pyannote models:
    # - https://huggingface.co/pyannote/speaker-diarization-3.1
    # - https://huggingface.co/pyannote/segmentation-3.0
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    diarization = pipeline(audio_file)
    speaker_segments = [
        (str(label), segment) for segment, _, label in diarization.itertracks(yield_label=True)
    ]
    return speaker_segments  # noqa: RET504


def load_audio_slice(audio_path: str, start: float, end: float) -> np.ndarray:
    """Load a slice of audio between `start` and `end` seconds."""
    waveform, sample_rate = torchaudio.load(audio_path)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    sliced_waveform = waveform[:, start_sample:end_sample]
    return sliced_waveform.mean(dim=0).numpy()  # convert to mono np.array


def group_speaker_segments(
    diarized_segments: list[tuple[str, Segment]],
    max_gap: float = 1.0,
) -> list[tuple[str, Segment]]:
    """Group consecutive segments from the same speaker that are close in time.

    Args:
        diarized_segments (list): List of (speaker, Segment) tuples.
        max_gap (float): Max gap in seconds to allow merging.

    Returns:
        list: List of (speaker, merged Segment) tuples.

    """
    grouped_segments = []
    last_speaker = None
    current_start = None
    current_end = None

    for speaker, segment in diarized_segments:
        if (
            speaker == last_speaker
            and current_end is not None
            and (segment.start - current_end) <= max_gap
        ):
            current_end = segment.end  # extend current segment
        else:
            if last_speaker is not None:
                grouped_segments.append(
                    (last_speaker, Segment(current_start, current_end)),
                )
            last_speaker = speaker
            current_start = segment.start
            current_end = segment.end

    if last_speaker is not None:
        grouped_segments.append((last_speaker, Segment(current_start, current_end)))

    return grouped_segments


def detect_speech_segments(audio_file: str) -> Timeline:
    """Run voice activity detection (VAD) and return speech regions as a Timeline.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        pyannote.core.Timeline: Detected speech segments.

    """
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)

    # Needs token with access to gated pyannote models:
    # - https://huggingface.co/pyannote/voice-activity-detection
    # - https://huggingface.co/pyannote/segmentation
    vad_pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=token,
    )
    vad_result = vad_pipeline(audio_file)
    return vad_result.get_timeline().support()


def transcribe_audio_with_diarization(
    audio_file: str, model_size: str = "base",
) -> tuple[str, str]:
    """Transcribe audio with speaker diarization."""
    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    patch_whisper_progress_bar()
    model = whisper.load_model(model_size, device=device)

    language = detect_language(audio_file, model, device)
    my_logger.info(f"Detected language: {language}")

    # Diarize speakers
    diarized_segments = diarize_speakers(audio_file)
    speech_timeline = detect_speech_segments(audio_file)

    # Keep only diarized segments that intersect with actual speech
    filtered_segments = [
        (speaker, segment)
        for speaker, segment in diarized_segments
        if speech_timeline.crop(segment)  # returns non-empty Timeline if overlaps
    ]

    # Group segments from the same speaker
    grouped_segments = group_speaker_segments(filtered_segments, max_gap=1.0)
    full_text = []
    MIN_SEGMENT_DURATION = 1.5  # seconds  # noqa: N806
    for speaker, segment in grouped_segments:
        # Skip segments that are too short (silence or noise)
        if segment.end - segment.start < MIN_SEGMENT_DURATION:
            continue
        sliced_audio = load_audio_slice(audio_file, segment.start, segment.end)
        segment_result = model.transcribe(
            sliced_audio, fp16=(device == "cuda"), language=language,
        )
        text = segment_result["text"].strip()
        if not text:    # Skip empty transcriptions
            continue
        full_text.append(f"{speaker}: {text}")

    return "\n".join(full_text), language


def transcribe_video_file_with_diarization(
    video_file: str, model_size: str = "base",
) -> tuple[str, str]:
    """Full pipeline: Extract audio from video, transcribe it with diarization."""
    my_logger.info(f"Processing with diarization: {video_file}")
    audio_path = extract_audio(video_file)
    try:
        transcription, language = transcribe_audio_with_diarization(
            audio_path, model_size=model_size,
        )
    finally:
        Path(audio_path).unlink()
    return transcription, language
