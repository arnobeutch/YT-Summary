"""Retrieve the text transcript from a local media file."""

import tempfile
from pathlib import Path

import ffmpeg
import torch.cuda
import tqdm
import whisper

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
        ).run(quiet=False, overwrite_output=True)
        # ).run(quiet=True, overwrite_output=True)
    return tmp_audio_path


def get_device() -> str:
    """Return 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_audio(audio_file: str, model_size: str = "base") -> tuple[str,str]:
    """Transcribe audio using Whisper and returns the transcribed text and the detected language."""
    my_logger.info(f"Transcribing audio file: {audio_file}")

    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    patch_whisper_progress_bar()
    model = whisper.load_model(model_size, device=device)

    # Detect language
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)

    my_logger.info(f"\tDetected language: {detected_lang}")

    result = model.transcribe(audio_file, fp16=False, language=detected_lang)   # fp16 param to remove annoying warning
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
