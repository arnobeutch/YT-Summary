"""Per-source handlers: URL, local media, local text file.

Each ``handle_*`` returns a :class:`Transcript` capturing what was produced
and where it came from. ``main.py`` is then a thin orchestrator that picks
the right handler, writes the transcript to disk, and optionally summarizes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import prepare_local_transcript as plt
import prepare_yt_audio as pya
import prepare_yt_transcript as pytt
import summarize_transcript as st
from formatting import sanitize_filename, wrap_transcript
from my_logger import my_logger
from my_settings import Settings
from prepare_yt_transcript import TranscriptUnavailableError

TranscriptSource = Literal["yt_caption", "whisper", "file"]


@dataclass(frozen=True)
class Transcript:
    """In-memory representation of a transcript ready to be written / summarized."""

    text: str
    language: str
    title: str
    source: TranscriptSource
    diarized: bool


def handle_url(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Fetch a YT transcript; on failure download audio + run whisper."""
    video_id = pya.extract_video_id(args.input_path)
    my_logger.debug(f"Video ID: {video_id}")
    try:
        text = pytt.get_youtube_transcript(video_id)
    except TranscriptUnavailableError as exc:
        my_logger.info(
            f"No YouTube transcript available ({exc.reason}: {exc}) — "
            f"falling back to local transcription.",
        )
        audio_path, raw_title = pya.download_youtube_audio(
            args.input_path,
            settings.downloads_dir,
        )
        if args.diarize:
            transcribed_text, language = plt.transcribe_audio_with_diarization(
                str(audio_path),
                model_size=settings.whisper_model_size,
            )
        else:
            transcribed_text, language = plt.transcribe_audio(
                str(audio_path),
                model_size=settings.whisper_model_size,
            )
        return Transcript(
            text=transcribed_text,
            language=language,
            title=sanitize_filename(raw_title),
            source="whisper",
            diarized=args.diarize,
        )

    my_logger.info("Transcript retrieved successfully")
    return Transcript(
        text=text,
        language=args.language,
        title=sanitize_filename(pya.fetch_video_title(args.input_path)),
        source="yt_caption",
        diarized=False,
    )


def handle_media(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Transcribe a local media file via whisper (and optionally pyannote)."""
    if args.language is not None:
        my_logger.warning(
            "Language argument is ignored for local media files, auto-detected instead.",
        )
    title = sanitize_filename(Path(args.input_path).stem)
    if args.diarize:
        text, language = plt.transcribe_video_file_with_diarization(
            args.input_path,
            model_size=settings.whisper_model_size,
        )
    else:
        text, language = plt.transcribe_video_file(
            args.input_path,
            model_size=settings.whisper_model_size,
        )
    return Transcript(
        text=text,
        language=language,
        title=title,
        source="whisper",
        diarized=args.diarize,
    )


def handle_text(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Read a pre-existing transcript from disk."""
    _ = settings  # not used yet (reserved for future language-detect)
    text = Path(args.input_path).read_text(encoding="utf8")
    return Transcript(
        text=text,
        language=args.language,
        title=sanitize_filename(Path(args.input_path).stem),
        source="file",
        diarized=False,
    )


def write_transcript_file(transcript: Transcript, settings: Settings) -> Path:
    """Write the transcript text to ``<output_dir>/<title> [diarized] transcript.txt``."""
    suffix = " diarized transcript" if transcript.diarized else " transcript"
    p = settings.output_dir / f"{transcript.title}{suffix}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        wrap_transcript(transcript.text, diarize=transcript.diarized, width=settings.wrap_width),
        encoding="utf8",
    )
    my_logger.info(f"Transcript written to {p}")
    return p


def summarize(transcript: Transcript, args: argparse.Namespace, settings: Settings) -> None:
    """Dispatch to the OpenAI or RAG summarizer."""
    if args.with_openai:
        st.summarize_transcript_with_openai(
            transcript.text,
            args.input_path,
            transcript.title,
            transcript.language,
        )
    else:
        st.summarize_transcript_with_rag(
            transcript.text,
            transcript.title,
            transcript.language,
            model=settings.llm_model or settings.ollama_model,
        )
