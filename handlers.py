# pyright: reportUnknownVariableType=false
"""Per-source handlers: URL, local media, local text file.

Each ``handle_*`` returns a :class:`Transcript` capturing what was produced
and where it came from. ``main.py`` is then a thin orchestrator that picks
the right handler, writes the transcript to disk, and optionally summarizes.

The ``# pyright`` header above suppresses ``reportUnknownVariableType`` across
this file — ``langdetect``'s public ``detect`` returns an annotated-but-
``Unknown`` type, and the pattern propagates everywhere we touch it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from langdetect import LangDetectException, detect

import prepare_local_transcript as plt
import prepare_yt_audio as pya
import prepare_yt_transcript as pytt
from formatting import sanitize_filename, wrap_transcript
from language import derive_summary_language, derive_whisper_summary_language
from model import Transcript
from my_logger import my_logger
from my_settings import Settings
from prepare_yt_transcript import TranscriptUnavailableError
from summarizers import make_summarizer


def handle_url(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Fetch a YT transcript honoring the language ladder; whisper-fallback if absent."""
    video_id = pya.extract_video_id(args.input_path)
    my_logger.debug(f"Video ID: {video_id}")
    requested_lang: str | None = args.language
    force: bool = bool(getattr(args, "force", False))

    try:
        track = pytt.get_youtube_transcript(video_id, requested_lang=requested_lang)
    except TranscriptUnavailableError as exc:
        my_logger.info(
            f"No YouTube transcript available ({exc.reason}: {exc}) — "
            f"falling back to local transcription.",
        )
        audio_path, raw_title = pya.download_youtube_audio(
            args.input_path,
            settings.downloads_dir,
            force=force,
        )
        title = sanitize_filename(raw_title)

        cached = _try_load_cached_transcript(title, settings, diarize=args.diarize, force=force)
        if cached is not None:
            return Transcript(
                text=cached,
                language=derive_whisper_summary_language(
                    requested_lang or "en",
                    requested_lang,
                ),
                title=title,
                source="whisper",
                diarized=args.diarize,
            )

        if args.diarize:
            transcribed_text, used_lang = plt.transcribe_audio_with_diarization(
                str(audio_path),
                model_size=settings.whisper_model_size,
                language=requested_lang,
            )
        else:
            transcribed_text, used_lang = plt.transcribe_audio(
                str(audio_path),
                model_size=settings.whisper_model_size,
                language=requested_lang,
            )
        summary_lang = derive_whisper_summary_language(used_lang, requested_lang)
        return Transcript(
            text=transcribed_text,
            language=summary_lang,
            title=title,
            source="whisper",
            diarized=args.diarize,
        )

    summary_lang = derive_summary_language(track.lang, requested_lang)
    my_logger.info(
        f"Caption track: {track.kind} '{track.lang}'; summary language: {summary_lang}",
    )
    return Transcript(
        text=track.text,
        language=summary_lang,
        title=sanitize_filename(pya.fetch_video_title(args.input_path)),
        source="yt_manual" if track.kind == "manual" else "yt_auto",
        diarized=False,
    )


def _try_load_cached_transcript(
    title: str,
    settings: Settings,
    *,
    diarize: bool,
    force: bool,
) -> str | None:
    """Return cached transcript text if a matching ``.txt`` already exists."""
    if force:
        return None
    suffix = " diarized transcript" if diarize else " transcript"
    cached = settings.output_dir / f"{title}{suffix}.txt"
    if not cached.exists():
        return None
    my_logger.info(f"Using cached transcript at {cached}")
    return cached.read_text(encoding="utf8")


def handle_media(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Transcribe a local media file via whisper (optionally pyannote-diarized).

    ``--language`` (if set) forces whisper to that language and becomes the
    summary language. Otherwise whisper autodetects; if the detection lands
    on en/fr the summary follows; otherwise summary is forced to English.
    """
    title = sanitize_filename(Path(args.input_path).stem)
    requested_lang: str | None = args.language
    if args.diarize:
        text, used_lang = plt.transcribe_video_file_with_diarization(
            args.input_path,
            model_size=settings.whisper_model_size,
            language=requested_lang,
        )
    else:
        text, used_lang = plt.transcribe_video_file(
            args.input_path,
            model_size=settings.whisper_model_size,
            language=requested_lang,
        )
    summary_lang = derive_whisper_summary_language(used_lang, requested_lang)
    my_logger.info(f"Transcribed in '{used_lang}'; summary language: {summary_lang}")
    return Transcript(
        text=text,
        language=summary_lang,
        title=title,
        source="whisper",
        diarized=args.diarize,
    )


def _detect_text_language(text: str) -> str:
    """Best-effort language detection; defaults to ``"en"`` on failure."""
    try:
        detected = detect(text)
    except LangDetectException:
        my_logger.warning("Could not detect text language; defaulting to 'en'")
        return "en"
    return cast(str, detected)


def handle_text(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Read a pre-existing transcript from disk."""
    _ = settings  # reserved for future use
    text = Path(args.input_path).read_text(encoding="utf8")
    requested_lang: str | None = args.language
    detected = requested_lang if requested_lang else _detect_text_language(text)
    summary_lang = derive_whisper_summary_language(detected, requested_lang)
    my_logger.info(f"Text-file language: {detected}; summary language: {summary_lang}")
    return Transcript(
        text=text,
        language=summary_lang,
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
    """Dispatch to the configured Summarizer backend."""
    summarizer = make_summarizer(settings)
    summarizer.summarize(transcript, input_path=args.input_path)
