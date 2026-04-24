"""Command-line parser for this script."""

import argparse
from pathlib import Path
from urllib.parse import urlparse

_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mp3", ".wav", ".mkv", ".avi", ".webm", ".m4a"}
)
_TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".srt", ".vtt"})


def is_valid_url(url: str) -> bool:
    """Check if the given string is a valid URL."""
    try:
        result = urlparse(url)
    except ValueError:
        return False
    else:
        return all([result.scheme, result.netloc])


def classify_input(path: str) -> dict[str, bool]:
    """Return type flags for a single input path.

    Raises :exc:`argparse.ArgumentTypeError` if the path is neither a
    valid URL nor an existing local file.
    """
    is_url = is_valid_url(path)
    is_file = Path(path).is_file()
    is_media_file = False
    is_text_file = False
    if is_file:
        ext = Path(path).suffix.lower()
        is_media_file = ext in _MEDIA_EXTENSIONS
        is_text_file = ext in _TEXT_EXTENSIONS
    if not is_file and not is_url:
        err_msg = f"Invalid input path: {path}. Must be a valid URL or an existing local file."
        raise argparse.ArgumentTypeError(err_msg)
    return {
        "is_url": is_url,
        "is_file": is_file,
        "is_media_file": is_media_file,
        "is_text_file": is_text_file,
    }


def parse_args() -> argparse.Namespace:
    """Define then parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Summarize YouTube videos, local audio/video files, or existing transcripts."),
        usage="yt-summary <url | path> [url | path ...] [options]",
    )
    parser.add_argument(
        "input_path",
        nargs="+",
        help="YouTube URL(s) or path(s) to local media / text file(s)",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices={"en", "fr"},
        default=None,
        help=(
            "Preferred source language ('en' or 'fr'). Used as a hint when "
            "picking a YouTube caption track, and to force whisper's "
            "transcription language. If omitted: autodetect. The summary "
            "always tracks the source language (English fallback for "
            "anything other than en/fr)."
        ),
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Diarize the audio (identify speakers) in the transcript (default: False)",
    )
    parser.add_argument(
        "-s",
        "--summarize",
        action="store_true",
        default=False,
        help="Output a summary of the video transcript (default: False)",
    )
    parser.add_argument(
        "--with-openai",
        "--with_openai",  # legacy alias — underscored form kept for backward compat
        dest="with_openai",
        action="store_true",
        default=False,
        help=(
            "Shortcut for --llm-provider openai (default: False). "
            "Kept for backward compatibility — prefer --llm-provider."
        ),
    )
    parser.add_argument(
        "--model-size",
        dest="model_size",
        choices={"tiny", "base", "small", "medium", "large"},
        default=None,
        help=(
            "Whisper model size for local transcription. "
            "Default: env WHISPER_MODEL_SIZE, or 'small'."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        dest="llm_provider",
        choices={"openai", "openrouter", "ollama"},
        default=None,
        help=(
            "LLM backend for summarization. "
            "Default: env LLM_PROVIDER, or 'openai'. "
            "Overrides --with-openai if both are given."
        ),
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help=(
            "Specific model name for the chosen provider (e.g. 'gpt-4o', "
            "'anthropic/claude-4.7-sonnet', 'gemma4:e4b'). "
            "Default: env LLM_MODEL, or the provider's default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Where transcripts and summaries land. Default: env OUTPUT_DIR, or ./results.",
    )
    parser.add_argument(
        "--downloads-dir",
        dest="downloads_dir",
        type=Path,
        default=None,
        help="Where downloaded YT audio is cached. Default: env DOWNLOADS_DIR, or ./downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download audio and re-transcribe even if cached outputs exist.",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        default=False,
        help=(
            "Also write .srt and .vtt subtitle files alongside the .txt "
            "transcript (whisper transcription only, not for YT captions or "
            "diarized output)."
        ),
    )
    parser.add_argument(
        "--transcript-only",
        dest="transcript_only",
        action="store_true",
        default=False,
        help=(
            "Stop after writing the transcript (and subtitles, if --subtitles); "
            "skip summarization. Implies --summarize is ignored."
        ),
    )
    parser.add_argument(
        "--summary-mode",
        dest="summary_mode",
        choices={"meeting", "source", "auto"},
        default=None,
        help=(
            "Summary scenario: meeting (multi-speaker discussion), source "
            "(lecture/article/commentary; tags facts vs opinion vs speculation), "
            "or auto (default — heuristic). Default: env SUMMARY_MODE, or 'auto'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "Print what the pipeline would do (input type, model, output dir) "
            "without downloading, transcribing, or summarizing."
        ),
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: enable DEBUG-level logging (default: False)",
    )

    args = parser.parse_args()

    # Validate every input path eagerly so the user gets an error before any
    # slow work begins.
    for path in args.input_path:
        classify_input(path)  # raises ArgumentTypeError on invalid input

    return args
