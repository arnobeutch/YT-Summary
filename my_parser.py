"""Command-line parser for this script."""

import argparse
from pathlib import Path
from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    """Check if the given string is a valid URL."""
    try:
        result = urlparse(url)
    except ValueError:
        return False
    else:
        return all([result.scheme, result.netloc])


def parse_args() -> argparse.Namespace:
    """Define then parse command-line arguments.

    Returns:
        argparse.Namespace: the arguments discovered as attributes and their values.

    """
    parser = argparse.ArgumentParser(
        description=(
            "A python script to summarize YouTube videos, local videos or local audio,"
            " or existing transcript (.txt)."
            "\nThe script will output the video/audio text transcript"
            "\nand a markdown-formatted summary of the video, if asked (default: no)."
            "\nNote 1: The YT video must have either English or French subtitles."
            "\n\tThe script will output an error message if no such transcript is found."
            "\nNote 2: You must have an OpenAI API key to summarize, and credited tokens."
            "\n\tDeclare your key in an environment variable OPENAI_API_KEY prior to running."
            "\nUse the -h flag for help."
        ),
        usage=(
            "python main.py <youtube_video_url | path to file> [optional_arguments]"
            "\nExample: python main.py https://www.youtube.com/watch?v=VIDEO_ID"
            "\nOr: python main.py <path to audio/video file> [optional_arguments]"
            "\nExample: python main.py ./my_video.mp4"
        ),
        epilog="Be careful: summarizing burns OpenAI API tokens!",
    )
    parser.add_argument(
        "input_path",
        help="URL of the YouTube video or path to a local media / text file to summarize",
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
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: enable DEBUG-level logging (default: False)",
    )

    args = parser.parse_args()
    args.is_url = is_valid_url(args.input_path)
    args.is_file = Path(args.input_path).is_file()
    args.is_media_file = False
    args.is_text_file = False
    if args.is_file:
        media_extensions = {".mp4", ".mp3", ".wav", ".mkv", ".avi", ".webm", ".m4a"}
        text_extensions = {".txt", ".srt", ".vtt"}
        file_extension = Path(args.input_path).suffix.lower()
        if file_extension in media_extensions:
            args.is_media_file = True
        elif file_extension in text_extensions:
            args.is_text_file = True

    if not args.is_file and not args.is_url:
        err_msg = (
            f"Invalid input path: {args.input_path}. Must be a valid URL or an existing local file."
        )
        raise argparse.ArgumentTypeError(err_msg)

    return args
