"""Command-line parser for this script."""

import argparse
from pathlib import Path
from urllib.parse import urlparse

from my_logger import my_logger


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
            "A python script to summarize YouTube videos, local videos or local audio, or existing transcript (.txt)."
            "\nThe script will output the video/audio text transcript"
            "\nand a markdown-formatted summary of the video, if asked (default: no)."
            "\nNote 1: The YT video must have either English or French subtitles."
            "\n\tThe script will output an error message if no such transcript is found."
            "\nNote 2: You must have an OpenAI API key to summarize, and credited tokens."
            "\n\tDeclare your key in an environment variable OPENAI_API_KEY prior to running the script."
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
    # mandatory argument: youtube video url
    parser.add_argument(
        "input_path",
        help="URL of the YouTube video or path to a local media / text file to summarize",
    )
    # optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Increases logging verbosity")
    parser.add_argument(
        "-l",
        "--language",
        choices={"en", "fr"},
        default="en",
        help="Use en or fr to specify the language of the summary (default: en)"
        "\nNote: this argument is ignored for local media files, as the language will be auto-detected.",
    )
    parser.add_argument(
        "-d",
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
        "--with_openai",
        action="store_true",
        default=False,
        help="Use OpenAI API to summarize the transcript (default: False)",
    )
    args = parser.parse_args()
    args.is_url = is_valid_url(args.input_path)
    if args.is_url:
        my_logger.debug(f"Input path is a valid URL: {args.input_path}")
    args.is_file = Path(args.input_path).is_file()
    if args.is_file:
        my_logger.debug(f"Input path is a valid file: {args.input_path}")
        # Detect if the file is a media file or a text file
        media_extensions = {".mp4", ".mp3", ".wav", ".mkv", ".avi", ".webm", "m4a"}
        text_extensions = {".txt", ".srt", ".vtt"}
        file_extension = Path(args.input_path).suffix.lower()
        if file_extension in media_extensions:
            args.is_media_file = True
            args.is_text_file = False
            my_logger.debug(f"Input file is detected as a media file: {args.input_path}")
        elif file_extension in text_extensions:
            args.is_media_file = False
            args.is_text_file = True
            my_logger.debug(f"Input file is detected as a text file: {args.input_path}")
        else:
            args.is_media_file = False
            args.is_text_file = False
            my_logger.warning(f"Input file type is unknown: {args.input_path}")
    else:
            args.is_media_file = False
            args.is_text_file = False

    # Check if input_path is a valid URL or an existing local file
    if not args.is_file and not args.is_url:
        err_msg = f"Invalid input path: {args.input_path}. Must be a valid URL or an existing local file."
        raise argparse.ArgumentTypeError(err_msg)

    return args
