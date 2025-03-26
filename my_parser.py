"""Command-line parser for this script."""

import argparse


def parse_args() -> argparse.Namespace:
    """Define then parse command-line arguments.

    Returns:
        argparse.Namespace: the arguments discovered as attributes and their values.

    """
    parser = argparse.ArgumentParser(
        description=(
            "A python script to summarize YouTube videos."
            "\nThe script will output a markdown formatted summary of the video."
            "\nNote 1: The video must have either English or French subtitles."
            "\n\tThe script will output an error message if no such transcript is found."
            "\nNote 2: You must have an OpenAI API key to use this script, and credited tokens."
            "\n\tDeclare your key in an environment variable OPENAI_API_KEY prior to running the script."
            "\nUse the -h flag for help."
        ),
        usage=(
            "python main.py <youtube_video_url> [optional_arguments]"
            "\nExample: python main.py https://www.youtube.com/watch?v=VIDEO_ID"
        ),
        epilog="Be careful: summarizing burns OpenAI API tokens!",
    )
    # mandatory argument: youtube video url
    parser.add_argument("youtube_video_url", help="URL of the YouTube video to summarize")
    # optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Increases logging verbosity")
    parser.add_argument(
        "-l",
        "--language",
        choices={"en", "fr"},
        default="en",
        help="Use en or fr to specify the language of the summary (default: en)",
    )
    parser.add_argument(
        "-e",
        "--extended",
        action="store_true",
        default=False,
        help="Add category, keywords and partial transcript to the output",
    )
    parser.add_argument(
        "-t",
        "--transcript",
        action="store_true",
        default=False,
        help="Output the full transcript & sentiment instead of a summary",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        default=False,
        help="Output the summary or transcript to file",
    )

    return parser.parse_args()

