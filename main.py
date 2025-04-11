"""AI agent to summarize YT videos.

Uses the YT package and the langchain_openai package.
"""

import logging
import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import analyze_yt_transcript as aytt
import format_utils as my_fmt
import my_parser
import prepare_local_transcript as plt
import prepare_yt_transcript as pytt
from my_logger import initialize_logger, my_logger


def main() -> None:
    """Retrieve transcript and analyze."""
    # set up logger
    initialize_logger()

    # get command-line arguments
    args = my_parser.parse_args()
    if args.verbose:
        my_logger.setLevel(logging.DEBUG)

    my_logger.info(f"Script called with the following arguments: {vars(args)}")

    if args.is_url:
        video_id = args.input_path.split("v=")[1]
        my_logger.debug(f"Video ID: {video_id}")
        transcript = pytt.get_youtube_transcript(video_id)
        # get video title
        r = requests.get(args.input_path, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        link = soup.find_all(name="title")[0]
        video_title = link.text

    if args.is_media_file:
        # language will be updated to the one used in the video
        # It's possible to set the transcription model size as argument: model_size="small" for example.
        # Default is "base"
        if args.language is not None:
            my_logger.warning("Language argument is ignored for local media files, it will be auto-detected.")
        transcript, args.language = plt.transcribe_video_file(args.input_path)
        video_title = Path(args.input_path).stem

    if args.is_text_file:
        with Path.open(args.input_path, encoding="utf8") as file:
            transcript = file.read()
        video_title = Path(args.input_path).stem

    my_logger.info(f"Video title: {video_title}")

    if "Error" not in transcript:
        my_logger.info("Transcript retrieved successfully")
        sentiment = pytt.analyze_sentiment(transcript)
        p = Path("./results/transcript.txt")
        with p.open(mode="w",encoding="utf8") as f:
            my_text = f"Transcript:\n{transcript}\n\nSentiment: {sentiment}"
            f.write(my_text)
        if args.summarize:
            my_logger.info("Generating summary...")
            summary = aytt.summarize_transcript(transcript, args.language)
            if summary is None:
                my_logger.error("Error: Summary could not be generated.")
                sys.exit(1)
            else:
                my_logger.info("Summary generated successfully")

                markdown_output = my_fmt.format_markdown(
                    video_title,
                    args.input_path,
                    summary,
                    sentiment,
                    args.language,
                )
                p = Path("./results/summary.md")
                with p.open(mode="w", encoding="utf8") as f:
                    f.write(markdown_output)
    else:
        print(transcript)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
