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
        # TODO: add error handling
        r = requests.get(args.input_path, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        link = soup.find_all(name="title")[0]
        video_title = link.text
        my_logger.info(f"Video title: {video_title}")

    if args.is_file:
        my_logger.error("Error: Local file not supported yet.")
        sys.exit(1)

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
