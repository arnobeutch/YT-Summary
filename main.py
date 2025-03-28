"""AI agent to summarize YT videos.

Uses the YT package and the langchain_openai package.
"""

import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import analyze_yt_transcript as aytt
import format_utils as my_fmt
import my_logger
import my_parser
import prepare_yt_transcript as pytt


def main() -> None:
    """Retrieve transcript and analyze."""
    args = my_parser.parse_args()

    # set up logger
    my_logger.initialize_logger(args)

    my_logger.log.info(f"Script called with the following arguments: {vars(args)}")

    load_dotenv()  # declare your OPENAI_API_KEY in a .env file
    video_id = args.youtube_video_url.split("v=")[1]
    my_logger.log.debug(f"Video ID: {video_id}")
    transcript = pytt.get_youtube_transcript(video_id)
    # get video title
    # TODO: add error handling
    r = requests.get(args.youtube_video_url, timeout=10)
    soup = BeautifulSoup(r.content, "html.parser")
    link = soup.find_all(name="title")[0]
    video_title = link.text
    my_logger.log.info(f"Video title: {video_title}")

    if "Error" not in transcript:
        my_logger.log.info("Transcript retrieved successfully")
        sentiment = pytt.analyze_sentiment(transcript)
        if args.transcript:
            if args.file:
                p = Path("./results/transcript.txt")
                with p.open(mode="w",encoding="utf8") as f:
                    my_text = f"Transcript:\n{transcript}\n\nSentiment: {sentiment}"
                    f.write(my_text)
            else:
                print(f"Transcript:\n{transcript}\n\n")
                print(f"Sentiment: {sentiment}")
            sys.exit(0)
        summary = aytt.summarize_transcript(transcript, args.language)
        if summary is None:
            my_logger.log.error("Error: Summary could not be generated.")
            sys.exit(1)
        else:
            my_logger.log.info("Summary generated successfully")

            markdown_output = my_fmt.format_markdown(
                video_title,
                args.youtube_video_url,
                summary,
                sentiment,
                args.language,
            )
        if args.file:
            p = Path("./results/summary.md")
            with p.open(mode="w", encoding="utf8") as f:
                f.write(markdown_output)
        else:
            print(markdown_output)
    else:
        print(transcript)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.log.critical("Interrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
