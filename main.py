"""AI agent to summarize YT videos.

Uses the YT package and the langchain_openai package.
"""

import logging
import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import my_parser
import prepare_local_transcript as plt
import prepare_yt_transcript as pytt
import summarize_transcript as st
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

    if (
        args.is_url
    ):  # TODO: move video_id extraction and output file creation to get_yt_transcript
        video_id = args.input_path.split("v=")[1]
        my_logger.debug(f"Video ID: {video_id}")
        transcript = pytt.get_youtube_transcript(video_id)
        # get video title
        r = requests.get(args.input_path, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        link = soup.find_all(name="title")[0]
        video_title = link.text
        # Remove illegal characters from video_title for file name
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            video_title = video_title.replace(char, '_')
        if (
            "Error" not in transcript
        ):  # TODO: refactor this (don't use transcript as a flag)
            my_logger.info("Transcript retrieved successfully")
            p = Path(f"./results/{video_title} transcript.txt")
            with p.open(mode="w", encoding="utf8") as f:
                f.write(transcript)
        else:
            print(transcript)

    if args.is_media_file:  # TODO: move output file creation to get_local_transcript
        # language will be updated to the one used in the video
        # It's possible to set the transcription model size as argument: model_size="small" for example.
        # Default is "base"
        if args.language is not None:
            my_logger.warning(
                "Language argument is ignored for local media files, it will be auto-detected.",
            )
        video_title = Path(args.input_path).stem
        if args.diarize:
            transcript, args.language = plt.transcribe_video_file_with_diarization(
                args.input_path,
                model_size="small",
            )
            p = Path(f"./results/{video_title} diarized transcript.txt")
        else:
            transcript, args.language = plt.transcribe_video_file(
                args.input_path,
                model_size="small",
            )
            p = Path(f"./results/{video_title} transcript.txt")
        with p.open(mode="w", encoding="utf8") as f:
            f.write(transcript)

    if args.is_text_file:
        with Path.open(args.input_path, encoding="utf8") as file:
            transcript = file.read()
        video_title = Path(args.input_path).stem

    my_logger.info(f"Video title: {video_title}")

    if args.summarize:
        my_logger.info("Generating summary...")
        # TODO: when analyzing from existing transcript the language is not detected
        # TODO: with or w/o OpenAI should use the same prompt
        # TODO: implement a meeting summary mode and an informational summary mode
        if args.with_openai:
            st.summarize_transcript_with_openai(
                transcript,
                args.input_path,
                video_title,
                args.language,
            )
        else:
            st.summarize_transcript_with_rag(
                transcript,
                args.input_path,
                video_title,
                args.language,
                model="mistral",
            )
        # if summary is None:
        #     my_logger.error("Error: Summary could not be generated.")
        #     sys.exit(1)
        # else:
        #     my_logger.info("Summary generated successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
