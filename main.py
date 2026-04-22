"""AI agent to summarize YouTube videos, local media, or existing transcripts."""

import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import my_parser
import prepare_local_transcript as plt
import prepare_yt_transcript as pytt
import summarize_transcript as st
from my_logger import initialize_logger, my_logger
from my_settings import Settings


def _sanitize_filename(name: str) -> str:
    illegal_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    for char in illegal_chars:
        name = name.replace(char, "_")
    return name


def main() -> None:
    """Retrieve transcript and analyze."""
    args = my_parser.parse_args()
    initialize_logger(args)
    settings = Settings.from_env()

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    transcript: str = ""
    video_title: str = ""

    if args.is_url:
        video_id = args.input_path.split("v=")[1]
        my_logger.debug(f"Video ID: {video_id}")
        transcript = pytt.get_youtube_transcript(video_id)
        r = requests.get(args.input_path, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        link = soup.find_all(name="title")[0]
        video_title = _sanitize_filename(link.text)
        if "Error" not in transcript:
            my_logger.info("Transcript retrieved successfully")
            p = Path(f"./results/{video_title} transcript.txt")
            with p.open(mode="w", encoding="utf8") as f:
                f.write(transcript)
        else:
            print(transcript)

    if args.is_media_file:
        if args.language is not None:
            my_logger.warning(
                "Language argument is ignored for local media files, auto-detected instead.",
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
        with Path(args.input_path).open(encoding="utf8") as file:
            transcript = file.read()
        video_title = Path(args.input_path).stem

    my_logger.info(f"Video title: {video_title}")

    if args.summarize:
        my_logger.info("Generating summary...")
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        sys.exit(0)
