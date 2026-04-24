"""AI agent to summarize YouTube videos, local media, or existing transcripts."""

import sys
import textwrap
from pathlib import Path

import my_parser
import prepare_local_transcript as plt
import prepare_yt_audio as pya
import prepare_yt_transcript as pytt
import summarize_transcript as st
from my_logger import initialize_logger, my_logger
from my_settings import Settings
from prepare_yt_transcript import TranscriptUnavailableError

_WRAP_WIDTH = 80


def _sanitize_filename(name: str) -> str:
    illegal_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    for char in illegal_chars:
        name = name.replace(char, "_")
    return name


def _wrap_transcript(text: str, *, diarize: bool) -> str:
    """Soft-wrap a transcript at 80 chars without breaking words.

    Skipped for diarized output — `SPEAKER: text` lines must stay intact.
    """
    if diarize:
        return text
    return textwrap.fill(
        text,
        width=_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )


def main() -> None:
    """Retrieve transcript and analyze."""
    args = my_parser.parse_args()
    initialize_logger(args)
    settings = Settings.from_env()  # seeds os.environ from .env

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    transcript: str = ""
    video_title: str = ""

    if args.is_url:
        video_id = pya.extract_video_id(args.input_path)
        my_logger.debug(f"Video ID: {video_id}")
        try:
            transcript = pytt.get_youtube_transcript(video_id)
        except TranscriptUnavailableError as exc:
            my_logger.info(
                f"No YouTube transcript available ({exc.reason}: {exc}) — "
                f"falling back to local transcription.",
            )
            audio_path, raw_title = pya.download_youtube_audio(
                args.input_path,
                Path("./downloads"),
            )
            video_title = _sanitize_filename(raw_title)
            if args.diarize:
                transcript, args.language = plt.transcribe_audio_with_diarization(
                    str(audio_path),
                    model_size="small",
                )
                p = Path(f"./results/{video_title} diarized transcript.txt")
            else:
                transcript, args.language = plt.transcribe_audio(
                    str(audio_path),
                    model_size="small",
                )
                p = Path(f"./results/{video_title} transcript.txt")
        else:
            my_logger.info("Transcript retrieved successfully")
            video_title = _sanitize_filename(pya.fetch_video_title(args.input_path))
            p = Path(f"./results/{video_title} transcript.txt")

        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode="w", encoding="utf8") as f:
            f.write(_wrap_transcript(transcript, diarize=args.diarize))

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
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode="w", encoding="utf8") as f:
            f.write(_wrap_transcript(transcript, diarize=args.diarize))

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
