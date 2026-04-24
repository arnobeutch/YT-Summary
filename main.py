"""AI agent to summarize YouTube videos, local media, or existing transcripts."""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING

import handlers
import my_parser
from my_logger import initialize_logger, my_logger
from my_settings import Settings

if TYPE_CHECKING:
    import argparse


def _apply_cli_overrides(args: argparse.Namespace, base: Settings) -> Settings:
    """Return a new ``Settings`` with CLI-provided values overlaid on ``base``."""
    return dataclasses.replace(
        base,
        output_dir=args.output_dir or base.output_dir,
        downloads_dir=args.downloads_dir or base.downloads_dir,
        whisper_model_size=args.model_size or base.whisper_model_size,
        llm_provider=args.llm_provider or base.llm_provider,
        llm_model=args.llm_model or base.llm_model,
    )


def main() -> None:
    """Parse args, build a Transcript, write it, and optionally summarize it."""
    args = my_parser.parse_args()
    initialize_logger(args)
    settings = _apply_cli_overrides(args, Settings.from_env())

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    if args.is_url:
        transcript = handlers.handle_url(args, settings)
    elif args.is_media_file:
        transcript = handlers.handle_media(args, settings)
    elif args.is_text_file:
        transcript = handlers.handle_text(args, settings)
    else:  # parse_args already guards against this; defensive only
        my_logger.error("No handler for the given input type")
        return

    handlers.write_transcript_file(transcript, settings)
    my_logger.info(f"Video title: {transcript.title}")

    if args.summarize:
        my_logger.info("Generating summary...")
        handlers.summarize(transcript, args, settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        sys.exit(0)
