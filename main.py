"""AI agent to summarize YouTube videos, local media, or existing transcripts."""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING

import handlers
import my_parser
from my_logger import initialize_logger, my_logger
from my_settings import Settings
from summarizers import MissingAPIKeyError, make_summarizer

if TYPE_CHECKING:
    import argparse


def _apply_cli_overrides(args: argparse.Namespace, base: Settings) -> Settings:
    """Return a new ``Settings`` with CLI-provided values overlaid on ``base``.

    ``--with-openai`` is treated as a shortcut for ``--llm-provider openai``.
    """
    provider = args.llm_provider or base.llm_provider
    if args.with_openai:
        provider = "openai"
    return dataclasses.replace(
        base,
        output_dir=args.output_dir or base.output_dir,
        downloads_dir=args.downloads_dir or base.downloads_dir,
        whisper_model_size=args.model_size or base.whisper_model_size,
        llm_provider=provider,
        llm_model=args.llm_model or base.llm_model,
        summary_mode=args.summary_mode or base.summary_mode,
    )


def main() -> None:
    """Parse args, build a Transcript, write it, and optionally summarize it."""
    args = my_parser.parse_args()
    initialize_logger(args)
    settings = _apply_cli_overrides(args, Settings.from_env())

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    will_summarize = args.summarize and not args.transcript_only

    # Preflight the LLM backend BEFORE the slow transcription pipeline so a
    # missing API key fails in seconds, not after a 10-minute whisper run.
    if will_summarize:
        try:
            make_summarizer(settings)
        except MissingAPIKeyError as exc:
            my_logger.error(str(exc))
            sys.exit(2)

    if args.is_url:
        transcript = handlers.handle_url(args, settings)
    elif args.is_media_file:
        transcript = handlers.handle_media(args, settings)
    elif args.is_text_file:
        transcript = handlers.handle_text(args, settings)
    else:  # parse_args already guards against this; defensive only
        my_logger.error("No handler for the given input type")
        return

    handlers.write_transcript_file(transcript, settings, subtitles=args.subtitles)
    my_logger.info(f"Video title: {transcript.title}")

    if will_summarize:
        my_logger.info("Generating summary...")
        handlers.summarize(transcript, args, settings)
    elif args.transcript_only and args.summarize:
        my_logger.info("--transcript-only set; skipping summary generation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        sys.exit(0)
