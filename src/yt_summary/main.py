"""AI agent to summarize YouTube videos, local media, or existing transcripts."""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys

import torch.cuda

from yt_summary import handlers, parser
from yt_summary.logger import initialize_logger, my_logger
from yt_summary.settings import Settings
from yt_summary.summarizers import MissingAPIKeyError, make_summarizer


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


def _gpu_warning() -> None:
    """Warn when nvidia-smi exists but CUDA is unavailable (driver/runtime mismatch)."""
    if shutil.which("nvidia-smi") and not torch.cuda.is_available():
        my_logger.warning(
            "nvidia-smi found but torch.cuda.is_available() is False — "
            "whisper will run on CPU. Check your CUDA driver/runtime installation.",
        )


def _dry_run_report(path: str, classification: dict[str, bool], settings: Settings) -> None:
    """Print a one-line dry-run summary for a single input."""
    if classification["is_url"]:
        kind = "youtube-url"
    elif classification["is_media_file"]:
        kind = "local-media"
    elif classification["is_text_file"]:
        kind = "local-text"
    else:
        kind = "local-file (unknown type)"
    my_logger.info(
        f"[dry-run] {path!r} → {kind} | model: {settings.whisper_model_size} | "
        f"output: {settings.output_dir}/",
    )


def main() -> None:
    """Parse args, build a Transcript for each input, write it, and optionally summarize."""
    args = parser.parse_args()
    initialize_logger(args)
    settings = _apply_cli_overrides(args, Settings.from_env())

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    _gpu_warning()

    will_summarize = args.summarize and not args.transcript_only

    # Preflight the LLM backend BEFORE the slow transcription pipeline so a
    # missing API key fails in seconds, not after a 10-minute whisper run.
    if will_summarize and not args.dry_run:
        try:
            make_summarizer(settings)
        except MissingAPIKeyError as exc:
            my_logger.error(str(exc))
            sys.exit(2)

    for path in args.input_path:
        classification = parser.classify_input(path)

        if args.dry_run:
            _dry_run_report(path, classification, settings)
            continue

        # Build a per-path namespace so handlers receive a single `input_path`
        # string along with the correct type flags.
        per_args = argparse.Namespace(**vars(args))
        per_args.input_path = path
        for key, val in classification.items():
            setattr(per_args, key, val)

        if per_args.is_url:
            transcript = handlers.handle_url(per_args, settings)
        elif per_args.is_media_file:
            transcript = handlers.handle_media(per_args, settings)
        elif per_args.is_text_file:
            transcript = handlers.handle_text(per_args, settings)
        else:
            my_logger.error(f"No handler for the given input type: {path}")
            continue

        handlers.write_transcript_file(transcript, settings, subtitles=args.subtitles)
        my_logger.info(f"Video title: {transcript.title}")

        if will_summarize:
            my_logger.info("Generating summary...")
            handlers.summarize(transcript, per_args, settings)
        elif args.transcript_only and args.summarize:
            my_logger.info("--transcript-only set; skipping summary generation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        sys.exit(0)
