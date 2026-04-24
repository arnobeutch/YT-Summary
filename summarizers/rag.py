"""Local RAG summarizer (langchain + Ollama + ChromaDB)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from markdown_writer import format_summary_markdown
from my_logger import my_logger
from preprocess_transcript import parse_transcript, try_resolve_speaker_names
from q_and_a_engine import generate_summary

from .base import analyze_sentiment

if TYPE_CHECKING:
    from model import Transcript
    from my_settings import Settings


class RagSummarizer:
    """Local RAG-based summarizer.

    Always uses Ollama; the model id comes from ``settings.llm_model`` if
    set, otherwise ``settings.ollama_model``. Writes a structured-section
    markdown file (Sujet / Hashtags / ...) plus a sentiment line.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the Settings instance for later access during ``summarize``."""
        self.settings = settings

    def _model_name(self) -> str:
        return self.settings.llm_model or self.settings.ollama_model

    def summarize(self, transcript: Transcript, *, input_path: str) -> Path:
        """Generate a markdown summary on disk for the given transcript."""
        _ = input_path  # not used by RAG (matches Summarizer Protocol)
        my_logger.info("Parsing transcript...")
        utterances = parse_transcript(transcript.text)
        utterances = try_resolve_speaker_names(utterances)

        my_logger.info("Generating summary via RAG...")
        try:
            raw_summary = generate_summary(
                utterances,
                language=transcript.language,
                model=self._model_name(),
            )
        except Exception:
            my_logger.exception("Error generating summary")
            raise

        sentiment = analyze_sentiment(transcript.text)

        my_logger.info("Formatting markdown...")
        formatted = format_summary_markdown(
            raw_summary,
            filename_stem=transcript.title,
            language=transcript.language,
        )
        # Append the sentiment as a last section so RAG output matches OpenAI's
        # "sentiment-everywhere" expectation.
        formatted += f"\n\n## Sentiment\n{sentiment}\n"

        suffix = "résumé" if transcript.language == "fr" else "summary"
        out_path = self.settings.output_dir / f"{transcript.title} - {suffix}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted, encoding="utf-8")
        my_logger.info(f"Summary written to {out_path}")
        return out_path
