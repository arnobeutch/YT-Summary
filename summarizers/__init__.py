"""Pluggable summarization backends behind a common Protocol."""

from __future__ import annotations

from .base import MissingAPIKeyError, Summarizer, analyze_sentiment, make_summarizer

__all__ = [
    "MissingAPIKeyError",
    "Summarizer",
    "analyze_sentiment",
    "make_summarizer",
]
