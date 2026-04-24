"""Pluggable summarization backends behind a common Protocol."""

from __future__ import annotations

from .base import MissingAPIKeyError, Summarizer, analyze_sentiment, make_summarizer
from .modes import SummaryMode, detect_mode, resolve_mode

__all__ = [
    "MissingAPIKeyError",
    "Summarizer",
    "SummaryMode",
    "analyze_sentiment",
    "detect_mode",
    "make_summarizer",
    "resolve_mode",
]
