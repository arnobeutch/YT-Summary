"""OpenAI summarizer (the default backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .openai_compatible import OpenAICompatibleSummarizer

if TYPE_CHECKING:
    from scriber.settings import Settings


class OpenAISummarizer(OpenAICompatibleSummarizer):
    """Default OpenAI Chat Completions backend."""

    def __init__(self, settings: Settings) -> None:
        """Capture API key from settings; ``base_url`` stays at OpenAI default."""
        super().__init__(settings)
        self.api_key = settings.openai_api_key
