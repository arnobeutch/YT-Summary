"""OpenRouter summarizer — OpenAI-compatible at a different ``base_url``.

Lets the user pick provider-prefixed model names like
``minimax/minimax-2.7``, ``moonshotai/kimi-k2``, or
``anthropic/claude-4.7-sonnet`` via ``--llm-model`` (or ``LLM_MODEL`` env).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .openai_compatible import OpenAICompatibleSummarizer

if TYPE_CHECKING:
    from yt_summary.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSummarizer(OpenAICompatibleSummarizer):
    """OpenRouter-backed Chat Completions."""

    def __init__(self, settings: Settings) -> None:
        """Capture OpenRouter API key + base URL from settings."""
        super().__init__(settings)
        self.api_key = settings.openrouter_api_key
        self.base_url = OPENROUTER_BASE_URL

    def _model_name(self) -> str:
        # OpenRouter requires a provider-prefixed model id; if neither
        # --llm-model nor LLM_MODEL is set, fall back to a sensible default.
        return self.settings.llm_model or "openai/gpt-4o-mini"
