"""Tests for summarize_transcript.

``analyze_sentiment`` runs TextBlob for real (pure, no network). OpenAI and
RAG entry points are tested with the external call mocked out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import openai
import pytest

from summarize_transcript import (
    analyze_sentiment,
    summarize_transcript_with_openai,
    summarize_transcript_with_rag,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestAnalyzeSentiment:
    def test_positive_text(self) -> None:
        text = "This is wonderful, amazing, and excellent! The best day ever."
        assert analyze_sentiment(text) == "Positive"

    def test_negative_text(self) -> None:
        text = "This is terrible, horrible, awful. Truly the worst experience."
        assert analyze_sentiment(text) == "Negative"

    def test_neutral_text(self) -> None:
        assert analyze_sentiment("The cat sat on the mat.") == "Neutral"

    def test_empty_text_is_neutral(self) -> None:
        assert analyze_sentiment("") == "Neutral"


class TestSummarizeTranscriptWithOpenAI:
    def test_unsupported_language_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="language not supported"):
            summarize_transcript_with_openai("transcript", "path", "video", "de")

    def test_english_happy_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        with patch("summarize_transcript.OpenAI") as mock_client:
            response = MagicMock()
            response.choices[0].message.content = "summary body"
            mock_client.return_value.chat.completions.create.return_value = response
            summarize_transcript_with_openai("transcript text", "http://v", "mytitle", "en")
        out = tmp_path / "results" / "mytitle.md"
        assert out.exists()
        content = out.read_text()
        assert "summary body" in content
        assert "mytitle" in content

    def test_french_uses_french_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        with patch("summarize_transcript.OpenAI") as mock_client:
            response = MagicMock()
            response.choices[0].message.content = "résumé"
            mock_client.return_value.chat.completions.create.return_value = response
            summarize_transcript_with_openai("t", "p", "title_fr", "fr")
        _, kwargs = mock_client.return_value.chat.completions.create.call_args
        prompt = kwargs["messages"][1]["content"]
        assert "expert en résumé" in prompt

    def test_openai_error_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = openai.OpenAIError(
                "boom",
            )
            # must not raise
            result = summarize_transcript_with_openai("t", "p", "title", "en")
        assert result is None
        assert not (tmp_path / "results").exists() or not any((tmp_path / "results").iterdir())

    def test_missing_api_key_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # OpenAI() raises OpenAIError at construction when OPENAI_API_KEY is unset.
        # The handler must catch this too — not just failures from .create().
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.OpenAI", side_effect=openai.OpenAIError("no key")):
            result = summarize_transcript_with_openai("t", "p", "title", "en")
        assert result is None

    def test_empty_content_skips_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.OpenAI") as mock_client:
            response = MagicMock()
            response.choices[0].message.content = None
            mock_client.return_value.chat.completions.create.return_value = response
            with caplog.at_level("ERROR"):
                summarize_transcript_with_openai("t", "p", "title", "en")
        assert any("empty content" in r.message for r in caplog.records)
        assert not (tmp_path / "results" / "title.md").exists()


class TestSummarizeTranscriptWithRAG:
    def test_happy_path_fr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.generate_summary") as gen:
            gen.return_value = "Sujet: test\nHashtags: #t\n"
            path = summarize_transcript_with_rag("Alice: hi", "vid1", "fr", model="mistral")
        assert path.exists()
        assert path.name == "vid1 - résumé.md"
        assert "# Résumé de la réunion — vid1" in path.read_text()

    def test_english_suffix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.generate_summary") as gen:
            gen.return_value = "Sujet: test\n"
            path = summarize_transcript_with_rag("Alice: hi", "vid2", "en")
        assert path.name == "vid2 - summary.md"

    def test_creates_results_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.generate_summary") as gen:
            gen.return_value = "Sujet: x\n"
            summarize_transcript_with_rag("A: x", "v", "en")
        assert (tmp_path / "results").is_dir()

    def test_passes_through_parsed_utterances(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("summarize_transcript.generate_summary") as gen:
            gen.return_value = "Sujet: ok\n"
            summarize_transcript_with_rag(
                "Alice: hello\nBob: hi",
                "v",
                "en",
                model="llama",
            )
        args, kwargs = gen.call_args
        utterances = args[0]
        assert ("Alice", "hello") in utterances
        assert ("Bob", "hi") in utterances
        assert kwargs["model"] == "llama"

    def test_generate_summary_error_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("summarize_transcript.generate_summary", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            summarize_transcript_with_rag("x: y", "v", "en")
