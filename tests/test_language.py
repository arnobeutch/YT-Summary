"""Tests for language-selection helpers."""

from __future__ import annotations

import pytest

from yt_summary.language import derive_summary_language, derive_whisper_summary_language


class TestDeriveSummaryLanguage:
    @pytest.mark.parametrize(
        ("track_lang", "requested", "expected"),
        [
            # Caption is the requested language → summary in that language.
            ("fr", "fr", "fr"),
            ("en", "en", "en"),
            # Caption is English (and not requested) → summary in English.
            ("en", "fr", "en"),
            ("en", None, "en"),
            # Caption is something else → summary forced to English.
            ("de", "fr", "en"),
            ("de", "en", "en"),
            ("de", None, "en"),
            ("it", "fr", "en"),
        ],
    )
    def test_table(self, track_lang: str, requested: str | None, expected: str) -> None:
        assert derive_summary_language(track_lang, requested) == expected


class TestDeriveWhisperSummaryLanguage:
    @pytest.mark.parametrize(
        ("detected", "requested", "expected"),
        [
            # User specified a language → that wins (whisper was forced to it).
            ("en", "fr", "fr"),
            ("de", "fr", "fr"),
            ("xx", "en", "en"),
            # Unspecified + detected ∈ {en, fr} → that.
            ("fr", None, "fr"),
            ("en", None, "en"),
            # Unspecified + detected outside {en, fr} → English.
            ("de", None, "en"),
            ("ja", None, "en"),
        ],
    )
    def test_table(self, detected: str, requested: str | None, expected: str) -> None:
        assert derive_whisper_summary_language(detected, requested) == expected
