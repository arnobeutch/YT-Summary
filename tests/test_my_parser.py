"""Tests for my_parser."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from my_parser import is_valid_url, parse_args

if TYPE_CHECKING:
    from pathlib import Path


class TestIsValidUrl:
    def test_http_url(self) -> None:
        assert is_valid_url("http://example.com")

    def test_https_url(self) -> None:
        assert is_valid_url("https://www.youtube.com/watch?v=abc123")

    def test_url_with_path(self) -> None:
        assert is_valid_url("https://example.com/path/to/thing")

    def test_empty_string(self) -> None:
        assert not is_valid_url("")

    def test_plain_string(self) -> None:
        assert not is_valid_url("just-a-string")

    def test_scheme_without_netloc(self) -> None:
        assert not is_valid_url("http://")

    def test_relative_path(self) -> None:
        assert not is_valid_url("./my_video.mp4")


def _run_parser(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    return parse_args()


class TestParseArgs:
    def test_url_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://youtube.com/watch?v=abc"], monkeypatch)
        assert ns.is_url
        assert not ns.is_file
        assert not ns.is_media_file
        assert not ns.is_text_file

    def test_media_file_input_uppercase_ext(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        media = tmp_path / "clip.MP4"  # uppercase extension — must still match
        media.write_text("")
        ns = _run_parser([str(media)], monkeypatch)
        assert ns.is_file
        assert ns.is_media_file
        assert not ns.is_text_file

    @pytest.mark.parametrize("ext", [".mp4", ".mp3", ".wav", ".mkv", ".avi", ".webm", ".m4a"])
    def test_all_media_extensions(
        self,
        ext: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / f"clip{ext}"
        f.write_text("")
        ns = _run_parser([str(f)], monkeypatch)
        assert ns.is_media_file

    @pytest.mark.parametrize("ext", [".txt", ".srt", ".vtt"])
    def test_all_text_extensions(
        self,
        ext: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / f"tr{ext}"
        f.write_text("")
        ns = _run_parser([str(f)], monkeypatch)
        assert ns.is_text_file

    def test_unknown_extension_is_neither(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = tmp_path / "thing.pdf"
        p.write_text("")
        ns = _run_parser([str(p)], monkeypatch)
        assert ns.is_file
        assert not ns.is_media_file
        assert not ns.is_text_file

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x"], monkeypatch)
        assert ns.language is None  # autodetect
        assert ns.diarize is False
        assert ns.summarize is False
        assert ns.with_openai is False
        assert ns.debug is False

    def test_language_fr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--language", "fr"], monkeypatch)
        assert ns.language == "fr"

    def test_language_short_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "-l", "fr"], monkeypatch)
        assert ns.language == "fr"

    def test_debug_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "-d"], monkeypatch)
        assert ns.debug is True

    def test_all_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--diarize", "--summarize", "--with-openai"],
            monkeypatch,
        )
        assert ns.diarize is True
        assert ns.summarize is True
        assert ns.with_openai is True

    def test_with_openai_canonical_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--with-openai"], monkeypatch)
        assert ns.with_openai is True

    def test_with_openai_legacy_underscore_still_works(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--with_openai"], monkeypatch)
        assert ns.with_openai is True

    def test_backend_flags_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Backend/config flags default to None so the resolver at use-site can
        # fall back to Settings (and thus env vars).
        ns = _run_parser(["https://y.com/watch?v=x"], monkeypatch)
        assert ns.model_size is None
        assert ns.llm_provider is None
        assert ns.llm_model is None
        assert ns.output_dir is None
        assert ns.downloads_dir is None
        assert ns.summary_mode is None

    def test_summary_mode_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--summary-mode", "meeting"],
            monkeypatch,
        )
        assert ns.summary_mode == "meeting"

    def test_summary_mode_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(["https://y.com/watch?v=x", "--summary-mode", "novel"], monkeypatch)

    def test_force_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x"], monkeypatch)
        assert ns.force is False

    def test_force_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--force"], monkeypatch)
        assert ns.force is True

    def test_subtitles_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x"], monkeypatch)
        assert ns.subtitles is False

    def test_subtitles_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--subtitles"], monkeypatch)
        assert ns.subtitles is True

    def test_transcript_only_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x"], monkeypatch)
        assert ns.transcript_only is False

    def test_transcript_only_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["https://y.com/watch?v=x", "--transcript-only"], monkeypatch)
        assert ns.transcript_only is True

    def test_model_size_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--model-size", "medium"],
            monkeypatch,
        )
        assert ns.model_size == "medium"

    def test_model_size_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(["https://y.com/watch?v=x", "--model-size", "huge"], monkeypatch)

    def test_llm_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--llm-provider", "openrouter"],
            monkeypatch,
        )
        assert ns.llm_provider == "openrouter"

    def test_llm_provider_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(
                ["https://y.com/watch?v=x", "--llm-provider", "together"],
                monkeypatch,
            )

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--llm-model", "anthropic/claude-4.7-sonnet"],
            monkeypatch,
        )
        assert ns.llm_model == "anthropic/claude-4.7-sonnet"

    def test_output_dir_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "out"
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--output-dir", str(target)],
            monkeypatch,
        )
        assert ns.output_dir == target

    def test_downloads_dir_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "dl"
        ns = _run_parser(
            ["https://y.com/watch?v=x", "--downloads-dir", str(target)],
            monkeypatch,
        )
        assert ns.downloads_dir == target

    def test_invalid_input_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid input path"):
            _run_parser(["not-a-url-nor-file"], monkeypatch)

    def test_unsupported_language_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # argparse rejects invalid choice with SystemExit(2)
        with pytest.raises(SystemExit):
            _run_parser(["https://y.com/watch?v=x", "-l", "de"], monkeypatch)
