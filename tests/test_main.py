"""Tests for main.main — orchestration only.

Per-handler logic lives in test_handlers.py; per-helper logic in
test_formatting.py. Here we just check that the dispatcher routes the
right `args.is_*` flag to the right handler and honors `--summarize`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from main import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_args(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "input_path": "",
        "language": "en",
        "diarize": False,
        "summarize": False,
        "with_openai": False,
        "debug": False,
        "is_url": False,
        "is_file": False,
        "is_media_file": False,
        "is_text_file": False,
        "model_size": None,
        "llm_provider": None,
        "llm_model": None,
        "output_dir": None,
        "downloads_dir": None,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_transcript(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "text": "body",
        "language": "en",
        "title": "T",
        "source": "yt_caption",
        "diarized": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestDispatcher:
    def test_url_branch_routes_to_handle_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.handlers.handle_url", return_value=_make_transcript()) as h_url,
            patch("main.handlers.handle_media") as h_media,
            patch("main.handlers.handle_text") as h_text,
            patch("main.handlers.write_transcript_file") as write,
            patch("main.handlers.summarize") as summ,
        ):
            parse.return_value = _make_args(input_path="https://y.com/watch?v=x", is_url=True)
            main()
        h_url.assert_called_once()
        h_media.assert_not_called()
        h_text.assert_not_called()
        write.assert_called_once()
        summ.assert_not_called()

    def test_media_branch_routes_to_handle_media(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.handlers.handle_url") as h_url,
            patch("main.handlers.handle_media", return_value=_make_transcript()) as h_media,
            patch("main.handlers.handle_text") as h_text,
            patch("main.handlers.write_transcript_file"),
            patch("main.handlers.summarize"),
        ):
            parse.return_value = _make_args(
                input_path="x.mp4",
                is_file=True,
                is_media_file=True,
            )
            main()
        h_media.assert_called_once()
        h_url.assert_not_called()
        h_text.assert_not_called()

    def test_text_branch_routes_to_handle_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.handlers.handle_url") as h_url,
            patch("main.handlers.handle_media") as h_media,
            patch("main.handlers.handle_text", return_value=_make_transcript()) as h_text,
            patch("main.handlers.write_transcript_file"),
            patch("main.handlers.summarize"),
        ):
            parse.return_value = _make_args(
                input_path="x.txt",
                is_file=True,
                is_text_file=True,
            )
            main()
        h_text.assert_called_once()
        h_url.assert_not_called()
        h_media.assert_not_called()

    def test_summarize_called_when_flag_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.handlers.handle_url", return_value=_make_transcript()),
            patch("main.handlers.write_transcript_file"),
            patch("main.handlers.summarize") as summ,
        ):
            parse.return_value = _make_args(
                input_path="https://y.com/watch?v=x",
                is_url=True,
                summarize=True,
            )
            main()
        summ.assert_called_once()
