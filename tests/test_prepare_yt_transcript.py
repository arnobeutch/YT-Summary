# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for prepare_yt_transcript (yt-dlp backed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from yt_dlp.utils import DownloadError

from prepare_yt_transcript import (
    TranscriptUnavailableError,
    _extract_text_from_subtitle_file,
    _pick_caption,
    get_youtube_transcript,
)

# A tiny SRT fixture used by the integration of get_youtube_transcript.
_SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a transcript

3
00:00:05,500 --> 00:00:07,000
This is a transcript
"""


def _ydl_returning(
    info: dict[str, Any],
    write_files: dict[str, str] | None = None,
) -> MagicMock:
    """Build a YoutubeDL mock that mimics extract_info + writes captions to disk.

    ``write_files`` maps relative filename (e.g. ``"abc.fr.srt"``) to its
    contents; on each ``extract_info`` call the mock writes them under the
    output template's parent directory (parsed from ``opts['outtmpl']``).
    """
    files_to_write = write_files or {}

    def _ctor(opts: dict[str, Any]) -> MagicMock:
        ctx = MagicMock()
        outtmpl = str(opts.get("outtmpl", ""))
        outdir = Path(outtmpl).parent if outtmpl else Path()

        def _extract_info(_url: str, *, download: bool = True) -> dict[str, Any]:
            _ = download
            for name, body in files_to_write.items():
                (outdir / name).write_text(body, encoding="utf-8")
            return info

        ctx.extract_info.side_effect = _extract_info
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        return ctx

    return MagicMock(side_effect=_ctor)


class TestPickCaption:
    def test_prefers_manual_fr_over_auto_fr(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"fr": [{}], "en": [{}]},
            "automatic_captions": {"fr": [{}]},
        }
        assert _pick_caption(info) == ("fr", "manual")

    def test_prefers_manual_en_when_fr_absent(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"en": [{}]},
            "automatic_captions": {"fr": [{}]},
        }
        # Manual beats auto across languages — manual en wins over auto fr.
        assert _pick_caption(info) == ("en", "manual")

    def test_falls_back_to_auto_fr_when_no_manual(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {},
            "automatic_captions": {"fr": [{}], "en": [{}]},
        }
        assert _pick_caption(info) == ("fr", "auto")

    def test_falls_back_to_auto_en_when_only_en_auto_present(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {},
            "automatic_captions": {"en": [{}]},
        }
        assert _pick_caption(info) == ("en", "auto")

    def test_falls_back_to_other_language_manual(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"de": [{}]},
            "automatic_captions": {},
        }
        assert _pick_caption(info) == ("de", "manual")

    def test_returns_none_when_no_captions(self) -> None:
        empty: dict[str, Any] = {"subtitles": {}, "automatic_captions": {}}
        assert _pick_caption(empty) is None


class TestExtractTextFromSubtitleFile:
    def test_strips_timestamps_and_indices(self, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        srt.write_text(_SAMPLE_SRT)
        text = _extract_text_from_subtitle_file(srt)
        assert "Hello world" in text
        assert "transcript" in text
        # No timestamp/index leftovers.
        assert "-->" not in text
        assert "00:00" not in text

    def test_deduplicates_consecutive_identical_lines(self, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        srt.write_text(_SAMPLE_SRT)
        text = _extract_text_from_subtitle_file(srt)
        # "This is a transcript" appears twice in source, once after dedup.
        assert text.count("This is a transcript") == 1

    def test_strips_vtt_header(self, tmp_path: Path) -> None:
        vtt = tmp_path / "x.vtt"
        vtt.write_text(
            "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:01.000\nHello\n",
        )
        assert _extract_text_from_subtitle_file(vtt) == "Hello"

    def test_strips_inline_tags(self, tmp_path: Path) -> None:
        vtt = tmp_path / "x.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<c.color1>Hello</c> world\n",
        )
        assert _extract_text_from_subtitle_file(vtt) == "Hello world"


class TestGetYoutubeTranscript:
    def test_happy_path_manual_fr(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": _SAMPLE_SRT})
        with patch("prepare_yt_transcript.yt_dlp.YoutubeDL", ydl):
            result = get_youtube_transcript("abc")
        assert "Hello world" in result
        for line in result.splitlines():
            assert len(line) <= 80

    def test_happy_path_auto_en_when_no_manual(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {},
            "automatic_captions": {"en": [{}]},
        }
        ydl = _ydl_returning(info, write_files={"abc.en.srt": _SAMPLE_SRT})
        with patch("prepare_yt_transcript.yt_dlp.YoutubeDL", ydl):
            result = get_youtube_transcript("abc")
        assert "Hello world" in result

    def test_no_captions_raises_lang_not_found(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info)
        with (
            patch("prepare_yt_transcript.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "lang_not_found"

    def test_download_error_raises_list_failed(self) -> None:
        ctx = MagicMock()
        ctx.extract_info.side_effect = DownloadError("boom")
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        with (
            patch("prepare_yt_transcript.yt_dlp.YoutubeDL", MagicMock(return_value=ctx)),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "list_failed"

    def test_listed_but_no_file_raises_empty_payload(self) -> None:
        # Track is listed but yt-dlp doesn't actually write the file.
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={})
        with (
            patch("prepare_yt_transcript.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "empty_payload"

    def test_empty_caption_file_raises_empty_payload(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": "WEBVTT\n\n"})
        with (
            patch("prepare_yt_transcript.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "empty_payload"
