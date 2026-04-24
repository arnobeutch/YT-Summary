"""Tests for main.main — each branch of the URL / media / text dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from main import _sanitize_filename, _wrap_transcript, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestSanitizeFilename:
    def test_replaces_illegal_chars(self) -> None:
        assert _sanitize_filename("a<b>c:d") == "a_b_c_d"

    def test_preserves_normal(self) -> None:
        assert _sanitize_filename("hello world") == "hello world"

    def test_all_illegal_chars(self) -> None:
        assert _sanitize_filename('<>:"/\\|?*') == "_________"

    def test_empty(self) -> None:
        assert _sanitize_filename("") == ""


class TestWrapTranscript:
    def test_wraps_long_single_line(self) -> None:
        text = "word " * 50  # ~250 chars, no existing line breaks
        out = _wrap_transcript(text, diarize=False)
        for line in out.splitlines():
            assert len(line) <= 80

    def test_does_not_break_words(self) -> None:
        # A single word longer than 80 chars must not be split.
        long_word = "supercalifragilistic" * 5
        out = _wrap_transcript(long_word, diarize=False)
        assert long_word in out

    def test_diarized_pass_through(self) -> None:
        text = "SPEAKER_00: hello there, this is a long speaker line\nSPEAKER_01: hi"
        assert _wrap_transcript(text, diarize=True) == text

    def test_short_text_unchanged(self) -> None:
        assert _wrap_transcript("short", diarize=False) == "short"


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
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestMainUrlBranch:
    def test_writes_transcript_and_calls_openai_when_flagged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.pya.fetch_video_title", return_value="My Video"),
            patch("main.pytt.get_youtube_transcript", return_value="transcript text") as yt,
            patch("main.st.summarize_transcript_with_openai") as openai_summ,
        ):
            parse.return_value = _make_args(
                input_path="https://www.youtube.com/watch?v=abc123",
                is_url=True,
                summarize=True,
                with_openai=True,
            )
            main()
        yt.assert_called_once_with("abc123")
        assert (tmp_path / "results" / "My Video transcript.txt").read_text() == "transcript text"
        openai_summ.assert_called_once()

    def test_short_link_url_extracts_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.pya.fetch_video_title", return_value="Short"),
            patch("main.pytt.get_youtube_transcript", return_value="body") as yt,
        ):
            parse.return_value = _make_args(
                input_path="https://youtu.be/f8cfH5XX-XU",
                is_url=True,
            )
            main()
        yt.assert_called_once_with("f8cfH5XX-XU")

    def test_no_transcript_falls_back_to_local_transcription(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from prepare_yt_transcript import TranscriptUnavailableError

        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch(
                "main.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "no caps"),
            ),
            patch(
                "main.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Remote Video"),
            ) as dl,
            patch(
                "main.plt.transcribe_audio",
                return_value=("transcribed body", "fr"),
            ) as tr,
        ):
            parse.return_value = _make_args(
                input_path="https://youtu.be/xyz",
                is_url=True,
            )
            main()
        dl.assert_called_once()
        tr.assert_called_once()
        out = tmp_path / "results" / "Remote Video transcript.txt"
        assert out.read_text() == "transcribed body"

    def test_fallback_with_diarization(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from prepare_yt_transcript import TranscriptUnavailableError

        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch(
                "main.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("empty_payload", "empty"),
            ),
            patch(
                "main.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Diarized"),
            ),
            patch(
                "main.plt.transcribe_audio_with_diarization",
                return_value=("Alice: hi", "en"),
            ) as tr,
        ):
            parse.return_value = _make_args(
                input_path="https://youtu.be/abc",
                is_url=True,
                diarize=True,
            )
            main()
        tr.assert_called_once()
        assert (tmp_path / "results" / "Diarized diarized transcript.txt").exists()

    def test_title_sanitized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.pya.fetch_video_title", return_value="Bad/Name:Here"),
            patch("main.pytt.get_youtube_transcript", return_value="text"),
        ):
            parse.return_value = _make_args(
                input_path="https://www.youtube.com/watch?v=a",
                is_url=True,
            )
            main()
        assert (tmp_path / "results" / "Bad_Name_Here transcript.txt").exists()


class TestMainMediaBranch:
    def test_non_diarize_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        media = tmp_path / "video.mp4"
        media.write_text("")
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch(
                "main.plt.transcribe_video_file",
                return_value=("Hello world", "en"),
            ) as transcribe,
            patch("main.st.summarize_transcript_with_rag") as rag_summ,
        ):
            parse.return_value = _make_args(
                input_path=str(media),
                language=None,
                is_file=True,
                is_media_file=True,
            )
            main()
        transcribe.assert_called_once_with(str(media), model_size="small")
        out = tmp_path / "results" / "video transcript.txt"
        assert out.read_text() == "Hello world"
        rag_summ.assert_not_called()

    def test_diarize_path_writes_diarized_filename(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        media = tmp_path / "clip.mp4"
        media.write_text("")
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch(
                "main.plt.transcribe_video_file_with_diarization",
                return_value=("Alice: hi", "en"),
            ) as transcribe,
        ):
            parse.return_value = _make_args(
                input_path=str(media),
                diarize=True,
                is_file=True,
                is_media_file=True,
            )
            main()
        transcribe.assert_called_once()
        assert (tmp_path / "results" / "clip diarized transcript.txt").exists()


class TestMainTextBranch:
    def test_reads_text_and_calls_rag_when_summarize(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        txt = tmp_path / "t.txt"
        txt.write_text("Alice: hi")
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.st.summarize_transcript_with_rag") as rag_summ,
        ):
            parse.return_value = _make_args(
                input_path=str(txt),
                is_file=True,
                is_text_file=True,
                summarize=True,
            )
            main()
        rag_summ.assert_called_once_with("Alice: hi", "t", "en", model="mistral")

    def test_no_summarize_flag_skips_rag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        txt = tmp_path / "t.txt"
        txt.write_text("Alice: hi")
        with (
            patch("main.my_parser.parse_args") as parse,
            patch("main.initialize_logger"),
            patch("main.st.summarize_transcript_with_rag") as rag_summ,
            patch("main.st.summarize_transcript_with_openai") as openai_summ,
        ):
            parse.return_value = _make_args(
                input_path=str(txt),
                is_file=True,
                is_text_file=True,
                summarize=False,
            )
            main()
        rag_summ.assert_not_called()
        openai_summ.assert_not_called()
