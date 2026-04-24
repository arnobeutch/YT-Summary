"""Tests for the per-source handlers (URL / media / text) + write/summarize."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from handlers import (
    Transcript,
    handle_media,
    handle_text,
    handle_url,
    summarize,
    write_transcript_file,
)
from my_settings import Settings
from prepare_yt_transcript import TranscriptUnavailableError

if TYPE_CHECKING:
    from pathlib import Path


def _args(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "input_path": "",
        "language": "en",
        "diarize": False,
        "summarize": False,
        "with_openai": False,
        "model_size": None,
        "llm_provider": None,
        "llm_model": None,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "log_level": "INFO",
        "openai_api_key": None,
        "openrouter_api_key": None,
        "huggingface_token": None,
        "llm_provider": "openai",
        "llm_model": None,
        "openai_model": "gpt-4o",
        "ollama_model": "mistral",
        "whisper_model_size": "small",
        "wrap_width": 80,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]  # frozen dataclass kwargs


class TestHandleUrl:
    def test_caption_happy_path(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("handlers.pya.extract_video_id", return_value="vid"),
            patch("handlers.pya.fetch_video_title", return_value="My Video"),
            patch("handlers.pytt.get_youtube_transcript", return_value="caption text"),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.text == "caption text"
        assert t.title == "My Video"
        assert t.source == "yt_caption"
        assert t.diarized is False

    def test_falls_back_to_whisper_when_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "no caps"),
            ),
            patch(
                "handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Remote Video"),
            ),
            patch(
                "handlers.plt.transcribe_audio",
                return_value=("transcribed body", "fr"),
            ) as transcribe,
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.text == "transcribed body"
        assert t.language == "fr"
        assert t.title == "Remote Video"
        assert t.source == "whisper"
        # Default model size from Settings ("small").
        transcribe.assert_called_once_with(str(tmp_path / "audio.wav"), model_size="small")

    def test_fallback_with_diarization(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("empty_payload", "empty"),
            ),
            patch(
                "handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Diarized"),
            ),
            patch(
                "handlers.plt.transcribe_audio_with_diarization",
                return_value=("Alice: hi", "en"),
            ) as transcribe,
        ):
            t = handle_url(
                _args(input_path="https://y.com/watch?v=vid", diarize=True),
                s,
            )
        assert t.diarized is True
        assert t.text == "Alice: hi"
        transcribe.assert_called_once()

    def test_settings_model_size_propagates(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            downloads_dir=tmp_path / "dl",
            whisper_model_size="medium",
        )
        with (
            patch("handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "x"),
            ),
            patch(
                "handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "X"),
            ),
            patch(
                "handlers.plt.transcribe_audio",
                return_value=("body", "en"),
            ) as transcribe,
        ):
            handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        transcribe.assert_called_once_with(str(tmp_path / "audio.wav"), model_size="medium")

    def test_title_sanitized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("handlers.pya.extract_video_id", return_value="vid"),
            patch("handlers.pya.fetch_video_title", return_value="Bad/Name:Here"),
            patch("handlers.pytt.get_youtube_transcript", return_value="t"),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.title == "Bad_Name_Here"


class TestHandleMedia:
    def test_non_diarize(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        with patch(
            "handlers.plt.transcribe_video_file",
            return_value=("Hello world", "en"),
        ) as transcribe:
            t = handle_media(_args(input_path=str(media), language=None), s)
        assert t.text == "Hello world"
        assert t.title == "video"
        assert t.source == "whisper"
        transcribe.assert_called_once_with(str(media), model_size="small")

    def test_diarize(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        with patch(
            "handlers.plt.transcribe_video_file_with_diarization",
            return_value=("Alice: hi", "en"),
        ):
            t = handle_media(
                _args(input_path=str(media), language=None, diarize=True),
                s,
            )
        assert t.diarized is True
        assert t.text == "Alice: hi"


class TestHandleText:
    def test_reads_file(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        txt = tmp_path / "t.txt"
        txt.write_text("Alice: hi", encoding="utf-8")
        result = handle_text(_args(input_path=str(txt), language="en"), s)
        assert result.text == "Alice: hi"
        assert result.title == "t"
        assert result.source == "file"


class TestWriteTranscriptFile:
    def test_writes_non_diarized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="hello", language="en", title="vid", source="yt_caption", diarized=False
        )
        out = write_transcript_file(t, s)
        assert out == tmp_path / "out" / "vid transcript.txt"
        assert out.read_text() == "hello"

    def test_writes_diarized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(text="A: x", language="en", title="vid", source="whisper", diarized=True)
        out = write_transcript_file(t, s)
        assert out == tmp_path / "out" / "vid diarized transcript.txt"
        assert out.read_text() == "A: x"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "deep" / "nested")
        t = Transcript(text="x", language="en", title="t", source="file", diarized=False)
        write_transcript_file(t, s)
        assert (tmp_path / "deep" / "nested").is_dir()


class TestSummarize:
    def test_openai_dispatch(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="body", language="en", title="title", source="yt_caption", diarized=False
        )
        with patch("handlers.st.summarize_transcript_with_openai") as openai_summ:
            summarize(t, _args(input_path="u", with_openai=True), s)
        openai_summ.assert_called_once_with("body", "u", "title", "en")

    def test_rag_dispatch_uses_settings_default_model(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", ollama_model="mistral")
        t = Transcript(
            text="body", language="en", title="title", source="yt_caption", diarized=False
        )
        with patch("handlers.st.summarize_transcript_with_rag") as rag:
            summarize(t, _args(input_path="u", with_openai=False), s)
        rag.assert_called_once_with("body", "title", "en", model="mistral")

    def test_rag_dispatch_settings_llm_model_overrides(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            ollama_model="mistral",
            llm_model="gemma4:e4b",
        )
        t = Transcript(
            text="body", language="en", title="title", source="yt_caption", diarized=False
        )
        with patch("handlers.st.summarize_transcript_with_rag") as rag:
            summarize(t, _args(input_path="u", with_openai=False), s)
        rag.assert_called_once_with("body", "title", "en", model="gemma4:e4b")
