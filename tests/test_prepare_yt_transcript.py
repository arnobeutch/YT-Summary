"""Tests for prepare_yt_transcript."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from xml.etree.ElementTree import ParseError as XMLParseError

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from prepare_yt_transcript import TranscriptUnavailableError, get_youtube_transcript


def _make_not_found() -> NoTranscriptFound:
    """Construct NoTranscriptFound bypassing its __init__ (needs internal TranscriptList)."""
    return NoTranscriptFound.__new__(NoTranscriptFound)


class TestGetYoutubeTranscript:
    def test_json_decode_error_raises_unavailable(self) -> None:
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            mock_api.return_value.list.side_effect = json.JSONDecodeError("x", "doc", 0)
            with pytest.raises(TranscriptUnavailableError) as excinfo:
                get_youtube_transcript("vid123")
        assert excinfo.value.reason == "list_decode_error"

    def test_transcripts_disabled_raises_unavailable(self) -> None:
        exc = TranscriptsDisabled.__new__(TranscriptsDisabled)
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            mock_api.return_value.list.side_effect = exc
            with pytest.raises(TranscriptUnavailableError) as excinfo:
                get_youtube_transcript("vid")
        assert excinfo.value.reason == "list_failed"

    def test_no_transcript_found_raises_unavailable(self) -> None:
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            transcript_list = MagicMock()
            transcript_list.find_transcript.side_effect = _make_not_found()
            mock_api.return_value.list.return_value = transcript_list
            with pytest.raises(TranscriptUnavailableError) as excinfo:
                get_youtube_transcript("vid")
        assert excinfo.value.reason == "lang_not_found"

    def test_xml_parse_error_raises_unavailable(self) -> None:
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            transcript = MagicMock()
            transcript.fetch.side_effect = XMLParseError("no element found: line 1, column 0")
            transcript_list = MagicMock()
            transcript_list.find_transcript.return_value = transcript
            mock_api.return_value.list.return_value = transcript_list
            with pytest.raises(TranscriptUnavailableError) as excinfo:
                get_youtube_transcript("vid")
        assert excinfo.value.reason == "empty_payload"

    def test_happy_path_joins_and_wraps(self) -> None:
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            e1, e2 = MagicMock(), MagicMock()
            e1.text = "Hello world this is a"
            e2.text = "transcription of a video"
            transcript = MagicMock()
            transcript.fetch.return_value = [e1, e2]
            transcript_list = MagicMock()
            transcript_list.find_transcript.return_value = transcript
            mock_api.return_value.list.return_value = transcript_list
            result = get_youtube_transcript("vid")
        assert "Hello world" in result
        assert "transcription" in result
        for line in result.splitlines():
            assert len(line) <= 80

    def test_prefers_fr_en_order(self) -> None:
        with patch("prepare_yt_transcript.YouTubeTranscriptApi") as mock_api:
            transcript_list = MagicMock()
            transcript = MagicMock()
            transcript.fetch.return_value = []
            transcript_list.find_transcript.return_value = transcript
            mock_api.return_value.list.return_value = transcript_list
            get_youtube_transcript("vid")
            transcript_list.find_transcript.assert_called_once_with(["fr", "en"])
