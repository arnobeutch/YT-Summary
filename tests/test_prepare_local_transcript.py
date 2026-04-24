"""Tests for prepare_local_transcript — pure helpers only.

Entry points that call whisper / pyannote / ffmpeg are covered by integration
tests (opt-in, ``pytest -m integration``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pyannote.core import Segment

import yt_summary.transcription.local as plt
from yt_summary.transcription.local import (
    _MODEL_CACHE,
    extract_audio,
    get_device,
    group_speaker_segments,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetDevice:
    def test_cuda_when_available(self) -> None:
        with patch("yt_summary.transcription.local.torch.cuda.is_available", return_value=True):
            assert get_device() == "cuda"

    def test_cpu_fallback(self) -> None:
        with patch("yt_summary.transcription.local.torch.cuda.is_available", return_value=False):
            assert get_device() == "cpu"


class TestGroupSpeakerSegments:
    def test_empty(self) -> None:
        assert group_speaker_segments([]) == []

    def test_single(self) -> None:
        result = group_speaker_segments([("A", Segment(0.0, 1.0))])
        assert len(result) == 1
        assert result[0][0] == "A"
        assert result[0][1].start == 0.0
        assert result[0][1].end == 1.0

    def test_merges_same_speaker_within_gap(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("A", Segment(1.5, 2.0)),  # gap of 0.5 ≤ max_gap=1.0
        ]
        result = group_speaker_segments(segs, max_gap=1.0)
        assert len(result) == 1
        assert result[0][1].start == 0.0
        assert result[0][1].end == 2.0

    def test_does_not_merge_across_wide_gap(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("A", Segment(3.0, 4.0)),  # gap of 2.0 > max_gap=1.0
        ]
        result = group_speaker_segments(segs, max_gap=1.0)
        assert len(result) == 2

    def test_does_not_merge_different_speakers(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("B", Segment(1.0, 2.0)),
        ]
        result = group_speaker_segments(segs)
        assert len(result) == 2
        assert result[0][0] == "A"
        assert result[1][0] == "B"

    def test_alternating_speakers(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("B", Segment(1.0, 2.0)),
            ("A", Segment(2.0, 3.0)),
            ("A", Segment(3.2, 4.0)),  # merged with previous A
        ]
        result = group_speaker_segments(segs, max_gap=0.5)
        assert [speaker for speaker, _ in result] == ["A", "B", "A"]
        assert result[-1][1].start == 2.0
        assert result[-1][1].end == 4.0


class TestExtractAudio:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_audio(str(tmp_path / "nope.mp4"))


class TestModelCache:
    def test_model_loaded_once_on_repeated_calls(self) -> None:
        _MODEL_CACHE.clear()
        fake_model = object()
        with patch(
            "yt_summary.transcription.local.whisper.load_model", return_value=fake_model
        ) as load:
            m1 = plt._load_model("tiny", "cpu")
            m2 = plt._load_model("tiny", "cpu")
        load.assert_called_once_with("tiny", device="cpu")
        assert m1 is m2 is fake_model

    def test_different_keys_load_separate_models(self) -> None:
        _MODEL_CACHE.clear()
        model_a = object()
        model_b = object()
        with patch(
            "yt_summary.transcription.local.whisper.load_model",
            side_effect=[model_a, model_b],
        ) as load:
            ma = plt._load_model("tiny", "cpu")
            mb = plt._load_model("small", "cpu")
        assert load.call_count == 2
        assert ma is model_a
        assert mb is model_b
