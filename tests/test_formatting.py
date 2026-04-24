"""Tests for the formatting helpers."""

from __future__ import annotations

from formatting import sanitize_filename, wrap_transcript


class TestSanitizeFilename:
    def test_replaces_illegal_chars(self) -> None:
        assert sanitize_filename("a<b>c:d") == "a_b_c_d"

    def test_preserves_normal(self) -> None:
        assert sanitize_filename("hello world") == "hello world"

    def test_all_illegal_chars(self) -> None:
        assert sanitize_filename('<>:"/\\|?*') == "_________"

    def test_empty(self) -> None:
        assert sanitize_filename("") == ""


class TestWrapTranscript:
    def test_wraps_long_single_line(self) -> None:
        text = "word " * 50  # ~250 chars, no existing line breaks
        out = wrap_transcript(text, diarize=False, width=80)
        for line in out.splitlines():
            assert len(line) <= 80

    def test_does_not_break_words(self) -> None:
        # A single word longer than the wrap width must not be split.
        long_word = "supercalifragilistic" * 5
        out = wrap_transcript(long_word, diarize=False, width=80)
        assert long_word in out

    def test_diarized_pass_through(self) -> None:
        text = "SPEAKER_00: hello there, this is a long speaker line\nSPEAKER_01: hi"
        assert wrap_transcript(text, diarize=True, width=80) == text

    def test_short_text_unchanged(self) -> None:
        assert wrap_transcript("short", diarize=False, width=80) == "short"

    def test_custom_wrap_width(self) -> None:
        text = "word " * 50
        out = wrap_transcript(text, diarize=False, width=40)
        for line in out.splitlines():
            assert len(line) <= 40
