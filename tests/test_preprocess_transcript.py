"""Tests for preprocess_transcript."""

from __future__ import annotations

from yt_summary.transcription.preprocess import parse_transcript, try_resolve_speaker_names


class TestParseTranscript:
    def test_empty_string(self) -> None:
        assert parse_transcript("") == []

    def test_single_utterance(self) -> None:
        assert parse_transcript("Alice: Hello") == [("Alice", "Hello")]

    def test_multi_line(self) -> None:
        raw = "Alice: Hello\nBob: Hi there\n"
        assert parse_transcript(raw) == [("Alice", "Hello"), ("Bob", "Hi there")]

    def test_skips_lines_without_separator(self) -> None:
        raw = "Alice: Hi\nNot a speaker line\nBob: Hey"
        assert parse_transcript(raw) == [("Alice", "Hi"), ("Bob", "Hey")]

    def test_strips_whitespace(self) -> None:
        assert parse_transcript("  Alice : Hello  ") == [("Alice", "Hello")]

    def test_text_with_colon_inside(self) -> None:
        # ": " split with maxsplit=1 — inner colons preserved
        assert parse_transcript("Alice: time: 3:00 PM") == [("Alice", "time: 3:00 PM")]

    def test_single_colon_no_space_skipped(self) -> None:
        # "a:b" has no ": " sep → skipped
        assert parse_transcript("Alice:Hello") == []


class TestTryResolveSpeakerNames:
    def test_empty_list(self) -> None:
        assert try_resolve_speaker_names([]) == []

    def test_no_nominative_clue(self) -> None:
        utt = [("SPEAKER_00", "Hello world, nothing to see here")]
        assert try_resolve_speaker_names(utt) == utt

    def test_merci_clue(self) -> None:
        utt = [("SPEAKER_00", "Merci Bernard pour ton intervention")]
        result = try_resolve_speaker_names(utt)
        assert result == [("Bernard", "Merci Bernard pour ton intervention")]

    def test_comme_disait_clue(self) -> None:
        utt = [("SPEAKER_01", "comme disait Alice, oui")]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Alice"

    def test_selon_clue(self) -> None:
        utt = [("SPEAKER_02", "selon Marie, c'est important")]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Marie"

    def test_parlait_de_clue(self) -> None:
        utt = [("S3", "il parlait de Pierre hier")]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Pierre"

    def test_case_insensitive(self) -> None:
        utt = [("S0", "MERCI BERNARD")]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Bernard"

    def test_accented_name_transliterated(self) -> None:
        # unidecode strips the accent before regex; "Hélène" → "Helene"
        utt = [("S0", "Merci Hélène pour ton aide")]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Helene"

    def test_mapping_applies_to_future_utterances(self) -> None:
        utt = [
            ("SPEAKER_00", "Merci Bernard"),
            ("SPEAKER_00", "ongoing monologue"),
            ("SPEAKER_01", "Hello"),
        ]
        result = try_resolve_speaker_names(utt)
        assert result == [
            ("Bernard", "Merci Bernard"),
            ("Bernard", "ongoing monologue"),
            ("SPEAKER_01", "Hello"),
        ]

    def test_first_match_wins_for_same_speaker(self) -> None:
        utt = [
            ("SPEAKER_00", "Merci Bernard"),
            ("SPEAKER_00", "Merci Alice"),  # same speaker — ignored, already mapped
        ]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Bernard"
        assert result[1][0] == "Bernard"

    def test_different_speakers_mapped_independently(self) -> None:
        utt = [
            ("S0", "Merci Bernard"),
            ("S1", "Merci Alice"),
        ]
        result = try_resolve_speaker_names(utt)
        assert result[0][0] == "Bernard"
        assert result[1][0] == "Alice"
