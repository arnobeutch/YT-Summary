"""Text formatting helpers shared across the URL / media / text handlers."""

from __future__ import annotations

import textwrap

_ILLEGAL_FILENAME_CHARS = ("<", ">", ":", '"', "/", "\\", "|", "?", "*")


def sanitize_filename(name: str) -> str:
    """Replace OS-illegal characters in a filename with underscores."""
    for char in _ILLEGAL_FILENAME_CHARS:
        name = name.replace(char, "_")
    return name


def wrap_transcript(text: str, *, diarize: bool, width: int) -> str:
    """Soft-wrap a transcript without breaking words.

    Skipped for diarized output — `SPEAKER: text` lines must stay intact.
    """
    if diarize:
        return text
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
