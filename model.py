"""Shared domain dataclasses (no business logic).

Lives in its own module to avoid import cycles between handlers and the
summarizers package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TranscriptSource = Literal["yt_manual", "yt_auto", "whisper", "file"]


@dataclass(frozen=True)
class Transcript:
    """In-memory representation of a transcript ready to be written / summarized."""

    text: str
    language: str  # the *summary* language, derived from source
    title: str
    source: TranscriptSource
    diarized: bool
