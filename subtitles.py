"""Subtitle (SRT / VTT) writers.

Whisper returns ``segments`` like::

    [{"start": 0.0, "end": 2.4, "text": " Hello world"}, ...]

We accept that loose shape (a list of dicts with ``start``/``end``/``text``)
and emit standard SRT and VTT files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def _format_timestamp(seconds: float, *, separator: str) -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm``.

    SRT uses ``,`` as the millisecond separator; VTT uses ``.``.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def write_srt(segments: list[dict[str, Any]], path: Path) -> None:
    """Write whisper-style segments to ``path`` in SRT format."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(float(seg.get("start", 0.0)), separator=",")
        end = _format_timestamp(float(seg.get("end", 0.0)), separator=",")
        text = str(seg.get("text", "")).strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line between cues
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict[str, Any]], path: Path) -> None:
    """Write whisper-style segments to ``path`` in WebVTT format."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(float(seg.get("start", 0.0)), separator=".")
        end = _format_timestamp(float(seg.get("end", 0.0)), separator=".")
        text = str(seg.get("text", "")).strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
