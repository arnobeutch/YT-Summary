# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Get YouTube transcripts via yt-dlp.

We use yt-dlp's caption-download path (rather than a separate API) so we can:
  * benefit from the same yt-dlp we already use for audio,
  * cleanly distinguish *manual* (author-provided) and *automatic* captions,
  * handle the same edge cases yt-dlp handles (member-only videos, region
    blocks, subtitle-disabled videos, etc.).

``get_youtube_transcript`` returns a :class:`CaptionTrack` carrying the
caption text, its actual language code, and whether it came from a manual or
auto-generated track. The pick follows the ladder spelled out in the README's
"Language selection" section (manual beats auto across languages).
"""

from __future__ import annotations

import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yt_dlp
from yt_dlp.utils import DownloadError

from my_logger import my_logger

CaptionKind = Literal["manual", "auto"]


@dataclass(frozen=True)
class CaptionTrack:
    """A retrieved caption track ready for downstream use."""

    text: str
    lang: str
    kind: CaptionKind


class TranscriptUnavailableError(Exception):
    """Raised when no YouTube transcript can be obtained.

    Callers should catch this and fall back to local transcription.
    The ``reason`` attribute carries a short machine-friendly tag; the
    exception message is human-readable.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Initialize with a tag and a message."""
        super().__init__(message)
        self.reason = reason


def _build_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _pick_caption(
    info: dict[str, Any],
    requested_lang: str | None = None,
) -> tuple[str, CaptionKind] | None:
    """Return ``(lang, kind)`` for the best available caption track.

    Priority, top-down (manual beats auto across languages):
      1. Manual @ ``requested_lang`` (if set).
      2. Auto @ ``requested_lang`` (if set).
      3. Manual @ ``"en"``.
      4. Auto @ ``"en"``.
      5. Manual @ any other language (first encountered).
      6. Auto @ any other language (first encountered).

    """
    subtitles = cast(dict[str, Any], info.get("subtitles") or {})
    auto = cast(dict[str, Any], info.get("automatic_captions") or {})

    preferred: list[str] = []
    if requested_lang:
        preferred.append(requested_lang)
    if "en" not in preferred:
        preferred.append("en")

    for lang in preferred:
        if lang in subtitles:
            return (lang, "manual")
    for lang in preferred:
        if lang in auto:
            return (lang, "auto")
    if subtitles:
        return (next(iter(subtitles)), "manual")
    if auto:
        return (next(iter(auto)), "auto")
    return None


_TIMESTAMP_LINE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}.*$",
)
_INDEX_LINE = re.compile(r"^\d+$")
_VTT_HEADER_LINE = re.compile(r"^(WEBVTT|Kind:|Language:)")
_TAG = re.compile(r"<[^>]+>")


def _extract_text_from_subtitle_file(path: Path) -> str:
    """Strip timestamps + cue indices from an SRT/VTT subtitle file.

    Returns a single string with cue text joined by spaces, deduplicated
    against consecutive identical lines (auto captions are noisy this way).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    last: str = ""
    for raw_line in raw.splitlines():
        line = _TAG.sub("", raw_line).strip()
        if not line:
            continue
        if _TIMESTAMP_LINE.match(line) or _INDEX_LINE.match(line) or _VTT_HEADER_LINE.match(line):
            continue
        if line == last:
            continue
        lines.append(line)
        last = line
    return " ".join(lines)


def get_youtube_transcript(video_id: str, requested_lang: str | None = None) -> CaptionTrack:
    """Fetch a YouTube transcript honoring the language-selection ladder.

    Args:
        video_id: YouTube video ID (the part after ``v=`` or after
            ``youtu.be/``).
        requested_lang: Optional preferred language code (typically
            ``"fr"`` or ``"en"``). When set, manual @ this lang outranks
            manual @ English.

    Raises:
        TranscriptUnavailableError: No usable caption track is retrievable.
            Caller should route to the whisper-based fallback path.

    """
    url = _build_url(video_id)

    # Ask yt-dlp for both the requested language (if any) and English so we
    # always have a sensible fallback. Other languages get downloaded only
    # when the picker selects them — but we hint at them up-front so yt-dlp
    # actually fetches the file.
    sub_langs: list[str] = []
    if requested_lang:
        sub_langs.append(requested_lang)
    if "en" not in sub_langs:
        sub_langs.append("en")
    sub_langs.append("all")  # fallback so yt-dlp doesn't refuse other langs

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        opts: Any = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": sub_langs,
            "subtitlesformat": "srt",
            "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = cast(dict[str, Any], ydl.extract_info(url, download=True))
        except DownloadError as exc:
            my_logger.exception("yt-dlp could not retrieve caption metadata", stack_info=True)
            raise TranscriptUnavailableError(
                "list_failed",
                "yt-dlp could not retrieve caption metadata.",
            ) from exc

        pick = _pick_caption(info, requested_lang=requested_lang)
        if pick is None:
            raise TranscriptUnavailableError(
                "lang_not_found",
                "No caption track found in any language.",
            )

        lang, kind = pick
        my_logger.info(f"Using {kind} captions in '{lang}' (requested: {requested_lang or 'auto'})")

        srt_path = tmpdir / f"{info['id']}.{lang}.srt"
        if not srt_path.exists():
            raise TranscriptUnavailableError(
                "empty_payload",
                f"yt-dlp listed {kind} captions in '{lang}' but produced no file.",
            )

        text = _extract_text_from_subtitle_file(srt_path)

    if not text.strip():
        raise TranscriptUnavailableError(
            "empty_payload",
            "Caption file existed but contained no usable text.",
        )

    wrapped = textwrap.fill(text, width=80, break_long_words=False, break_on_hyphens=False)
    return CaptionTrack(text=wrapped, lang=lang, kind=kind)
