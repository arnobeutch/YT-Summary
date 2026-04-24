# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Get YouTube transcripts via yt-dlp.

We use yt-dlp's caption-download path (rather than a separate API) so we can:
  * benefit from the same yt-dlp we already use for audio,
  * cleanly distinguish *manual* (author-provided) and *automatic* captions,
  * handle the same edge cases yt-dlp handles (member-only videos, region
    blocks, subtitle-disabled videos, etc.).

The current `get_youtube_transcript` keeps its previous API: given a video id,
return a single text string in the first available of (manual fr, auto fr,
manual en, auto en). The full language-selection ladder lands in Step 8.
"""

from __future__ import annotations

import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any, cast

import yt_dlp
from yt_dlp.utils import DownloadError

from my_logger import my_logger


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
    preferred_langs: tuple[str, ...] = ("fr", "en"),
) -> tuple[str, str] | None:
    """Return ``(lang, kind)`` for the best available caption track.

    ``kind`` is ``"manual"`` or ``"auto"``. Manual beats auto across languages
    (decided behavior; see plan §C "Language-selection decision tree").
    """
    subtitles = cast(dict[str, Any], info.get("subtitles") or {})
    auto = cast(dict[str, Any], info.get("automatic_captions") or {})

    for lang in preferred_langs:
        if lang in subtitles:
            return (lang, "manual")
    for lang in preferred_langs:
        if lang in auto:
            return (lang, "auto")
    # Last resort: any other language (manual first, then auto).
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


def get_youtube_transcript(video_id: str) -> str:
    """Fetch a YouTube transcript in French or English (manual preferred).

    Raises:
        TranscriptUnavailableError: No usable caption track is retrievable.
            Caller should route to the whisper-based fallback path.

    """
    url = _build_url(video_id)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        opts: Any = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["fr", "en"],
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

        pick = _pick_caption(info)
        if pick is None:
            raise TranscriptUnavailableError(
                "lang_not_found",
                "No French or English caption track listed for this video.",
            )

        lang, kind = pick
        my_logger.info(f"Using {kind} captions in '{lang}'")

        # yt-dlp writes <id>.<lang>.srt to the outtmpl directory.
        srt_path = tmpdir / f"{info['id']}.{lang}.srt"
        if not srt_path.exists():
            # Some tracks are listed but yt-dlp fails to write them (rare).
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

    return textwrap.fill(text, width=80, break_long_words=False, break_on_hyphens=False)
