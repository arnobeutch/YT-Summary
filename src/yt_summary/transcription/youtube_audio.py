# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Download YouTube audio for local transcription when captions aren't available."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yt_dlp

from yt_summary.logger import my_logger


def extract_video_id(url: str) -> str:
    """Extract the video ID from any common YouTube URL form.

    Supported: ``watch?v=<id>``, ``youtu.be/<id>``, ``embed/<id>``, ``shorts/<id>``.
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]
    v = parse_qs(parsed.query).get("v")
    if v:
        return v[0]
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"embed", "shorts", "v"}:
        return parts[1]
    err_msg = f"Could not extract video ID from URL: {url}"
    raise ValueError(err_msg)


def fetch_video_title(url: str) -> str:
    """Return the video title via yt-dlp metadata (no audio download)."""
    opts: Any = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = cast(dict[str, Any], ydl.extract_info(url, download=False))
    return cast(str, info.get("title") or info.get("id") or "unknown")


def download_youtube_audio(
    url: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, str]:
    """Download the audio track of a YouTube video as a wav file.

    Args:
        url: Full YouTube URL.
        output_dir: Directory to save the downloaded wav file in.
        force: If True, re-download even if the .wav already exists.

    Returns:
        ``(audio_path, video_title)`` — path to the downloaded wav and the
        video's human-readable title (unsanitized).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache hit: the .wav for this video id already exists.
    if not force:
        video_id = extract_video_id(url)
        cached_wav = output_dir / f"{video_id}.wav"
        if cached_wav.exists():
            my_logger.info(f"Using cached audio at {cached_wav}")
            title = fetch_video_title(url)
            return cached_wav, title

    my_logger.info(f"Downloading audio from {url}")

    opts: Any = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            },
        ],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = cast(dict[str, Any], ydl.extract_info(url, download=True))
    video_id = cast(str, info["id"])
    title = cast(str, info.get("title") or video_id)
    audio_path = output_dir / f"{video_id}.wav"
    if not audio_path.exists():
        err_msg = f"yt-dlp reported success but {audio_path} is missing"
        raise FileNotFoundError(err_msg)
    return audio_path, title
