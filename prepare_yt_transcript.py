"""Get, format and analyze (sentiment) YouTube transcript."""

import json
import textwrap
from xml.etree.ElementTree import ParseError as XMLParseError

# see: https://pypi.org/project/youtube-transcript-api/
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, NoTranscriptFound

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


def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video in English or French.

    Raises:
        TranscriptUnavailableError: No caption track is retrievable. Caller
            should route to the whisper-based fallback path.

    """
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
    except json.decoder.JSONDecodeError as exc:
        # catching this due to https://github.com/jdepoix/youtube-transcript-api/issues/407
        my_logger.exception("JSONDecodeError while getting transcripts list", stack_info=True)
        raise TranscriptUnavailableError(
            "list_decode_error",
            "Transcript list could not be decoded.",
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        # TranscriptsDisabled, VideoUnavailable, IpBlocked, AgeRestricted, etc.
        my_logger.exception("Could not list transcripts", stack_info=True)
        raise TranscriptUnavailableError(
            "list_failed",
            "Transcript unavailable (disabled, blocked, or removed).",
        ) from exc

    # filter for transcripts, french first, otherwise english
    # note: youtube_transcript_api always chooses manually created transcripts over automatically created ones
    try:
        transcript = transcript_list.find_transcript(["fr", "en"])
    except NoTranscriptFound as exc:
        my_logger.exception(
            "NoTranscriptFound while searching transcripts in French or English",
            stack_info=True,
        )
        raise TranscriptUnavailableError(
            "lang_not_found",
            "No French or English transcript available.",
        ) from exc

    try:
        fetched_transcript = transcript.fetch()
    except XMLParseError as exc:
        # Empty XML body — YouTube returned no transcript content even though
        # the transcript was listed (can happen on videos without real captions).
        my_logger.exception("Empty transcript payload while fetching", stack_info=True)
        raise TranscriptUnavailableError(
            "empty_payload",
            "Transcript payload was empty.",
        ) from exc

    transcript_text = " ".join([entry.text for entry in fetched_transcript])
    return textwrap.fill(transcript_text, width=80)
