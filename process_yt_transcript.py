"""Get, format and analyze (sentiment) YouTube transcript."""

import json
import textwrap

# see: https://pypi.org/project/youtube-transcript-api/
from textblob import TextBlob
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

import my_constants
from main import log


def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video in English or French."""
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
    except json.decoder.JSONDecodeError:
        # catching this due to https://github.com/jdepoix/youtube-transcript-api/issues/407
        log.exception("JSONDecodeError while getting transcripts list", stack_info=True)
        return "Error: transcript list not found"
    # filter for transcripts, french first, otherwise english
    # note: youtube_transcript_api always chooses manually created transcripts over automatically created ones
    try:
        transcript = transcript_list.find_transcript(["fr", "en"])
    except NoTranscriptFound:
        log.exception("NoTranscriptFound while searching transcripts in French or English", stack_info=True)
        return "Error: Transcript not found."

    fetched_transcript = transcript.fetch()

    # Combine text
    transcript_text = " ".join([entry.text for entry in fetched_transcript])
    return textwrap.fill(transcript_text, width=80)


def analyze_sentiment(text: str) -> str:
    """Determine the sentiment of the transcript."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > my_constants.POLARITY_POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < my_constants.POLARITY_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"
