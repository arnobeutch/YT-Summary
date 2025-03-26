"""AI agent to summarize YT videos.

Uses the YT package and the langchain_openai package.
"""

import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# see: https://pypi.org/project/openai/
from openai import OpenAI
from textblob import TextBlob

# see: https://pypi.org/project/youtube-transcript-api/
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

import my_constants
import my_parser

log = logging.getLogger(__name__)


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


def categorize_topic(text: str, language: str) -> str | None:
    """Use OpenAI to categorize the topic of the transcript."""
    if language == "fr":
        prompt = my_constants.CATEGORIZE_PROMPT_FR + f"{text[:1000]}"
    elif language == "en":
        prompt = my_constants.CATEGORIZE_PROMPT_EN + f"{text[:1000]}"
    else:
        return "Error: summarizer language not supported."

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
        )
        return response.choices[0].message.content
    except OpenAI.error.AuthenticationError:
        log.exception("AuthenticationError while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request was not authorized\nCheck or set your API key"
    except OpenAI.error.Timeout:
        log.exception("Timeout while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request timed out"
    except Exception:
        log.exception("Unexpected error while performing OpenAI API request", stack_info=True)
        return "Error: unexpected error while performing OpenAI API request"


def extract_keywords(text: str, language: str) -> str | None:
    """Use OpenAI to extract keywords from the transcript."""
    if language == "fr":
        prompt = my_constants.EXTRACT_KEYWORDS_PROMPT_FR + f"{text[:1000]}"
    elif language == "en":
        prompt = my_constants.EXTRACT_KEYWORDS_PROMPT_EN + f"{text[:1000]}"
    else:
        return "Error: summarizer language not supported."

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        return response.choices[0].message.content
    except OpenAI.error.AuthenticationError:
        log.exception("AuthenticationError while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request was not authorized\nCheck or set your API key"
    except OpenAI.error.Timeout:
        log.exception("Timeout while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request timed out"
    except Exception:
        log.exception("Unexpected error while performing OpenAI API request", stack_info=True)
        return "Error: unexpected error while performing OpenAI API request"


def summarize_transcript(transcript: str, language: str) -> str | None:
    """Summarize the video transcript into theme, key ideas, and takeaways."""
    if language == "fr":
        prompt = my_constants.SUMMARIZE_PROMPT_FR + f"{transcript}"
    elif language == "en":
        prompt = my_constants.SUMMARIZE_PROMPT_EN + f"{transcript}"
    else:
        return "Error: summarizer language not supported."

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide concise and insightful summaries."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except OpenAI.error.AuthenticationError:
        log.exception("AuthenticationError while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request was not authorized\nCheck or set your API key"
    except OpenAI.error.Timeout:
        log.exception("Timeout while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request timed out"
    except Exception:
        log.exception("Unexpected error while performing OpenAI API request", stack_info=True)
        return "Error: unexpected error while performing OpenAI API request"


def format_markdown(
    video_title: str,
    video_url: str,
    summary: str,
    sentiment: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 YouTube Video Summary
- Video title: {video_title}
- From: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    if language == "fr":
        return f"""
## 📺 Résumé Vidéo YouTube
- Titre de la vidéo: {video_title}
- De: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."


def format_markdown_extended(
    video_title: str,
    video_url: str,
    summary: str,
    sentiment: str,
    category: str,
    keywords: str,
    transcript: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
# 📺 YouTube Video Summary
- Video title: {video_title}
- From: {video_url}

## 🎯 Theme & Summary
{summary}

## 🔑 Key Insights
- **Sentiment:** {sentiment}
- **Topic Category:** {category}
- **Keywords:** {keywords}

## 📜 Transcript Extract
{transcript[:1000]}...
"""
    if language == "fr":
        return f"""
## 📺 Résumé Vidéo YouTube
- Titre de la vidéo: {video_title}
- De: {video_url}
- **Sentiment:** {sentiment}
- **Catégorie:** {category}
- **Mots Clefs:** {keywords}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."


def main() -> None:
    """Retrieve transcript and analyze."""
    args = my_parser.parse_args()

    # set up logger
    log.setLevel(logging.INFO if not args.verbose else logging.DEBUG)
    log_screen_handler = logging.StreamHandler(stream=sys.stdout)
    log.addHandler(log_screen_handler)
    log.propagate = False

    log.info(f"Script called with the following arguments: {vars(args)}")

    load_dotenv()  # declare your OPENAI_API_KEY in a .env file
    video_id = args.youtube_video_url.split("v=")[1]
    log.debug(f"Video ID: {video_id}")
    transcript = get_youtube_transcript(video_id)
    # get video title
    # TODO: add error handling
    r = requests.get(args.youtube_video_url, timeout=10)
    soup = BeautifulSoup(r.content, "html.parser")
    link = soup.find_all(name="title")[0]
    video_title = link.text
    log.info(f"Video title: {video_title}")

    if "Error" not in transcript:
        sentiment = analyze_sentiment(transcript)
        if args.transcript:
            if args.file:
                p = Path("transcript.txt")
                with p.open(encoding="utf8") as f:
                    my_text = f"Transcript:\n{transcript}\n\nSentiment: {sentiment}"
                    f.write(my_text)
            else:
                print(f"Transcript:\n{transcript}\n\n")
                print(f"Sentiment: {sentiment}")
            sys.exit(0)
        summary = summarize_transcript(transcript, args.language)
        if summary is None:
            log.error("Error: Summary could not be generated.")
            sys.exit(1)
        if args.extended:
            category = categorize_topic(transcript, args.language)
            keywords = extract_keywords(transcript, args.language)
            markdown_output = format_markdown_extended(
                video_title,
                args.youtube_video_url,
                summary,
                sentiment,
                str(category),
                str(keywords),
                transcript,
                args.language,
            )
        else:
            markdown_output = format_markdown(
                video_title,
                args.youtube_video_url,
                summary,
                sentiment,
                args.language,
            )
        if args.file:
            p = Path("summary.md")
            with p.open(encoding="utf8") as f:
                f.write(markdown_output)
        else:
            print(markdown_output)
    else:
        print(transcript)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.critical("Interrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
