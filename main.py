"""
AI agent to summarize YT videos
Uses the YT package and the langchain_openai package
"""

import sys
import argparse
import logging
import os
import textwrap
import json
import traceback

from dotenv import load_dotenv

# see: https://pypi.org/project/youtube-transcript-api/
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

from bs4 import BeautifulSoup
import requests

# see: https://pypi.org/project/openai/
from openai import OpenAI

import my_constants

# see:
#   * https://pypi.org/project/textblob/
#   * https://textblob.readthedocs.io/en/dev/api_reference.html
#   * https://pypi.org/project/textblob/
from textblob import TextBlob

log = logging.getLogger(__name__)


def get_youtube_transcript(video_id: str) -> str:
    """Fetches the transcript of a YouTube video in English or French."""
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
    except json.decoder.JSONDecodeError as e:
        # getting a "Extra data: line 1 column 56651 (char 56650)"
        # similar to https://github.com/jdepoix/youtube-transcript-api/issues/144
        # on one specific (and private) video
        log.debug(f"JSONDecodeError while getting transcripts list: {e}")
        log.debug(f"Traceback: {traceback.format_exc()}")
        return "Error: transcript list not found"
    # filter for transcripts, french first, otherwise english
    # note: youtube_transcript_api always chooses manually created transcripts over automatically created ones
    try:
        transcript = transcript_list.find_transcript(["fr", "en"])
    except NoTranscriptFound:
        return "Error: Transcript not found."

    fetched_transcript = transcript.fetch()

    # Combine text
    transcript_text = " ".join([entry.text for entry in fetched_transcript])
    transcript_text = textwrap.fill(transcript_text, width=80)
    return transcript_text


def analyze_sentiment(text: str) -> str:
    """Determines the sentiment of the transcript."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"


def categorize_topic(text: str, language: str) -> str | None:
    """Uses OpenAI to categorize the topic of the transcript."""
    if language == "fr":
        prompt = my_constants.CATEGORIZE_PROMPT_FR + f"{text[:1000]}"
    elif language == "en":
        prompt = my_constants.CATEGORIZE_PROMPT_EN + f"{text[:1000]}"
    else:
        return "Error: Language not supported."
    # f"Classify the following video transcript into one broad category (e.g., Technology, Business, Education, Science, Entertainment, Motivation, Health, News, etc.):\n\n{text[:1000]}"

    # TODO: catch "no API key"
    client = OpenAI()
    # TODO: catch other shit
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=30
    )
    return response.choices[0].message.content


def extract_keywords(text: str, language: str) -> str | None:
    """Uses OpenAI to extract keywords from the transcript."""
    if language == "fr":
        prompt = my_constants.EXTRACT_KEYWORDS_PROMPT_FR + f"{text[:1000]}"
    elif language == "en":
        prompt = my_constants.EXTRACT_KEYWORDS_PROMPT_EN + f"{text[:1000]}"
    else:
        return "Error: Language not supported."
    # prompt = f"Extract the top 5 keywords from the following transcript:\n\n{text[:1000]}"

    # TODO: catch "no API key"
    client = OpenAI()
    # TODO: catch other shit
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=100
    )
    return response.choices[0].message.content


def summarize_transcript(transcript: str, language: str) -> str | None:
    """Summarizes the video transcript into theme, key ideas, and takeaways."""
    if language == "fr":
        prompt = my_constants.SUMMARIZE_PROMPT_FR + f"{transcript}"
    elif language == "en":
        prompt = my_constants.SUMMARIZE_PROMPT_EN + f"{transcript}"
    else:
        return "Error: Language not supported."

    # TODO: catch "no API key"
    client = OpenAI()
    # TODO: catch other shit
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You provide concise and insightful summaries."},
            {"role": "user", "content": prompt},
        ],
        # max_tokens=300
    )
    return response.choices[0].message.content

def format_markdown(
    video_title: str, video_url: str, summary: str, sentiment: str, language: str
) -> str:
    """Formats the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 YouTube Video Summary
- Video title: {video_title}
- From: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    elif language == "fr":
        return f"""
## 📺 Résumé Vidéo YouTube
- Titre de la vidéo: {video_title}
- De: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    else:
        return "Error: Language not supported."


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
    """Formats the final output in Markdown."""
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
    elif language == "fr":
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
    else:
        return "Error: Language not supported."


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description=(
            "A python script to summarize YouTube videos."
            + "\nThe script will output a markdown formatted summary of the video."
            + "\nNote 1: The video must have either English or French subtitles."
            + "\n\tThe script will output an error message if no such transcript is found."
            + "\nNote 2: You must have an OpenAI API key to use this script, and credited tokens."
            + "\n\tDeclare your key in an environment variable OPENAI_API_KEY prior to running the script."
            + "\nUse the -h flag for help."
        ),
        usage=(
            "python main.py <youtube_video_url> [optional_arguments]"
            + "\nExample: python main.py https://www.youtube.com/watch?v=VIDEO_ID"
        ),
        epilog="Be careful: summarizing burns OpenAI API tokens!",
    )
    # mandatory argument: youtube video url
    parser.add_argument("youtube_video_url", help="URL of the YouTube video to summarize")
    # optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Increases logging verbosity")
    parser.add_argument(
        "-l",
        "--language",
        choices={"en", "fr"},
        default="en",
        help="Use en or fr to specify the language of the summary (default: en)",
    )
    parser.add_argument(
        "-e",
        "--extended",
        action="store_true",
        default=False,
        help="Add category, keywords and partial transcript to the output",
    )
    parser.add_argument(
        "-t",
        "--transcript",
        action="store_true",
        default=False,
        help="Output the full transcript & sentiment instead of a summary",
    )
    # TODO: add option to output to files (summary.md and transcript.txt)
    args = parser.parse_args()

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
        #     category = categorize_topic(transcript, args.language)
        #     keywords = extract_keywords(transcript, args.language)
        if args.transcript:
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
            # TODO: control empty responses
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
                video_title, args.youtube_video_url, summary, sentiment, args.language
            )
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
