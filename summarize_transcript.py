"""Analyze video transcripts (YT, local) with OpenAI API."""

from pathlib import Path
from typing import Any, cast

import openai
from openai import OpenAI  # see: https://pypi.org/project/openai/
from textblob import TextBlob

import my_constants
from markdown_writer import format_summary_markdown, simple_format_markdown
from my_logger import my_logger
from my_settings import Settings
from preprocess_transcript import parse_transcript, try_resolve_speaker_names
from q_and_a_engine import generate_summary


def analyze_sentiment(text: str) -> str:
    """Determine the sentiment of the transcript."""
    # textblob's `.sentiment` is a cached_property with partially-unknown typing; cast to skip.
    blob = cast(Any, TextBlob(text))
    polarity: float = blob.sentiment.polarity

    if polarity > my_constants.POLARITY_POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < my_constants.POLARITY_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def summarize_transcript_with_rag(
    transcript: str,
    video_title: str,
    language: str,
    model: str = "mistral",
) -> Path:
    """Return path to saved markdown summary of a meeting transcript.

    Args:
        transcript: Raw transcript string ("<speaker>: <text>" format).
        video_title: Title used for the output filename and markdown heading.
        language: 'fr' or 'en'.
        model: Ollama model to use (e.g. 'mistral').

    Returns:
        Path to the markdown summary in ./results/.

    """
    my_logger.info("Parsing transcript...")
    utterances = parse_transcript(transcript)
    utterances = try_resolve_speaker_names(utterances)

    my_logger.info("Generating summary via RAG...")
    try:
        raw_summary = generate_summary(utterances, language=language, model=model)
    except Exception:
        my_logger.exception("Error generating summary")
        raise

    my_logger.info("Formatting markdown...")
    formatted = format_summary_markdown(
        raw_summary,
        filename_stem=video_title,
        language=language,
    )
    suffix = "résumé" if language == "fr" else "summary"
    output_filename = f"{video_title} - {suffix}.md"
    output_path = Path("results") / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(formatted)

    my_logger.info(f"Summary written to {output_path}")
    return output_path


def summarize_transcript_with_openai(
    transcript: str,
    input_path: str,
    video_title: str,
    language: str,
) -> None:
    """Summarize the video transcript into theme, key ideas, and takeaways using OpenAI API."""
    if language == "fr":
        prompt = my_constants.OPENAI_PROMPT_FR + f"{transcript}"
    elif language == "en":
        prompt = my_constants.OPENAI_PROMPT_EN + f"{transcript}"
    else:
        err_msg = "Error: summarizer language not supported."
        raise ValueError(err_msg)

    sentiment = analyze_sentiment(transcript)

    Settings.from_env()  # populate os.environ from .env (OPENAI_API_KEY etc.)
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide concise and insightful summaries."},
                {"role": "user", "content": prompt},
            ],
        )
    except openai.AuthenticationError:
        my_logger.exception("AuthenticationError while performing OpenAI API request")
        return
    except openai.APITimeoutError:
        my_logger.exception("Timeout while performing OpenAI API request")
        return
    except openai.OpenAIError:
        # Covers missing OPENAI_API_KEY at client construction as well as
        # runtime API errors (base class of all openai exceptions).
        my_logger.exception("OpenAI API error — is OPENAI_API_KEY set in .env or the environment?")
        return

    content = response.choices[0].message.content
    if content is None:
        my_logger.error("OpenAI returned empty content")
        return

    markdown_output = simple_format_markdown(
        video_title,
        input_path,
        content,
        sentiment,
        language,
    )
    p = Path(f"./results/{video_title}.md")
    with p.open(mode="w", encoding="utf8") as f:
        f.write(markdown_output)
