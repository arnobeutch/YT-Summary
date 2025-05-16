"""Analyze video transcripts (YT, local) with OpenAI API."""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI  # see: https://pypi.org/project/openai/
from textblob import TextBlob

import my_constants
from markdown_writer import format_summary_markdown, simple_format_markdown
from my_logger import my_logger
from preprocess_transcript import parse_transcript, try_resolve_speaker_names
from q_and_a_engine import generate_summary


def analyze_sentiment(text: str) -> str:
    """Determine the sentiment of the transcript."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > my_constants.POLARITY_POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < my_constants.POLARITY_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def summarize_transcript_with_rag(
    transcript: str,
    input_filename: str,
    # TODO: adapt to youtybe videos
    # if transcript was obtained from youtube, then input_filename is irrelevant
    # if transcript was generated locally from downloaded youtube audio, then input_filename is relevant
    video_title: str,
    language: str,
    model: str = "mistral",
) -> Path | None:
    """Return path to saved markdown summary of a meeting transcript.

    Args:
        transcript (str): Raw transcript string ("<speaker>: <text>" format).
        input_filename (str): Original name of the input file (for title + output name).
        video_title (str): Title of the video (for output name).
        language (str): 'fr' or 'en'.
        model (str): Ollama model to use (e.g. 'mistral').

    Returns:
        Path: Path to the markdown summary in ./results/

    """
    my_logger.info("Parsing transcript...")
    utterances = parse_transcript(transcript)
    utterances = try_resolve_speaker_names(utterances)

    my_logger.info("Generating summary via RAG...")
    try:
        raw_summary = generate_summary(utterances, language=language, model=model)
    except Exception as e:
        my_logger.error(f"Error generating summary: {e}")
        raise

    stem = Path(input_filename).stem
    my_logger.info("Formatting markdown...")
    formatted = format_summary_markdown(
        raw_summary, filename_stem=stem, language=language,
    )
    # TODO: add sentiment to summary
    # TODO: use video_title instead of input_filename for output filename
    suffix = "résumé" if language == "fr" else "summary"
    output_filename = f"{stem} - {suffix}.md"
    output_path = Path("results") / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(formatted)

    my_logger.info(f"Summary written to {output_path}")

def summarize_transcript_with_openai(
    transcript: str, input_path: str, video_title: str, language: str,
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

    load_dotenv()  # declare your OPENAI_API_KEY in a .env file
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide concise and insightful summaries."},
                {"role": "user", "content": prompt},
            ],
        )
    except OpenAI.error.AuthenticationError:
        my_logger.exception("AuthenticationError while performing OpenAI API request", stack_info=True)
    except OpenAI.error.Timeout:
        my_logger.exception("Timeout while performing OpenAI API request", stack_info=True)
    except Exception:
        my_logger.exception("Unexpected error while performing OpenAI API request", stack_info=True)
    markdown_output = simple_format_markdown(
        video_title,
        input_path,
        response.choices[0].message.content,
        sentiment,
        language,
    )
    p = Path(f"./results/{video_title}.md")
    with p.open(mode="w", encoding="utf8") as f:
        f.write(markdown_output)
