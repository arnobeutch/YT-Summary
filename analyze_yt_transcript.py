"""Analyze YouTube transcript with OpenAI API."""

# see: https://pypi.org/project/openai/
from openai import OpenAI

import my_constants
from main import log


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
