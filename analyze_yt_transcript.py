"""Analyze video transcripts (YT, local) with OpenAI API."""

from dotenv import load_dotenv
from openai import OpenAI  # see: https://pypi.org/project/openai/

import my_constants
from my_logger import my_logger


def summarize_transcript(transcript: str, language: str) -> str | None:
    """Summarize the video transcript into theme, key ideas, and takeaways."""
    if language == "fr":
        prompt = my_constants.SUMMARIZE_PROMPT_FR + f"{transcript}"
    elif language == "en":
        prompt = my_constants.SUMMARIZE_PROMPT_EN + f"{transcript}"
    else:
        return "Error: summarizer language not supported."

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
        return response.choices[0].message.content
    except OpenAI.error.AuthenticationError:
        my_logger.exception("AuthenticationError while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request was not authorized\nCheck or set your API key"
    except OpenAI.error.Timeout:
        my_logger.exception("Timeout while performing OpenAI API request", stack_info=True)
        return "Error: OpenAI API request timed out"
    except Exception:
        my_logger.exception("Unexpected error while performing OpenAI API request", stack_info=True)
        return "Error: unexpected error while performing OpenAI API request"
