"""Markdown formatter for Obsidian-compatible meeting summaries."""

import re

from scriber.constants import RAG_SECTION_TITLES
from scriber.logger import my_logger


def extract_sections(summary: str, language: str) -> dict[str, str]:
    """Return mapping of section title -> content from raw text.

    Args:
        summary (str): Raw summary text from the model.
        language (str): 'fr' or 'en'

    Returns:
        dict[str, str]: Extracted sections keyed by logical name.

    """
    titles = list(RAG_SECTION_TITLES[language].keys())
    joined_titles = "|".join(re.escape(t) for t in titles)
    # Bound each section at the next known title or end-of-string. Using the
    # title alternation in the lookahead (instead of `\n\S+?:`) correctly
    # handles multi-word titles like "Principaux enseignements".
    pattern = rf"({joined_titles})\s*:\s*(.*?)(?=\n(?:{joined_titles})\s*:|\Z)"

    matches = re.findall(pattern, summary, flags=re.DOTALL)
    return {title.strip(): body.strip() for title, body in matches}


def clean_section(text: str, language: str) -> str:
    """Return cleaned section content, or default if empty.

    Args:
        text (str): Section body.
        language (str): 'fr' or 'en'

    Returns:
        str: Cleaned section body or default filler.

    """
    cleaned = text.strip()
    default = "Aucune" if language == "fr" else "None"
    if not cleaned or cleaned.lower() in {"none", "aucune", "n/a"}:
        return default
    return cleaned


def format_summary_markdown(raw_summary: str, filename_stem: str, language: str) -> str:
    """Return Obsidian-ready markdown summary from raw model output.

    Args:
        raw_summary (str): Summary output from the RAG engine.
        filename_stem (str): Name (stem only) of the input file to use as title.
        language (str): 'fr' or 'en'

    Returns:
        str: Complete markdown string.

    """
    my_logger.debug("Formatting summary markdown...")
    section_headers = RAG_SECTION_TITLES[language]
    sections = extract_sections(raw_summary, language)

    completed_sections = {
        title: clean_section(sections.get(title, ""), language) for title in section_headers
    }

    title_line = (
        f"# Résumé de la réunion — {filename_stem}"
        if language == "fr"
        else f"# Meeting Summary — {filename_stem}"
    )

    lines = [title_line, ""]
    for key, header in section_headers.items():
        lines.append(header)
        lines.append(completed_sections[key])
        lines.append("")

    return "\n".join(lines)


def simple_format_markdown(
    video_title: str,
    video_path: str,
    summary: str,
    sentiment: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 Video Summary
- Title: {video_title}
- From: {video_path}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    if language == "fr":
        return f"""
## 📺 Résumé de la vidéo
- Titre : {video_title}
- De : {video_path}
- **Sentiment :** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."
