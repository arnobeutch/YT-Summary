"""Language-selection helpers for the YT-Summary pipeline.

The summary language tracks the source's language (including titles); the
user-supplied ``--language`` is a *preference hint*, not a hard override.

See README "Language selection" for the full spec.
"""

from __future__ import annotations


def derive_summary_language(track_lang: str, requested_lang: str | None) -> str:
    """Decide the summary language from the chosen caption / transcription language.

    Rules:
      * caption is the user's requested language → summary in that language
      * caption is English → summary in English
      * everything else → summary in English (forced; we instruct the LLM
        to translate)

    """
    if requested_lang and track_lang == requested_lang:
        return requested_lang
    if track_lang == "en":
        return "en"
    return "en"


def derive_whisper_summary_language(detected_lang: str, requested_lang: str | None) -> str:
    """Summary language when whisper transcribes (no captions available).

    Rules:
      * ``--language`` specified → whisper was forced to that language; summary
        in that language too.
      * Auto-detected ∈ {en, fr} → summary in detected language.
      * Otherwise → summary in English (forced).

    """
    if requested_lang:
        return requested_lang
    if detected_lang in {"en", "fr"}:
        return detected_lang
    return "en"
