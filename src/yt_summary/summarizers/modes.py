"""Summary modes (meeting / source / auto-detect) and their prompt templates."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from yt_summary.model import Transcript

SummaryMode = Literal["meeting", "source", "auto"]
ResolvedMode = Literal["meeting", "source"]

# --- Meeting mode ----------------------------------------------------------

MEETING_PROMPT_EN = """\
You are an expert summarizer of a multi-speaker meeting. Given the
transcript below, produce a structured summary in English with these
sections (use the exact headers, in this order):

Topic: <one-line meeting topic>
Hashtags: <5-8 relevant tags on one line>
Main takeaways:
- <bullet> (attribute to the speaker who expressed it when possible)
Questions / Answers:
- <Q (asker) — A (answerer)>
Decisions:
- <bullet>
Action items:
- <action> (owner)

Transcript:
"""

MEETING_PROMPT_FR = """\
Vous êtes un expert en résumé de réunion à plusieurs intervenants.
À partir de la transcription ci-dessous, produisez un résumé structuré en
français avec ces sections (utilisez exactement ces en-têtes, dans cet
ordre) :

Sujet : <thème de la réunion en une ligne>
Hashtags : <5 à 8 hashtags pertinents sur une ligne>
Principaux enseignements :
- <puce> (attribuez la prise de parole quand possible)
Questions / Réponses :
- <Q (auteur) — R (répondant)>
Décisions :
- <puce>
Actions à suivre :
- <action> (responsable)

Transcription :
"""

# --- Source mode -----------------------------------------------------------

SOURCE_PROMPT_EN = """\
You are an expert critical summarizer. The transcript below is from a
single source (lecture, interview, article reading, commentary). Produce
a structured summary in English with these sections (use the exact
headers, in this order):

TL;DR: <2-3 sentences>
Key takeaways:
- <bullet>
Facts:
- <claim> — supported by [observation/citation/data referenced in the source]
Opinions:
- <claim> — speaker/author's opinion (no external evidence offered)
Speculation / unverified:
- <claim> — speaker speculates or asserts without support
Counterpoints / alternatives:
- <alternative perspective the source did not address>
Information quality / reliability: <one short paragraph rating the
source's overall reliability — citations, evidence quality, neutrality,
acknowledged uncertainty>

Transcript:
"""

SOURCE_PROMPT_FR = """\
Vous êtes un expert en analyse critique. La transcription ci-dessous
provient d'une source unique (cours, interview, lecture d'article,
commentaire). Produisez un résumé structuré en français avec ces
sections (utilisez exactement ces en-têtes, dans cet ordre) :

TL;DR : <2 à 3 phrases>
Points clés :
- <puce>
Faits :
- <affirmation> — étayée par [observation/citation/donnée mentionnée]
Opinions :
- <affirmation> — opinion de l'auteur (sans preuve externe avancée)
Spéculations / non vérifié :
- <affirmation> — l'auteur spécule ou affirme sans étayer
Contrepoints / alternatives :
- <perspective alternative non abordée par la source>
Qualité de l'information / fiabilité : <court paragraphe évaluant la
fiabilité globale — citations, qualité des preuves, neutralité,
incertitudes reconnues>

Transcription :
"""


def get_prompt(mode: ResolvedMode, language: str) -> str:
    """Return the prompt template for ``(mode, language)``.

    Raises ``ValueError`` for unsupported language.
    """
    if language not in {"en", "fr"}:
        err_msg = f"Summarizer language not supported: {language!r}"
        raise ValueError(err_msg)
    if mode == "meeting":
        return MEETING_PROMPT_EN if language == "en" else MEETING_PROMPT_FR
    return SOURCE_PROMPT_EN if language == "en" else SOURCE_PROMPT_FR


# --- Auto-detect heuristic -------------------------------------------------

# Words/phrases that signal "I'm sharing my view" rather than reporting facts.
_OPINION_MARKERS = re.compile(
    r"\b("
    r"i think|i believe|in my opinion|i feel|"
    r"je pense|je crois|à mon avis|selon moi|d'après moi|"
    r"probably|maybe|perhaps|likely|"
    r"peut-être|probablement|sans doute"
    r")\b",
    flags=re.IGNORECASE,
)

# A diarized line looks like ``SPEAKER_00: text`` or ``Alice: text``.
_DIARIZED_LINE = re.compile(r"^\s*[A-Z][\w\s.-]{0,30}:\s+\S")


def _count_distinct_speakers(text: str) -> int:
    speakers: set[str] = set()
    for line in text.splitlines():
        if _DIARIZED_LINE.match(line):
            speakers.add(line.split(":", 1)[0].strip())
    return len(speakers)


def detect_mode(transcript: Transcript) -> ResolvedMode:
    """Pick ``meeting`` or ``source`` based on transcript shape.

    Order:
      1. Diarized output with 2+ distinct speakers → ``meeting``.
      2. ``opinion`` density above 1 marker per 1000 words → ``source``.
      3. Default → ``source``.

    """
    if transcript.diarized and _count_distinct_speakers(transcript.text) >= 2:
        return "meeting"

    word_count = max(1, len(transcript.text.split()))
    opinion_hits = len(_OPINION_MARKERS.findall(transcript.text))
    if opinion_hits and (opinion_hits / word_count) * 1000 >= 1.0:
        return "source"

    return "source"


def resolve_mode(requested: SummaryMode, transcript: Transcript) -> ResolvedMode:
    """``auto`` triggers ``detect_mode``; otherwise the user choice wins."""
    if requested == "auto":
        return detect_mode(transcript)
    return requested
