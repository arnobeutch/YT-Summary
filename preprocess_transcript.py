"""Preprocess transcript text into structured utterances."""

import re

from unidecode import unidecode


def parse_transcript(transcript: str) -> list[tuple[str, str]]:
    """Return list of (speaker, text) from raw transcript string.

    Args:
        transcript (str): Raw transcript in "<speaker>: <text>" per line format.

    Returns:
        list[tuple[str, str]]: List of speaker-utterance pairs.

    """
    lines = transcript.strip().splitlines()
    utterances = []

    for line in lines:
        if ": " not in line:
            continue  # Skip malformed lines
        speaker, text = line.split(": ", 1)
        utterances.append((speaker.strip(), text.strip()))

    return utterances


def try_resolve_speaker_names(
    utterances: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return updated utterances with guessed speaker names if nominative clues exist.

    Args:
        utterances (list[tuple[str, str]]): List of (speaker_id, text) tuples.

    Returns:
        list[tuple[str, str]]: Updated list with inferred names when possible.

    """
    speaker_name_map: dict[str, str] = {}

    for speaker, text in utterances:
        if speaker in speaker_name_map:
            continue

        match = re.search(
            r"(?:merci|parlait de|comme disait|selon)\s+([A-ZÉÈÊÀÂÎÔÛa-zéèêàâîôû]+)",
            unidecode(text),
            re.IGNORECASE,
        )
        if match:
            guessed_name = match.group(1).capitalize()
            speaker_name_map[speaker] = guessed_name

    return [
        (speaker_name_map.get(speaker, speaker), text) for speaker, text in utterances
    ]
