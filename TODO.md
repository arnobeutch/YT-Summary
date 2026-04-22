# TODO

- Improve transcript quality (spaCy / nltk cleanup, Levenshtein on phoneme sequences, NER-filtered corrections).
- Unify OpenAI and RAG prompts.
- Meeting summary vs. informational summary modes.
- Detect language when summarizing from an existing transcript.
- Bring the six app modules (`prepare_*`, `preprocess_transcript`, `summarize_transcript`, `q_and_a_engine`, `markdown_writer`) up to ruff-ALL + pyright-strict.
- Add proper unit tests (logger fixtures are already set up in `tests/conftest.py`).
