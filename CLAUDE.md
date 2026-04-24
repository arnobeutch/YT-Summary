# YT-Summary — Claude Reference

## Project

CLI tool to summarize YouTube videos, local audio/video files, or pre-existing text transcripts. Pipeline: fetch/transcribe → optional diarization → sentiment/polarity → LLM summary (OpenAI API or local RAG via langchain + Ollama).

## Tech stack

- Python `==3.11.9` (pinned — `openai-whisper`, `pyannote-audio`, `torchaudio` block 3.12+).
- `uv` for dependency/env management.
- `ruff` lint (`select = ["ALL"]`; ignores in `pyproject.toml`, `target-version = "py311"`).
- `pyright` strict mode (`[tool.pyright]` in `pyproject.toml`, `pythonVersion = "3.11"`) — matches VSCode Pylance.
- `.claude/hooks/python-quality.sh` runs after every `Edit`/`Write` on a `.py` file. If it exits 2, read the stderr diagnostics and fix before continuing. See `.claude/rules/python_strict.md` for the recurring traps.

## File map

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Parses args, initializes logger, loads settings, dispatches to a handler. ~30 lines. |
| `handlers.py` | `Transcript` dataclass + `handle_url` / `handle_media` / `handle_text` + `write_transcript_file` + `summarize`. The actual orchestration. |
| `formatting.py` | `sanitize_filename`, `wrap_transcript` — text helpers shared by handlers. |
| `my_parser.py` | argparse: `input_path` + `--language`, `--diarize`, `--summarize`, `--with_openai`, `-d/--debug`. Detects URL vs. media vs. text file. |
| `my_logger.py` | Custom logging: `ColorFormatter`, `MyJSONFormatter`, `NonErrorFilter`, `install_excepthook`. |
| `logger_config.yaml` | dictConfig YAML. Handlers: stdout (non-errors), stderr (WARNING+), rotating `logs/yt-summary.log`. |
| `my_constants.py` | Prompts (`OPENAI_PROMPT_EN/FR`, `RAG_FRENCH/ENGLISH_PROMPT`, `RAG_SECTION_TITLES`) + polarity thresholds. |
| `my_settings.py` | Frozen `Settings` dataclass + stdlib `.env` loader (replaces the old `python-dotenv` dep). |
| `prepare_yt_transcript.py` | yt-dlp-backed YouTube caption fetch. Picks manual > auto across `["fr", "en"]`. Raises `TranscriptUnavailableError` on failure. |
| `prepare_yt_audio.py` | yt-dlp-based audio download + video-id extraction + title metadata (used for the captionless-video fallback path). |
| `prepare_local_transcript.py` | ffmpeg → whisper transcription, optional pyannote diarization. |
| `preprocess_transcript.py` | Cleanup + speaker-name heuristics. |
| `summarize_transcript.py` | OpenAI and RAG (langchain + Ollama + ChromaDB) summarizers. |
| `q_and_a_engine.py` | RAG chain plumbing. |
| `markdown_writer.py` | Format summary output as markdown. |

## Dev workflow

- `uv add`, `uv run`, `uv sync` — never `pip`/`venv` directly.
- `just lint`, `just typecheck`, `just test`, `just all`.
- `pre-commit` runs ruff + pyright + pytest on `git commit` once installed (`uv run pre-commit install`).
- `pytest -m integration` runs opt-in ML tests (whisper / pyannote). They need a fixture at `tests/integration/data/hello.wav` (see the test module's docstring) and download whisper models on first run. Skipped by default.
- Do not auto-commit.
- Keep changes minimal and focused.

## Known state

- **App modules + tests are ruff-ALL + pyright-strict clean.** `just all` runs green (158 tests, 2 deselected integration). Boundary with untyped ML deps (`whisper`, `pyannote`, `torchaudio`, `ffmpeg`, `yt-dlp`, `langchain`, `chromadb`) is handled via file-level `# pyright: reportUnknown... = false` headers in `prepare_local_transcript.py` and `prepare_yt_audio.py`, and explicit `cast(Any, ...)` at call sites elsewhere. Apply `.claude/rules/python_strict.md` patterns when extending.
- `results/`, `downloads/`, and `chroma_db/` are runtime outputs (gitignored).
- `.env` holds `OPENAI_API_KEY`, optionally `HUGGINGFACE_TOKEN` (for diarization), and any `LOG_LEVEL` override.
- Improvement plan lives at `/home/mprz/.claude/plans/ok-now-that-we-inherited-pascal.md`. In-flight work is tracked there; `TODO.md` is the grooming backlog.
