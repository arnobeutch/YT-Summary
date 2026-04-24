# YT-Summary — Claude Reference

## Project

CLI tool to summarize YouTube videos, local audio/video files, or pre-existing text transcripts. Pipeline: fetch/transcribe → optional diarization → sentiment/polarity → LLM summary (OpenAI API or local RAG via langchain + Ollama).

## Tech stack

- Python `==3.11.9` (pinned — `openai-whisper`, `pyannote-audio`, `torchaudio` block 3.12+).
- `uv` for dependency/env management.
- `ruff` lint (`select = ["ALL"]`; ignores in `pyproject.toml`, `target-version = "py311"`).
- `pyright` strict mode (`[tool.pyright]` in `pyproject.toml`, `pythonVersion = "3.11"`) — matches VSCode Pylance.
- `.claude/hooks/python-quality.sh` runs after every `Edit`/`Write` on a `.py` file. If it exits 2, read the stderr diagnostics and fix before continuing. See `.claude/rules/python_strict.md` for the recurring traps.

## Layout

`src/yt_summary/` — the installable package (`uv run yt-summary` invokes `yt_summary.main:main`).

## File map

| File | Purpose |
| --- | --- |
| `src/yt_summary/main.py` | Entry point. Parses args, initializes logger, loads settings, loops over `args.input_path` (one or more), dispatches per-path to the right handler, optionally summarizes. Also: preflight LLM key check, GPU warning, `--dry-run` report. |
| `src/yt_summary/handlers.py` | `handle_url` / `handle_media` / `handle_text` + `write_transcript_file(subtitles=bool)` + `summarize`. The actual orchestration. |
| `src/yt_summary/model.py` | Shared dataclasses (currently just `Transcript`) — kept import-cycle-free. |
| `src/yt_summary/formatting.py` | `sanitize_filename`, `wrap_transcript` — text helpers shared by handlers. |
| `src/yt_summary/language.py` | `derive_summary_language` + `derive_whisper_summary_language`. Pure functions implementing the language-selection ladder (see README). |
| `src/yt_summary/subtitles.py` | `write_srt` / `write_vtt` from whisper-style segment dicts. Used by `--subtitles`. |
| `src/yt_summary/summarizers/` | Pluggable summarization backends: `OpenAISummarizer`, `OpenRouterSummarizer`, `RagSummarizer` behind a shared `Summarizer` Protocol; `make_summarizer(settings)` factory; `MissingAPIKeyError` for preflight; `analyze_sentiment`. `modes.py` carries the `meeting`/`source`/`auto` prompts + autodetect heuristic. `markdown.py` formats output; `engine.py` is the RAG chain. |
| `src/yt_summary/parser.py` | argparse: `input_path` (`nargs="+"`, one or more URLs/paths), all flags. `classify_input(path)` helper returns `{is_url, is_file, is_media_file, is_text_file}` per path (called by `main.py` per-input and eagerly in `parse_args()` for validation). |
| `src/yt_summary/logger.py` | Custom logging: `ColorFormatter`, `MyJSONFormatter`, `NonErrorFilter`, `install_excepthook`. |
| `src/yt_summary/logger_config.yaml` | dictConfig YAML. Handlers: stdout (non-errors), stderr (WARNING+), rotating `logs/yt-summary.log`. |
| `src/yt_summary/constants.py` | Prompts (`OPENAI_PROMPT_EN/FR`, `RAG_FRENCH/ENGLISH_PROMPT`, `RAG_SECTION_TITLES`) + polarity thresholds. |
| `src/yt_summary/settings.py` | Frozen `Settings` dataclass + stdlib `.env` loader (replaces the old `python-dotenv` dep). |
| `src/yt_summary/transcription/youtube_captions.py` | yt-dlp-backed YouTube caption fetch. Picks manual > auto across `["fr", "en"]`. Raises `TranscriptUnavailableError` on failure. |
| `src/yt_summary/transcription/youtube_audio.py` | yt-dlp-based audio download + video-id extraction + title metadata (used for the captionless-video fallback path). Smart-caches: returns existing `.wav` unless `force=True`. |
| `src/yt_summary/transcription/local.py` | ffmpeg → whisper transcription, optional pyannote diarization. Module-level `_MODEL_CACHE` avoids reloading whisper across calls. `transcribe_audio_full` is the primary entry point (returns text + lang + segments). Module constants: `MIN_SEGMENT_DURATION`, `_MAX_SPEAKER_GAP`. |
| `src/yt_summary/transcription/preprocess.py` | Cleanup + speaker-name heuristics. |

## Dev workflow

- `uv add`, `uv run`, `uv sync` — never `pip`/`venv` directly.
- `just lint`, `just typecheck`, `just test`, `just all`.
- `pre-commit` runs ruff + pyright + pytest on `git commit` once installed (`uv run pre-commit install`).
- `pytest -m integration` runs opt-in ML tests (whisper / pyannote). They need a fixture at `tests/integration/data/hello.wav` (see the test module's docstring) and download whisper models on first run. Skipped by default.
- Do not auto-commit.
- Keep changes minimal and focused.

## Known state

- **App modules + tests are ruff-ALL + pyright-strict clean.** `just all` runs green (268 tests, 2 deselected integration). Boundary with untyped ML deps (`whisper`, `pyannote`, `torchaudio`, `ffmpeg`, `yt-dlp`, `langchain`, `chromadb`) is handled via file-level `# pyright: reportUnknown... = false` headers in `transcription/local.py` and `transcription/youtube_audio.py`, and explicit `cast(Any, ...)` at call sites elsewhere. Apply `.claude/rules/python_strict.md` patterns when extending.
- `results/`, `downloads/`, and `chroma_db/` are runtime outputs (gitignored).
- `.env` holds `OPENAI_API_KEY`, optionally `HUGGINGFACE_TOKEN` (for diarization), and any `LOG_LEVEL` override.
- Improvement plan lives at `/home/mprz/.claude/plans/ok-now-that-we-inherited-pascal.md`. In-flight work is tracked there; `TODO.md` is the grooming backlog.
