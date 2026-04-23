# YT-Summary

Summarize YouTube videos, local audio/video files, or existing text transcripts.

## Features

- `youtube_transcript_api` to fetch YouTube captions (en / fr).
- `whisper` transcription for local media (`ffmpeg-python` for audio extraction).
- Optional speaker diarization via `pyannote-audio`.
- `textblob` sentiment/polarity.
- OpenAI API summary (`--with_openai`) or local RAG summary via `langchain` + `langchain-ollama` + `chromadb` (default, uses `mistral`).
- Markdown-formatted summary output.

## Usage

```bash
uv run main.py <youtube_url | path/to/file> [options]
```

Examples:

```bash
uv run main.py https://www.youtube.com/watch?v=VIDEO_ID --summarize --with-openai
uv run main.py ./my_meeting.mp4 --diarize --summarize
uv run main.py ./existing_transcript.txt --summarize
```

### Options

| Flag | Description |
| --- | --- |
| `-l`, `--language` | `en` or `fr` (default: `en`). Ignored for local media (auto-detected). |
| `--diarize` | Identify speakers when transcribing local media (default: False). |
| `-s`, `--summarize` | Produce a summary (default: False). |
| `--with-openai` | Summarize via OpenAI API instead of local RAG (default: False). `--with_openai` still works as a legacy alias. |
| `-d`, `--debug` | Enable DEBUG-level logging (default: False). |

## Configuration

Runtime settings live in `my_settings.py` (`Settings.from_env()` reads `.env` + `os.environ`; shell env wins over `.env`). Copy `.env.example` to `.env` and adjust. `.env` is gitignored.

| Env var | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. `-d`/`--debug` forces DEBUG. |
| `OPENAI_API_KEY` | — | Required for `--with-openai` (or `--llm-provider openai`). |
| `OPENROUTER_API_KEY` | — | Required for `--llm-provider openrouter`. |
| `HUGGINGFACE_TOKEN` | — | Required for `--diarize` (pyannote speaker-diarization-3.1 + voice-activity-detection gated models). |
| `LLM_PROVIDER` | `openai` | One of `openai`, `openrouter`, `ollama` (CLI flag overrides). |
| `LLM_MODEL` | provider default | E.g. `gpt-4o`, `anthropic/claude-4.7-sonnet`, `mistral` (CLI flag overrides). |
| `OPENAI_MODEL` | `gpt-4o` | Model for the OpenAI provider. |
| `OLLAMA_MODEL` | `mistral` | Model for the local Ollama/RAG provider. |
| `WHISPER_MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`, `large`. |
| `OUTPUT_DIR` | `results` | Where transcripts and summaries land. |
| `DOWNLOADS_DIR` | `downloads` | Where downloaded YT audio is cached. |
| `WRAP_WIDTH` | `80` | Soft-wrap width for non-diarized transcripts (words are never split). |

## Logging

Configured via `logger_config.yaml`. Handlers:

| Handler | Stream | Level |
| --- | --- | --- |
| stdout | stdout | INFO/DEBUG (non-errors) |
| stderr | stderr | WARNING+ |
| file | `logs/yt-summary.log` | DEBUG+ (rotating) |

Uncomment `- json_file` under `root.handlers` in `logger_config.yaml` to also emit structured JSON to `logs/yt-summary.jsonl`. Uncaught exceptions are routed through the logger via `install_excepthook()`.

## Dev workflow

All quality commands run through `just`:

| Command | Does |
| --- | --- |
| `just` | List recipes |
| `just lint` | `ruff check` + `ruff format --check` |
| `just format` | `ruff format` + `ruff check --fix` |
| `just typecheck` | `uv run pyright` |
| `just test` | `uv run pytest` |
| `just all` | `lint` + `typecheck` + `test` |

Install the pre-commit gate once per clone:

```bash
uv run pre-commit install
```

`git commit` will then run ruff + pyright + pytest on staged Python files.

## Roadmap

Grooming backlog with completed items: [TODO.md](TODO.md).
