# TODO

Planned improvements, grouped by theme. Items are unordered within each group unless they have an explicit dependency.

## Design refactors

- Extract per-source handlers from `main.py` into `_handle_url` / `_handle_media` / `_handle_text` returning a `Transcript` dataclass; shrink `main()` to a tiny orchestrator.
- Raise `TranscriptUnavailable` from `get_youtube_transcript` instead of returning `"Error: ..."` strings; drop the `"Error" in transcript` string-match in `main.py`.
- Implement the full language-selection decision tree (see README "Language selection" once written). `--language` is a preference hint, not a hard override; the summary follows the source's language. Priority: manual captions beat auto across languages.
- Promote `Settings` to the real config object — add `openai_api_key`, `openrouter_api_key`, `huggingface_token`, `llm_provider`, `llm_model`, `whisper_model_size`, `output_dir`, `downloads_dir`, `wrap_width`. Read once at startup; stop calling `Settings.from_env()` for its side effects in every module.
- Introduce a `Transcript` dataclass (`text, language, source, title, diarized, segments?`) and replace the tuple returns from `transcribe_*` functions.
- Unify transcribe entry points — single `transcribe(input_path, *, diarize, model_size)` that auto-detects audio vs. video by extension. Halves the current 4-function surface.
- Summarizer Protocol — `OpenAISummarizer`, `OpenRouterSummarizer`, `RagSummarizer` behind a shared interface; sentiment applied in the base so both backends emit it.
- Unify prompt templates — collapse the near-identical EN/FR pairs into one template + a dict of lang-varying phrases.
- Rename `RAG_SECTION_TITLES` keys from French literals (`"Sujet"`, `"Principaux enseignements"`, ...) to neutral (`topic`, `hashtags`, `takeaways`, `qa`, `decisions`, `actions`). Language becomes a presentation concern only.
- Cache the whisper model across calls in-process (module-level dict keyed by `(model_size, device)`).
- Consolidate filename sanitization — always apply `_sanitize_filename` before writing, not just for the URL branch.
- Normalize flag style — rename `--with_openai` to `--with-openai` (keep underscore as a deprecated alias).

## Features / CLI options

- `--transcript-only` short-circuits before summarization.
- `--model-size {tiny,base,small,medium,large}` (default via `Settings`). Today's code hardcodes `"small"` in `main.py` and `"base"` in `prepare_local_transcript.py` — three values for one knob.
- `--llm-provider {openai,openrouter,ollama}` + `--llm-model NAME`.
- `--output-dir PATH` (default `./results`) and `--downloads-dir PATH` (default `./downloads`).
- Smart caching: skip download if `downloads/<id>.wav` exists; skip transcription if `results/<title> transcript.txt` exists. `--force` overrides.
- Subtitle / timestamped output — emit `.srt` + `.vtt` alongside `.txt` from whisper segments (`--subtitles`).
- Batch mode — multiple positional inputs (`main.py a.mp4 b.mp4 URL`).
- API-key preflight — fail fast when `--summarize --llm-provider openai` is set but `OPENAI_API_KEY` is missing (not mid-pipeline after a 10-min transcription).
- GPU-not-detected warning when `nvidia-smi` works but `torch.cuda.is_available()` returns False.
- Two summary scenarios + autodetect (`--summary-mode {meeting,source,auto}`):
  - `meeting`: topic, hashtags, takeaways, Q&A, decisions, action items.
  - `source`: TL;DR, key takeaways, facts vs. opinion vs. speculation, alternatives / counterpoints, overall information-quality / reliability score.
  - `auto`: speaker count (from diarization) → else opinion-marker density → else `source`.
- `--summary-style {brief,detailed,bullets,prose}` orthogonal to `--summary-mode`.
- OpenRouter LLM backend — reuses the OpenAI SDK with `base_url="https://openrouter.ai/api/v1"`. `--llm-model` accepts provider-prefixed names (`minimax/minimax-2.7`, `moonshotai/kimi-k2`, `anthropic/claude-4.7-sonnet`, ...).
- Replace `youtube_transcript_api` with `yt-dlp` for captions (we already use yt-dlp for audio). yt-dlp separates manual vs auto captions cleanly, which the language-selection tree needs.

## Summary polish

- Add sentiment to RAG summaries too (currently only the OpenAI path emits it). Falls out of the Summarizer Protocol refactor.
- Unify OpenAI and RAG output format — both should emit structured sections (better Obsidian integration). Today OpenAI writes one-block markdown while RAG writes sectioned.

## RAG backend hygiene

- Replace `langchain_community.vectorstores.Chroma` with the `langchain-chroma` package. Fixes the deprecation warning and the numpy incompatibility.
- Review chunking params — `chunk_size=500, chunk_overlap=50` on diarized transcripts yields one-doc-per-utterance. Either bypass the splitter for already-segmented input, or merge utterances up to `chunk_size`.

## Known constants to expose as Settings / CLI

- `MIN_SEGMENT_DURATION = 1.5` (in `transcribe_audio_with_diarization`).
- `max_gap=1.0` in `group_speaker_segments`.
- `_WRAP_WIDTH = 80` in `main.py`.

## Deferred research (long-horizon)

- **Python 3.12+/3.13 upgrade.** Current pin `==3.11.9` because `openai-whisper`, `pyannote-audio`, and `torchaudio` block newer Python. Review quarterly. If still blocked, evaluate:
  - `faster-whisper` (SYSTRAN) — CTranslate2-based, no PyTorch, typically 4× faster on CPU, same model family. Drops PyTorch and unblocks Python 3.13.
  - `whisper.cpp` bindings (e.g. `pywhispercpp`) — ggml quantized; best fit for the packaged-binary goal below.
  - For diarization alternatives (pyannote pulls PyTorch): `simple-diarizer`, `speechbrain`, or making diarization an optional extra.
- **Better local LLM default** for the Ollama backend. Gemma 4 family (released 2026-04-02, obsoletes Gemma 3):
  - E2B (~2.3B effective via Per-Layer Embeddings) — phone-tier, 128K context.
  - **E4B** (~4.5B effective) — laptop CPU sweet spot, 128K context. Strongest candidate for the new default.
  - 26B A4B (MoE, 4B active) — 256K context, reasoning-grade. Good on 32GB+ laptops.
  - 31B dense — workstation-tier. Overkill for laptop.
  - Other candidates to benchmark: Llama 3.3 70B quantized, Qwen 2.5 (7B / 14B), Phi 3.5 Mini (3.8B).
  - Tentative laptop default: `gemma4:e4b`. Verify via Ollama tag list before wiring.
- **Portable packaged executable** (single binary, no installer). Allowed deps: whisper (or faster-whisper / whisper.cpp), ffmpeg, yt-dlp. Approaches:
  - PyInstaller `--onefile` with a "minimal" build flavor (drop pyannote + openai + langchain + chromadb; keep whisper + yt-dlp + ffmpeg).
  - Nuitka — smaller binaries, longer build, better runtime. Head-to-head comparison.
  - whisper.cpp + yt-dlp binary + tiny Go/Python wrapper — smallest footprint, biggest implementation cost.
  - Feature gating via optional extras (`pip install yt-summary[summarize,diarize]`).
- **Feature-flag driven builds** (`[summarize]`, `[diarize]`, `[openrouter]` extras). Enables the packaged-binary plan cheaply and makes contributor onboarding lighter.
- **Context-file support** for source-summary mode (`--context-file path.txt` appends extra material to the LLM prompt).
- **Claim-tagging output** in source-mode — tag each claim `{factual | opinion | speculation}` with confidence.
- **Dry-run mode** (`--dry-run`) — print the planned pipeline (caption source, fallback need, model, output dir) without doing work.
- **"Auto" model-size** — pick `tiny`/`base`/`small`/`medium` from `(hardware, audio duration)`.
- **Chapter / TOC extraction** from yt-dlp's `chapters` field, with `?t=<ss>` deep-links.
- **`langdetect` replacement survey** — only if `langdetect` proves unreliable on short transcripts; candidate: `lingua-py`.
- **Transcript cleanup** — spaCy / nltk cleanup, Levenshtein on phoneme sequences, NER-filtered corrections.
- **YouTube audio diarization** — currently diarization only runs on local media; extend to YT audio (uses the downloaded `.wav` we already have).
- **Speaker identification at summarization time** for YT transcripts (currently only speaker IDs are shown).

## Completed

- 2026-04-22 — Unit test suite added (158 tests across all tiers; opt-in `integration` marker for whisper / pyannote).
- 2026-04-22 — App modules + tests brought up to ruff-ALL + pyright-strict clean. `just all` runs green.
- 2026-04-22 — YouTube captionless-video fallback (yt-dlp audio download + whisper transcription) when no captions are available; handles `TranscriptsDisabled`, empty XML payloads, and short-link / embed / shorts URL forms.
- 2026-04-22 — Transcripts soft-wrapped at 80 chars without breaking words when written to disk.
