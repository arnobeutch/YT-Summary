# TODO

Planned improvements, grouped by priority. Items are unordered within each group unless they have an explicit dependency.

## Near-term

- **Fix RAG chunking params.** `chunk_size=500, chunk_overlap=50` on diarized transcripts yields one-doc-per-utterance. Either bypass the splitter for already-segmented input, or merge utterances up to `chunk_size`. This is a correctness bug, not polish — retrieval degrades to keyword lookup on single turns.
- **Unify OpenAI and RAG output format.** Both should emit structured sections for clean Obsidian integration. Today OpenAI writes one-block markdown while RAG writes sectioned.
- **Unify EN/FR prompt templates.** Collapse the near-identical pairs into one template + a dict of language-varying phrases.
- **Rename `RAG_SECTION_TITLES` keys** from French literals (`"Sujet"`, `"Principaux enseignements"`, …) to neutral (`topic`, `hashtags`, `takeaways`, `qa`, `decisions`, `actions`). Language becomes a presentation concern only.
- **Chapter / TOC extraction** from yt-dlp's `chapters` field, with `?t=<ss>` deep-links.
- **YouTube audio diarization.** Currently diarization only runs on local media; extend to YT audio (uses the downloaded `.wav` we already have).
- **`--context-file path.txt`** for source-summary mode (appends extra material to the LLM prompt).
- **Progress bar** during whisper transcription + download steps. Silent minutes are a UX problem.
- **Streaming LLM output.** Long transcripts take minutes and the CLI is silent; `stream=True` on the OpenAI SDK should flush tokens as they arrive.
- **Batch-mode resilience.** `--continue-on-error` so one yt-dlp rate-limit / geo-block / age-gate doesn't kill the whole run. Post-run summary table (✓/✗ per input).

## Medium-term

- **Eval harness.** Even 10 reference `(transcript, expected-sections)` pairs with a cosine-similarity smoke test would make LLM provider/model swaps defensible. Currently there's no way to know if a new default degrades output quality — this is a prerequisite for any default-model swap.
- **Feature-flag driven builds** (`[summarize]`, `[diarize]`, `[openrouter]` extras). Enables the packaged-binary plan cheaply and makes contributor onboarding lighter.
- **"Auto" model-size.** Pick `tiny`/`base`/`small`/`medium` from `(hardware, audio duration)`.
- **Speaker identification at summarization time** for YT transcripts (currently only speaker IDs are shown).

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
  - Tentative laptop default: `gemma4:e4b`. Verify via Ollama tag list before wiring. **Prerequisite:** eval harness (above).
- **Portable packaged executable** (single binary, no installer). Allowed deps: whisper (or faster-whisper / whisper.cpp), ffmpeg, yt-dlp. **Prerequisite:** feature-flag extras (above). Approaches:
  - PyInstaller `--onefile` with a "minimal" build flavor (drop pyannote + openai + langchain + chromadb; keep whisper + yt-dlp + ffmpeg).
  - Nuitka — smaller binaries, longer build, better runtime. Head-to-head comparison.
  - whisper.cpp + yt-dlp binary + tiny Go/Python wrapper — smallest footprint, biggest implementation cost.
- **Claim-tagging output** in source-mode — tag each claim `{factual | opinion | speculation}` with confidence.
- **Transcript cleanup** — spaCy / nltk cleanup, Levenshtein on phoneme sequences, NER-filtered corrections.
- **`--summary-style {brief,detailed,bullets,prose}`** orthogonal to `--summary-mode`. 4× the prompt surface to maintain; value unclear — keep deferred or drop.

## Dropped / indefinite defer

- **`langdetect` replacement survey.** Candidate was `lingua-py`. No evidence of failures on short transcripts — revisit only if reliability becomes a real problem.

## Completed

- 2026-04-24 — **CLI split into `scriber transcribe` / `scriber summarize` subcommands; project renamed `yt-summary` → `scriber`.** `--summarize` and `--transcript-only` flags removed.
- 2026-04-24 — Sentiment added to RAG summaries (parity with OpenAI/OpenRouter backends).
- 2026-04-22 — Unit test suite added (271 tests across all tiers; opt-in `integration` marker for whisper / pyannote).
- 2026-04-22 — App modules + tests brought up to ruff-ALL + pyright-strict clean. `just all` runs green.
- 2026-04-22 — YouTube captionless-video fallback (yt-dlp audio download + whisper transcription) when no captions are available; handles `TranscriptsDisabled`, empty XML payloads, and short-link / embed / shorts URL forms.
- 2026-04-22 — Transcripts soft-wrapped at 80 chars without breaking words when written to disk.
- 2026-04-24 — CLI flag `--with-openai` normalized (hyphen canonical; underscore kept as deprecated alias).
- 2026-04-24 — `Settings` promoted to full config dataclass (all keys, single startup read, `.env` + env override).
- 2026-04-24 — `--model-size`, `--llm-provider`, `--llm-model`, `--output-dir`, `--downloads-dir` CLI flags.
- 2026-04-24 — `TranscriptUnavailableError` exception replaces string-typed sentinel returns.
- 2026-04-24 — `youtube_transcript_api` replaced with yt-dlp for captions (dep dropped).
- 2026-04-24 — Per-source handlers extracted from `main.py` (`handlers.py`); `Transcript` dataclass introduced.
- 2026-04-24 — Full language-selection ladder (manual > auto; requested > en > other; see README).
- 2026-04-24 — Pluggable Summarizer Protocol + OpenRouter backend + sentiment everywhere.
- 2026-04-24 — `--summary-mode {meeting,source,auto}` with autodetect heuristic.
- 2026-04-24 — Smart caching: skip download/transcription when outputs exist; `--force` bypasses.
- 2026-04-24 — `--subtitles` (.srt + .vtt from whisper segments); `--transcript-only` (later superseded by the `transcribe` subcommand).
- 2026-04-24 — Whisper model cache in-process (`_MODEL_CACHE` keyed by model+device); `MIN_SEGMENT_DURATION` and `_MAX_SPEAKER_GAP` promoted to module constants; dead wrappers removed.
- 2026-04-24 — Batch mode (`nargs="+"` multiple inputs), `--dry-run`, GPU-not-detected warning, API-key preflight.
- 2026-04-24 — `langchain_community.vectorstores.Chroma` → `langchain-chroma`.
