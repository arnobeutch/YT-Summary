# Simple script to summarize a YT video

## Uses

* youtube_transcript_api to fetch the transcript
* textblob to compute sentiment
* openai api to build a a summary (needs API KEY & purchase tokens)

## To do

* Summarize with local LLM i/o burning OPENAI_API credits (command-line switch)
* Rather than relying on YT transcripts of dubious quality, download YT audio w. yt_dlp and process locally with whisper
* Use spacy & nltk to improve transcript quality. Extend further by:
  * Integrating a Levenshtein distance on phoneme sequences
  * Prioritizing corrections based on contextual language models (e.g., transformers)
  * Filtering using part-of-speech or NER if needed
* Use PyAnnote so that it recognizes speakers (a.k.a. speaker diarization) so as to be able to summarize separately their opinions / rationale from a conversation
