# Simple script to summarize a YT video

## Uses

* youtube_transcript_api to fetch the transcript
* textblob to compute sentiment
* openai api to build a a summary (needs API KEY & purchase tokens)

## To do

* Process a local video or audio file
  * Change command line parsing so that it accepts a local file next to YT videos
  * If video file was provided, extract audio using ...
  * Use xxx to extract transcript from audio file
* Use yyy to improve transcript quality
* Use PyAnnote so that it recognizes speakers (a.k.a. speaker diarization) so as to be able to summarize separately their opinions / rationale from a conversation
