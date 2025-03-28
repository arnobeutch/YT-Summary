"""Script constants."""

SUMMARIZE_PROMPT_EN = """
You are an expert summarizer. Given the following YouTube video transcript, provide:

1. **Theme of the video**
2. **Key ideas discussed**
3. **Main takeaways**
4. **Top 5 keywords**
5. **3 main topic categories**

Provide a structured summary.

Transcript:
"""
SUMMARIZE_PROMPT_FR = """
Vous êtes un expert en résumé. Étant donné la transcription vidéo YouTube suivante, fournissez:

1. **Thème de la vidéo**
2. **Idées clés discutées**
3. **Principal à retenir**
4. **5 mots-clés principaux**
5. **3 catégories principales**

Fournir un résumé structuré.

Transcription:
"""
POLARITY_POSITIVE_THRESHOLD = 0.2
POLARITY_NEGATIVE_THRESHOLD = -0.2
