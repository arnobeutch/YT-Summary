"""Output formatting utilities."""

def format_markdown(
    video_title: str,
    video_url: str,
    summary: str,
    sentiment: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 YouTube Video Summary
- Video title: {video_title}
- From: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    if language == "fr":
        return f"""
## 📺 Résumé Vidéo YouTube
- Titre de la vidéo: {video_title}
- De: {video_url}
- **Sentiment:** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."
