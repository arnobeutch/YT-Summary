"""Output formatting utilities."""

def format_markdown(
    video_title: str,
    video_path: str,
    summary: str,
    sentiment: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 Video Summary
- Title: {video_title}
- From: {video_path}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    if language == "fr":
        return f"""
## 📺 Résumé de la vidéo
- Titre : {video_title}
- De : {video_path}
- **Sentiment :** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."
