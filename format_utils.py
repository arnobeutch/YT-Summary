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


def format_markdown_extended(
    video_title: str,
    video_url: str,
    summary: str,
    sentiment: str,
    category: str,
    keywords: str,
    transcript: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
# 📺 YouTube Video Summary
- Video title: {video_title}
- From: {video_url}

## 🎯 Theme & Summary
{summary}

## 🔑 Key Insights
- **Sentiment:** {sentiment}
- **Topic Category:** {category}
- **Keywords:** {keywords}

## 📜 Transcript Extract
{transcript[:1000]}...
"""
    if language == "fr":
        return f"""
## 📺 Résumé Vidéo YouTube
- Titre de la vidéo: {video_title}
- De: {video_url}
- **Sentiment:** {sentiment}
- **Catégorie:** {category}
- **Mots Clefs:** {keywords}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."
