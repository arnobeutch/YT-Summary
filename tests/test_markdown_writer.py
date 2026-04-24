"""Tests for markdown_writer.

Note: ``extract_sections`` keys are always the French labels from
``RAG_SECTION_TITLES`` regardless of the ``language`` arg — the RAG prompts
currently emit those French labels for both languages.
"""

from yt_summary.summarizers.markdown import (
    clean_section,
    extract_sections,
    format_summary_markdown,
    simple_format_markdown,
)

FR_SAMPLE = """
Sujet: Lancement produit
Hashtags: #produit #client
Principaux enseignements: Ça avance bien.
Questions / Réponses: Rien.
Décisions: Lancement en juillet.
Actions à suivre: Alice écrit la spec.
"""


class TestExtractSections:
    def test_extracts_all_sections(self) -> None:
        result = extract_sections(FR_SAMPLE, "fr")
        assert "Lancement produit" in result["Sujet"]
        assert "#produit" in result["Hashtags"]
        assert "avance bien" in result["Principaux enseignements"]
        assert "Rien" in result["Questions / Réponses"]
        assert "Lancement en juillet" in result["Décisions"]
        assert "Alice écrit" in result["Actions à suivre"]

    def test_language_does_not_affect_matched_keys(self) -> None:
        en = extract_sections(FR_SAMPLE, "en")
        fr = extract_sections(FR_SAMPLE, "fr")
        assert set(en.keys()) == set(fr.keys())

    def test_missing_section_absent_from_dict(self) -> None:
        partial = "Sujet: Seul sujet ici.\nHashtags: #x\n"
        result = extract_sections(partial, "fr")
        assert "Seul sujet" in result.get("Sujet", "")
        assert "#x" in result.get("Hashtags", "")
        assert "Décisions" not in result

    def test_empty_summary(self) -> None:
        assert extract_sections("", "fr") == {}


class TestCleanSection:
    def test_empty_returns_default_en(self) -> None:
        assert clean_section("", "en") == "None"

    def test_empty_returns_default_fr(self) -> None:
        assert clean_section("", "fr") == "Aucune"

    def test_none_literal(self) -> None:
        assert clean_section("none", "en") == "None"
        assert clean_section("NONE", "en") == "None"

    def test_aucune_literal(self) -> None:
        assert clean_section("aucune", "fr") == "Aucune"

    def test_na_literal(self) -> None:
        assert clean_section("n/a", "en") == "None"
        assert clean_section("N/A", "fr") == "Aucune"

    def test_normal_text_preserved(self) -> None:
        assert clean_section("hello world", "en") == "hello world"

    def test_strips_surrounding_whitespace(self) -> None:
        assert clean_section("  hello  ", "en") == "hello"


class TestFormatSummaryMarkdown:
    def test_english_output(self) -> None:
        out = format_summary_markdown(FR_SAMPLE, "video-1", "en")
        assert out.startswith("# Meeting Summary — video-1")
        assert "## Meeting Topic" in out
        assert "Lancement produit" in out
        assert "## Decisions" in out
        assert "## Action Items" in out

    def test_french_output(self) -> None:
        out = format_summary_markdown(FR_SAMPLE, "video-1", "fr")
        assert out.startswith("# Résumé de la réunion — video-1")
        assert "## Sujet de la réunion" in out
        assert "## Principaux enseignements" in out
        assert "Lancement produit" in out

    def test_missing_sections_get_defaults(self) -> None:
        out = format_summary_markdown("", "x", "en")
        # 6 sections total → 6 "None" placeholders
        assert out.count("None") >= 6

    def test_missing_sections_get_fr_defaults(self) -> None:
        out = format_summary_markdown("", "x", "fr")
        assert out.count("Aucune") >= 6


class TestSimpleFormatMarkdown:
    def test_english_output(self) -> None:
        out = simple_format_markdown("Title", "http://v.com", "body", "Positive", "en")
        assert "Video Summary" in out
        assert "Title: Title" in out
        assert "**Sentiment:** Positive" in out
        assert "body" in out

    def test_french_output(self) -> None:
        out = simple_format_markdown("Titre", "http://v.com", "corps", "Neutre", "fr")
        assert "Résumé de la vidéo" in out
        assert "Titre : Titre" in out
        assert "**Sentiment :** Neutre" in out

    def test_unsupported_language(self) -> None:
        out = simple_format_markdown("t", "p", "b", "s", "de")
        assert "Error" in out
