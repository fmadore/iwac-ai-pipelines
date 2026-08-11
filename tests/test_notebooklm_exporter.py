"""Tests for NotebookLM export routing and reverse-link collection."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "NotebookLM"
    / "omeka_items_to_md.py"
)
SPEC = importlib.util.spec_from_file_location("notebooklm_exporter", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def test_cli_request_supports_all_legacy_forms():
    assert exporter.parse_cli_request(["all"]) == exporter.ExportRequest("all")
    assert exporter.parse_cli_request(["123"]) == exporter.ExportRequest("item_set", "123")
    assert exporter.parse_cli_request(["subject:456"]) == exporter.ExportRequest(
        "subject", "456",
    )
    assert exporter.parse_cli_request(["--subject", "789"]) == exporter.ExportRequest(
        "subject", "789",
    )


def test_cli_request_rejects_malformed_ids():
    assert exporter.parse_cli_request(["subject:not-a-number"]) is None
    assert exporter.parse_cli_request(["--subject"]) is None
    assert exporter.parse_cli_request(["unknown"]) is None


def test_reverse_reference_skips_embedded_non_article_without_fetch():
    client = MagicMock()

    status, item = exporter._fetch_subject_reference(
        client,
        {"o:id": 7, "@type": ["o:Item", "bibo:Issue"]},
        set(),
    )

    assert (status, item) == ("non_article", None)
    client.get_item.assert_not_called()


def test_reverse_reference_fetches_article_and_deduplicates():
    client = MagicMock()
    article = {"o:id": 7, "@type": ["o:Item", "bibo:Article"]}
    client.get_item.return_value = article
    seen = set()
    reference = {"@id": "https://example.org/api/items/7"}

    assert exporter._fetch_subject_reference(client, reference, seen) == (
        "article", article,
    )
    assert exporter._fetch_subject_reference(client, reference, seen) == (
        "duplicate", None,
    )
    client.get_item.assert_called_once_with(7)


def test_article_file_write_replaces_complete_output(monkeypatch, tmp_path):
    output = tmp_path / "articles.md"
    output.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        exporter,
        "format_article",
        lambda article, client, country, include_summary: f"#{article['o:id']}\n",
    )

    exporter.write_articles_to_file(
        [{"o:id": 1}, {"o:id": 2}],
        str(output),
        "Item Set: Le Matin (ID 42)",
    )

    assert output.read_text(encoding="utf-8") == (
        "# Item Set: Le Matin (ID 42)\n\n#1\n#2\n"
    )


def test_split_parts_are_named_in_their_header(monkeypatch, tmp_path):
    """A part file is otherwise indistinguishable once NotebookLM ingests it."""
    output = tmp_path / "articles_part2.md"
    monkeypatch.setattr(
        exporter, "format_article", lambda *args, **kwargs: "",
    )

    exporter.write_articles_to_file(
        [], str(output), "Item Set: Le Matin (ID 42)", 2, num_parts=5,
    )

    assert output.read_text(encoding="utf-8").startswith(
        "# Item Set: Le Matin (ID 42) — Part 2 of 5\n"
    )


ARTICLE = {
    "o:id": 2233,
    "o:title": "Le CERFI forme 68 jeunes",
    "dcterms:date": [{"@value": "2018-04-03"}],
    "dcterms:publisher": [{"display_title": "L'Observateur"}],
    "bibo:content": [{"@value": "Corps de l'article."}],
    "bibo:shortDescription": [
        {"@value": "Résumé français.", "@language": "fr"},
        {"@value": "English summary.", "@language": "en"},
    ],
}


def _client():
    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    return client


def test_article_is_h2_with_traceable_item_link():
    markdown = exporter.format_article(ARTICLE, _client(), "Burkina Faso")

    assert markdown.startswith("## Le CERFI forme 68 jeunes\n")
    assert (
        "**Journal :** L'Observateur | **Date :** 2018-04-03 | "
        "**Pays :** Burkina Faso | "
        "**Item :** [2233](https://islam.zmo.de/s/afrique_ouest/item/2233)"
    ) in markdown
    assert markdown.rstrip().endswith("---")


def test_article_falls_back_to_bare_id_without_a_client():
    markdown = exporter.format_article(ARTICLE)

    assert "**Item :** 2233" in markdown
    assert "](http" not in markdown


def test_summaries_are_omitted_unless_requested():
    assert "Résumé (IA)" not in exporter.format_article(ARTICLE, _client())

    with_summary = exporter.format_article(
        ARTICLE, _client(), None, include_summary=True,
    )
    assert "**Résumé (IA) :** Résumé français." in with_summary
    # The French literal is the one exported; English is a separate value.
    assert "English summary." not in with_summary


def test_summary_extraction_is_language_aware():
    assert exporter.extract_summary(ARTICLE) == "Résumé français."
    assert exporter.extract_summary(ARTICLE, "en") == "English summary."


def test_untagged_legacy_summary_counts_as_french():
    """Summaries written before 2026-08-06 carry no @language tag."""
    legacy = {"bibo:shortDescription": [{"@value": "Ancien résumé."}]}

    assert exporter.extract_summary(legacy) == "Ancien résumé."
    assert exporter.extract_summary(legacy, "en") is None


def test_summary_extraction_tolerates_missing_property():
    assert exporter.extract_summary({}) is None
    assert exporter.extract_summary({"bibo:shortDescription": []}) is None


def test_summary_flag_is_accepted_on_either_side_of_the_mode():
    assert exporter.extract_summary_flag(["12345", "--with-summaries"]) == (
        ["12345"], True,
    )
    assert exporter.extract_summary_flag(["--with-summaries", "all"]) == (
        ["all"], True,
    )
    assert exporter.extract_summary_flag(["all"]) == (["all"], False)
