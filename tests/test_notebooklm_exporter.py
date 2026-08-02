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
        lambda article, client, country: f"#{article['o:id']}\n",
    )

    exporter.write_articles_to_file(
        [{"o:id": 1}, {"o:id": 2}],
        str(output),
        "unused",
    )

    assert output.read_text(encoding="utf-8") == "#1\n#2\n"
