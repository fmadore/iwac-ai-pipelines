"""The reference-indexing enrichment step: skipping rules, cleaning, durable output."""

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "AI_reference_indexing"
    / "02_enrich_references.py"
)
SPEC = importlib.util.spec_from_file_location("reference_enrichment", SCRIPT_PATH)
assert SPEC and SPEC.loader
enrich = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = enrich
SPEC.loader.exec_module(enrich)


def _row(item_id, content="Texte sur les confréries à Dakar.", subjects="", spatial=""):
    return {
        "o:id": str(item_id),
        "Title": f"Référence {item_id}",
        "Existing Subject IDs": subjects,
        "Existing Spatial IDs": spatial,
        "bibo:content": content,
    }


def _client(subjects=("Confréries", "Tidjaniyya"), spatial=("Dakar", "Sénégal")):
    client = Mock()
    client.generate_structured.return_value = enrich.ReferenceKeywords(
        subjects=list(subjects), spatial=list(spatial)
    )
    return client


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        enrich.main(["--help"])
    assert excinfo.value.code == 0


def test_enriched_name_keeps_the_prefix_step_three_looks_for():
    out = enrich.enriched_path_for(Path("output/items_78405_20260902.csv"))
    assert out.name == "items_enriched_78405_20260902.csv"


def test_items_with_no_text_or_both_link_sets_are_skipped():
    assert not enrich.needs_enrichment(_row(1, content="  "), reindex=False)
    assert not enrich.needs_enrichment(_row(1, subjects="10", spatial="20"), reindex=False)
    assert enrich.needs_enrichment(_row(1, subjects="10"), reindex=False)
    assert enrich.needs_enrichment(_row(1, subjects="10", spatial="20"), reindex=True)
    assert not enrich.needs_enrichment(_row(1, content="", subjects="10"), reindex=True)


def test_clean_terms_drops_generic_repeated_and_existing_terms():
    cleaned = enrich.clean_terms(
        ["Islam", " Confréries ", "confréries", "", "Tidjaniyya", "Musulmans"],
        existing=["tidjaniyya"],
    )
    assert cleaned == ["Confréries"]


def test_existing_links_are_named_for_the_model_and_not_repeated(tmp_path):
    titles = {"10": "Tidjaniyya", "20": "Dakar"}
    client = _client(subjects=("Tidjaniyya", "Confréries"), spatial=("Dakar", "Sénégal"))

    result = enrich.enrich_reference(client, "system", _row(1, subjects="10", spatial="20"), titles)

    user_prompt = client.generate_structured.call_args.args[1]
    assert "EXISTING SUBJECTS: Tidjaniyya" in user_prompt
    assert "EXISTING SPATIAL: Dakar" in user_prompt
    assert result.subjects == ["Confréries"]
    assert result.spatial == ["Sénégal"]


def test_run_writes_every_row_flushed_and_resumes_without_repeating(tmp_path):
    output = tmp_path / "items_enriched_x.csv"
    rows = [_row(1), _row(2, content=""), _row(3)]
    fieldnames = list(rows[0]) + [enrich.SUBJECT_COLUMN, enrich.SPATIAL_COLUMN]
    client = _client()

    stats = enrich.run(
        client, "system", rows, {}, output, fieldnames=fieldnames, resume=False, reindex=False
    )

    assert stats["enriched"] == 2 and stats["skipped"] == 1
    with output.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [r["o:id"] for r in written] == ["1", "2", "3"]
    assert written[0][enrich.SUBJECT_COLUMN] == "Confréries|Tidjaniyya"
    assert written[1][enrich.SUBJECT_COLUMN] == ""

    # A resumed run appends only the rows it is handed and calls the model for those alone.
    client.generate_structured.reset_mock()
    enrich.run(
        client, "system", [_row(4)], {}, output, fieldnames=fieldnames, resume=True, reindex=False
    )
    assert client.generate_structured.call_count == 1
    with output.open(encoding="utf-8", newline="") as handle:
        assert [r["o:id"] for r in csv.DictReader(handle)] == ["1", "2", "3", "4"]


def test_failed_rows_are_left_out_so_a_rerun_retries_them(tmp_path):
    output = tmp_path / "items_enriched_x.csv"
    client = Mock()
    client.generate_structured.side_effect = RuntimeError("boom")
    fieldnames = list(_row(1)) + [enrich.SUBJECT_COLUMN, enrich.SPATIAL_COLUMN]

    stats = enrich.run(
        client, "system", [_row(1)], {}, output, fieldnames=fieldnames, resume=False, reindex=False
    )

    assert stats["failed"] == 1
    with output.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_keyword_summary_counts_terms_by_type(tmp_path):
    enriched = tmp_path / "items_enriched_x.csv"
    with enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["o:id", "Subject AI", "Spatial AI"])
        writer.writeheader()
        writer.writerow({"o:id": "1", "Subject AI": "Confréries|Presse", "Spatial AI": "Dakar"})
        writer.writerow({"o:id": "2", "Subject AI": "Presse", "Spatial AI": ""})

    summary = enrich.write_keyword_summary(enriched, tmp_path)

    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {"Term": "Presse", "Type": "subject", "Count": "2"}
    assert {(r["Term"], r["Type"]) for r in rows} == {
        ("Presse", "subject"), ("Confréries", "subject"), ("Dakar", "spatial")
    }
