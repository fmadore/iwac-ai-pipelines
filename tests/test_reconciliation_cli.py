"""The reconciliation run shared by AI_NER/02 and AI_reference_indexing/03."""

import csv
import io
import json
import os

from rich.console import Console

from common import reconciliation_cli
from common.checkpoint import checkpoint_path_for
from common.reconciliation_cli import (
    SPATIAL_OUTPUT_COLUMN,
    SUBJECT_OUTPUT_COLUMN,
    checkpoint_country,
    find_input_csv,
    run_reconciliation,
)


class FakeOmekaClient:
    def __init__(self, items_by_set):
        self.items_by_set = items_by_set

    def get_items(self, item_set_id):
        return self.items_by_set.get(str(item_set_id), [])


def _item(item_id, title):
    return {"o:id": item_id, "dcterms:title": [{"@value": title}]}


AUTHORITIES = {
    "268": [_item(10, "Ouagadougou"), _item(11, "Cotonou")],
    "854": [_item(20, "Conseil supérieur islamique")],
    "1": [_item(30, "Ramadan")],
}


def _write_ai_csv(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["o:id", "Spatial AI", "Subject AI"])
        writer.writeheader()
        writer.writerow({"o:id": "1", "Spatial AI": "Ouagadougou|Koudougou", "Subject AI": "Ramadan"})
        writer.writerow({"o:id": "2", "Spatial AI": "Cotonou", "Subject AI": "Conseil supérieur islamique|Zakat"})


def test_newest_ai_output_is_chosen_and_derived_files_never_are(tmp_path):
    old, new = tmp_path / "item_set_1_processed.csv", tmp_path / "item_set_2_processed.csv"
    derived = tmp_path / "item_set_2_processed_reconciled.csv"
    for index, path in enumerate((old, new, derived)):
        path.write_text("o:id\n", encoding="utf-8")
        os.utime(path, (1_000 + index, 1_000 + index))
    assert find_input_csv(tmp_path) == new
    assert find_input_csv(tmp_path, accept=lambda name: "set_1" in name) == old
    assert find_input_csv(tmp_path / "absent") is None


def test_country_comes_from_the_checkpoint_beside_the_input(tmp_path):
    source = tmp_path / "run.csv"
    assert checkpoint_country(source) is None
    checkpoint_path_for(source).write_text(
        json.dumps({"context": {"spatial_filter": "Burkina Faso"}}), encoding="utf-8",
    )
    assert checkpoint_country(source) == "Burkina Faso"


def test_run_writes_ids_for_matches_and_lists_what_is_left(tmp_path):
    source = tmp_path / "item_set_1_processed.csv"
    _write_ai_csv(source)

    reconciled = run_reconciliation(
        FakeOmekaClient(AUTHORITIES), source, subject_tag="subject_and_topic",
        console=Console(file=io.StringIO()),
    )

    assert reconciled == tmp_path / "item_set_1_processed_reconciled.csv"
    with reconciled.open(encoding="utf-8", newline="") as handle:
        rows = {row["o:id"]: row for row in csv.DictReader(handle)}
    assert rows["1"][SPATIAL_OUTPUT_COLUMN] == "10"
    assert rows["1"][SUBJECT_OUTPUT_COLUMN] == "30"
    assert rows["2"][SPATIAL_OUTPUT_COLUMN] == "11"
    assert rows["2"][SUBJECT_OUTPUT_COLUMN] == "20"
    unmatched_places = (tmp_path / "item_set_1_processed_unreconciled_spatial.csv").read_text(encoding="utf-8")
    unmatched_subjects = (
        tmp_path / "item_set_1_processed_unreconciled_subject_and_topic.csv"
    ).read_text(encoding="utf-8")
    assert "Koudougou" in unmatched_places and "Ouagadougou" not in unmatched_places
    assert "Zakat" in unmatched_subjects


def test_main_fails_without_credentials_or_input(tmp_path, monkeypatch):
    options = dict(output_dir=tmp_path, subject_tag="subject", banner="b", description="d")

    def no_credentials():
        raise ValueError("OMEKA_BASE_URL is not set")

    monkeypatch.setattr(reconciliation_cli.OmekaClient, "from_env", staticmethod(no_credentials))
    assert reconciliation_cli.main([], **options) == 1

    monkeypatch.setattr(
        reconciliation_cli.OmekaClient, "from_env", staticmethod(lambda: FakeOmekaClient(AUTHORITIES)),
    )
    assert reconciliation_cli.main([], **options) == 1

    source = tmp_path / "item_set_1_processed.csv"
    _write_ai_csv(source)
    assert reconciliation_cli.main([], **options) == 0
    assert (tmp_path / "item_set_1_processed_reconciled.csv").exists()
