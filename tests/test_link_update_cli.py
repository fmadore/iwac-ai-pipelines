"""The link write step shared by AI_NER/03 and AI_reference_indexing/05, end to end."""

import argparse
import csv
import io
import os
from unittest.mock import Mock

import pytest
from rich.console import Console

from common import link_update_cli
from common.iwac_config import AI_MODEL_ITEMS
from common.link_update_cli import (
    add_link_update_args,
    find_latest_reconciled_csv,
    load_newly_created_mapping,
    read_reconciled_rows,
    run_link_update,
)

MODEL_KEY = next(iter(AI_MODEL_ITEMS))


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def reconciled(tmp_path):
    path = tmp_path / "item_set_1_processed_reconciled.csv"
    _write_csv(path, ["o:id", "Subject AI Reconciled ID", "Spatial AI Reconciled ID"], [
        {"o:id": "1", "Subject AI Reconciled ID": "42", "Spatial AI Reconciled ID": "7"},
        {"o:id": "2", "Subject AI Reconciled ID": "", "Spatial AI Reconciled ID": ""},
    ])
    return path


@pytest.fixture
def client(monkeypatch):
    fake = Mock()
    fake.base_url = "https://example.invalid/api"
    fake.get_items_by_ids.return_value = {1: {"o:id": 1}, 2: {"o:id": 2}}
    fake.update_item.return_value = True
    monkeypatch.setattr(link_update_cli.OmekaClient, "from_env", staticmethod(lambda: fake))
    return fake


def _args(tmp_path, *argv):
    parser = add_link_update_args(argparse.ArgumentParser(), output_dir=tmp_path)
    return parser.parse_args(list(argv))


def _run(tmp_path, *argv):
    return run_link_update(
        _args(tmp_path, *argv), output_dir=tmp_path, banner="b", backup_label="test",
        console=Console(file=io.StringIO()),
    )


def test_newest_main_csv_is_found_and_diagnostics_are_not(tmp_path):
    older = tmp_path / "a_reconciled.csv"
    newer = tmp_path / "b_reconciled.csv"
    diagnostic = tmp_path / "b_unreconciled_spatial_reconciled.csv"
    for index, path in enumerate((older, newer, diagnostic)):
        path.write_text("o:id\n", encoding="utf-8")
        os.utime(path, (1_000 + index, 1_000 + index))
    assert find_latest_reconciled_csv(tmp_path) == newer
    assert find_latest_reconciled_csv(tmp_path / "absent") is None


def test_a_csv_that_names_no_items_is_refused(tmp_path):
    headerless = tmp_path / "empty.csv"
    headerless.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_reconciled_rows(headerless)
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("title\nx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="o:id"):
        read_reconciled_rows(wrong)


def test_new_item_mapping_is_case_folded_and_skips_incomplete_rows(tmp_path):
    path = tmp_path / "newly_created_items_subject.csv"
    _write_csv(path, ["term", "o:id"], [
        {"term": " Zakat ", "o:id": "900"}, {"term": "", "o:id": "901"}, {"term": "Hadj", "o:id": ""},
    ])
    assert load_newly_created_mapping(str(path)) == {"zakat": "900"}
    assert load_newly_created_mapping(None) == {}
    assert load_newly_created_mapping(str(tmp_path / "absent.csv")) == {}


def test_dry_run_reads_but_never_writes_or_backs_up(tmp_path, reconciled, client):
    assert _run(tmp_path, "--dry-run", "--model", MODEL_KEY) == 0
    client.update_item.assert_not_called()
    assert not list(tmp_path.glob("*.jsonl"))


def test_live_run_annotates_each_added_link_and_backs_up_first(tmp_path, reconciled, client):
    assert _run(tmp_path, "--yes", "--model", MODEL_KEY) == 0

    (item_id, payload), _ = client.update_item.call_args
    assert item_id == 1 and client.update_item.call_count == 1
    for prop, target in (("dcterms:subject", 42), ("dcterms:spatial", 7)):
        (link,) = payload[prop]
        assert link["value_resource_id"] == target
        (stamp,) = link["@annotation"]["iwac:nerModel"]
        assert stamp["value_resource_id"] == AI_MODEL_ITEMS[MODEL_KEY]["item_id"]
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_declined_confirmation_and_missing_input_both_fail_without_writing(tmp_path, reconciled, client, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    assert _run(tmp_path, "--model", MODEL_KEY) == 1
    assert _run(tmp_path, "--yes", "--model", MODEL_KEY, "--input", str(tmp_path / "absent.csv")) == 1
    client.update_item.assert_not_called()


def test_a_failed_patch_makes_the_run_exit_non_zero(tmp_path, reconciled, client):
    client.update_item.return_value = False
    assert _run(tmp_path, "--yes", "--model", MODEL_KEY) == 1
