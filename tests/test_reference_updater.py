"""Tests for the final reference-indexing Omeka update step."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "AI_reference_indexing"
    / "05_update_omeka.py"
)
SPEC = importlib.util.spec_from_file_location("reference_omeka_updater", SCRIPT_PATH)
assert SPEC and SPEC.loader
updater = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = updater
SPEC.loader.exec_module(updater)


def test_resolved_ids_combines_reconciliation_and_new_authorities():
    row = {
        "Subject AI Reconciled ID": "10|invalid|20",
        "Subject AI": "History | Unknown | PRESS",
    }

    assert updater.resolved_ids(
        row,
        "Subject AI Reconciled ID",
        "Subject AI",
        {"history": "30", "press": "40"},
    ) == [10, 20, 30, 40]


def test_update_reconciled_item_adds_only_missing_links():
    client = MagicMock()
    item = {
        "o:id": 7,
        "dcterms:spatial": [{"type": "resource:item", "value_resource_id": 10}],
    }
    client.get_item.return_value = item
    client.update_item.return_value = True
    row = {
        "o:id": "7",
        "Spatial AI Reconciled ID": "10|20",
        "Subject AI Reconciled ID": "",
        "Subject AI": "History|Unknown",
    }

    result = updater.update_reconciled_item(
        client,
        row,
        new_spatial_map={},
        new_subject_map={"history": "30"},
    )

    assert result == updater.ItemUpdateResult(
        "modified", spatial_added=1, subject_added=1,
    )
    client.update_item.assert_called_once_with(7, item)
    assert [
        value["value_resource_id"] for value in item["dcterms:spatial"]
    ] == [10, 20]
    assert item["dcterms:subject"][0]["value_resource_id"] == 30


@pytest.mark.parametrize("item_id", ["", "not-an-id"])
def test_invalid_item_id_never_contacts_omeka(item_id):
    client = MagicMock()

    result = updater.update_reconciled_item(
        client,
        {"o:id": item_id},
        new_spatial_map={},
        new_subject_map={},
    )

    assert result.status in {"skipped", "error"}
    client.get_item.assert_not_called()
    client.update_item.assert_not_called()


def test_no_resolved_links_skips_fetch_and_patch():
    client = MagicMock()

    result = updater.update_reconciled_item(
        client,
        {"o:id": "7", "Subject AI": "Unknown"},
        new_spatial_map={},
        new_subject_map={},
    )

    assert result == updater.ItemUpdateResult("skipped")
    client.get_item.assert_not_called()


def test_failed_patch_is_an_error_and_does_not_report_links_added():
    client = MagicMock()
    client.get_item.return_value = {"o:id": 7}
    client.update_item.return_value = False

    result = updater.update_reconciled_item(
        client,
        {"o:id": "7", "Subject AI Reconciled ID": "30"},
        new_spatial_map={},
        new_subject_map={},
    )

    assert result == updater.ItemUpdateResult("error")


def test_read_reconciled_rows_rejects_missing_item_id(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Subject AI\nHistory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing columns: o:id"):
        updater.read_reconciled_rows(str(csv_path))


def test_dry_run_reports_without_patching():
    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    client.get_item.return_value = {"o:id": 7}

    result = updater.update_reconciled_item(
        client,
        {"o:id": "7", "Subject AI Reconciled ID": "30"},
        new_spatial_map={},
        new_subject_map={},
        dry_run=True,
    )

    assert result == updater.ItemUpdateResult("modified", spatial_added=0, subject_added=1)
    client.update_item.assert_not_called()


def test_pre_write_snapshot_is_taken_before_links_are_appended():
    """The snapshot is the only route back after a bulk PATCH — it must be the original."""
    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    client.get_item.return_value = {
        "o:id": 7,
        "dcterms:subject": [{"type": "resource:item", "value_resource_id": 10}],
    }
    client.update_item.return_value = True
    snapshots = []

    updater.update_reconciled_item(
        client,
        {"o:id": "7", "Subject AI Reconciled ID": "30"},
        new_spatial_map={},
        new_subject_map={},
        on_pre_write=snapshots.append,
    )

    assert len(snapshots) == 1
    assert snapshots[0]["dcterms:subject"] == [
        {"type": "resource:item", "value_resource_id": 10}
    ]


def test_batch_dry_run_writes_no_backup_and_no_patch(tmp_path):
    from common.write_guard import WriteGuard

    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    client.get_item.return_value = {"o:id": 7}

    stats = updater.update_reconciled_items(
        client,
        [{"o:id": "7", "Subject AI Reconciled ID": "30"}],
        new_spatial_map={},
        new_subject_map={},
        guard=WriteGuard(dry_run=True, backup_dir=tmp_path),
    )

    assert stats["modified"] == 1
    client.update_item.assert_not_called()
    assert list(tmp_path.glob("_pre_write_*.json")) == []


def test_batch_live_run_dumps_pre_write_payloads(tmp_path):
    from common.write_guard import WriteGuard

    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    client.get_item.return_value = {"o:id": 7}
    client.update_item.return_value = True

    updater.update_reconciled_items(
        client,
        [{"o:id": "7", "Subject AI Reconciled ID": "30"}],
        new_spatial_map={},
        new_subject_map={},
        guard=WriteGuard(backup_dir=tmp_path),
    )

    assert len(list(tmp_path.glob("_pre_write_reference_links_*.json"))) == 1
