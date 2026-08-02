"""The NER write step: argv handling, dry runs, and the pre-write backup.

On 2026-08-02 this script had no argument parser, so ``--help`` fell through to
the real update and PATCHed 630 live items before it was stopped. The first test
here is the regression test for exactly that.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.write_guard import WriteGuard


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "AI_NER" / "03_Omeka_update.py"
SPEC = importlib.util.spec_from_file_location("ner_omeka_updater", SCRIPT_PATH)
assert SPEC and SPEC.loader
updater = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = updater
SPEC.loader.exec_module(updater)


ROWS = [
    {"o:id": "7", "Spatial AI Reconciled ID": "10|20", "Subject AI Reconciled ID": "30"},
]


def fake_client(item=None, *, patch_ok=True):
    client = MagicMock()
    client.base_url = "https://islam.zmo.de/api"
    client.get_item.return_value = item if item is not None else {"o:id": 7}
    client.update_item.return_value = patch_ok
    return client


def test_help_exits_without_touching_omeka():
    """The regression test for the 2026-08-02 incident."""
    with pytest.raises(SystemExit) as excinfo:
        updater.main(["--help"])
    assert excinfo.value.code == 0


def test_unknown_flag_aborts_instead_of_running():
    with pytest.raises(SystemExit) as excinfo:
        updater.main(["--patch-everything"])
    assert excinfo.value.code != 0


def test_parser_exposes_the_write_safety_flags():
    args = updater.build_parser().parse_args([])
    assert args.dry_run is False
    assert args.yes is False
    assert hasattr(args, "backup_dir")


def test_dry_run_reports_changes_without_patching():
    client = fake_client()

    stats = updater.update_rows(client, ROWS, guard=WriteGuard(dry_run=True))

    client.update_item.assert_not_called()
    assert stats["modified"] == 1
    assert stats["spatial_added"] == 2
    assert stats["subject_added"] == 1


def test_dry_run_and_live_run_report_the_same_totals():
    dry = updater.update_rows(fake_client(), ROWS, guard=WriteGuard(dry_run=True))
    live = updater.update_rows(fake_client(), ROWS, guard=WriteGuard())

    assert dry["modified"] == live["modified"]
    assert dry["spatial_added"] == live["spatial_added"]
    assert dry["subject_added"] == live["subject_added"]


def test_live_run_dumps_pre_write_payloads_before_patching(tmp_path):
    item = {
        "o:id": 7,
        "dcterms:spatial": [{"type": "resource:item", "value_resource_id": 10}],
    }
    client = fake_client(item)

    updater.update_rows(client, ROWS, guard=WriteGuard(backup_dir=tmp_path))

    dumps = list(tmp_path.glob("_pre_write_ner_links_*.json"))
    assert len(dumps) == 1
    backup = json.loads(dumps[0].read_text(encoding="utf-8"))
    # The snapshot is the item as fetched, before any link was appended.
    assert backup == [item] or backup[0]["dcterms:spatial"] == [
        {"type": "resource:item", "value_resource_id": 10}
    ]


def test_backup_records_only_items_that_actually_change(tmp_path):
    """An item already carrying every link is never PATCHed, so it is not backed up."""
    already_linked = {
        "o:id": 7,
        "dcterms:spatial": [
            {"type": "resource:item", "value_resource_id": 10},
            {"type": "resource:item", "value_resource_id": 20},
        ],
        "dcterms:subject": [{"type": "resource:item", "value_resource_id": 30}],
    }
    client = fake_client(already_linked)

    stats = updater.update_rows(client, ROWS, guard=WriteGuard(backup_dir=tmp_path))

    client.update_item.assert_not_called()
    assert stats["modified"] == 0
    assert list(tmp_path.glob("_pre_write_*.json")) == []


def test_failed_patch_is_counted_as_an_error():
    client = fake_client(patch_ok=False)

    stats = updater.update_rows(client, ROWS, guard=WriteGuard())

    assert stats["errors"] == 1
    assert stats["modified"] == 0


def test_rows_without_an_id_are_skipped():
    client = fake_client()

    stats = updater.update_rows(client, [{"o:id": ""}], guard=WriteGuard())

    client.get_item.assert_not_called()
    assert stats["skipped"] == 1
