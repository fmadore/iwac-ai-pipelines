"""Tests for the shared Omeka text-update runner.

The behaviours pinned here are the ones that used to differ between the four
pipelines that each had their own copy: skipping unchanged items, honouring
--dry-run, and never PATCHing when the operator declines.
"""

import io
from unittest.mock import MagicMock

from rich.console import Console

from common.omeka_text_updater import (
    PropertyTarget,
    TextUpdate,
    run_text_updates,
    update_item_text,
    updates_from_directory,
)

TARGET = PropertyTarget(term="bibo:content", property_id=91, property_label="content")


def quiet_console():
    return Console(file=io.StringIO(), force_terminal=False)


def fake_client(item=None, update_ok=True):
    client = MagicMock()
    client.base_url = "https://example.org/api"
    client.get_item.return_value = item
    client.update_item.return_value = update_ok
    return client


# ---------------------------------------------------------------------------
# update_item_text
# ---------------------------------------------------------------------------

def test_updates_when_text_differs():
    client = fake_client({"o:id": 1})
    assert update_item_text(client, 1, "new text", TARGET) == "updated"
    client.update_item.assert_called_once()


def test_skips_patch_when_already_up_to_date():
    """The regression this shared module was written to fix.

    Two of the four updaters re-PATCHed every item on every run because they
    never compared against what Omeka already held.
    """
    existing = {"o:id": 1, "bibo:content": [
        {"@value": "same text", "property_id": 91, "type": "literal"}
    ]}
    client = fake_client(existing)

    assert update_item_text(client, 1, "same text", TARGET) == "unchanged"
    client.update_item.assert_not_called()


def test_dry_run_never_patches():
    client = fake_client({"o:id": 1})
    assert update_item_text(client, 1, "new text", TARGET, dry_run=True) == "would_update"
    client.update_item.assert_not_called()


def test_missing_item_reported_not_found():
    client = fake_client(None)
    assert update_item_text(client, 1, "text", TARGET) == "not_found"
    client.update_item.assert_not_called()


def test_failed_patch_reported():
    client = fake_client({"o:id": 1}, update_ok=False)
    assert update_item_text(client, 1, "text", TARGET) == "failed"


# ---------------------------------------------------------------------------
# updates_from_directory
# ---------------------------------------------------------------------------

def test_reads_item_ids_from_filenames(tmp_path):
    (tmp_path / "123.txt").write_text("  first  ", encoding="utf-8")
    (tmp_path / "456.txt").write_text("second", encoding="utf-8")

    updates = updates_from_directory(tmp_path)

    assert [(u.item_id, u.text) for u in updates] == [(123, "first"), (456, "second")]


def test_skips_non_numeric_stems(tmp_path):
    """The item ID comes from the filename; a non-numeric stem is not ours."""
    (tmp_path / "123.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "report_final.txt").write_text("skip", encoding="utf-8")

    assert [u.item_id for u in updates_from_directory(tmp_path)] == [123]


def test_strip_false_keeps_layout(tmp_path):
    (tmp_path / "1.txt").write_text("\n  indented page\n\n", encoding="utf-8")
    assert updates_from_directory(tmp_path, strip=False)[0].text == "\n  indented page\n\n"


# ---------------------------------------------------------------------------
# run_text_updates
# ---------------------------------------------------------------------------

def test_declining_confirmation_writes_nothing(monkeypatch):
    client = fake_client({"o:id": 1})
    console = quiet_console()
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: "n")

    stats = run_text_updates(
        client, [TextUpdate("1.txt", 1, "text")], TARGET, console=console,
    )

    assert stats == {}
    client.update_item.assert_not_called()


def test_unresolved_items_counted_not_dropped():
    client = fake_client({"o:id": 1})
    updates = [TextUpdate("missing", None, ""), TextUpdate("1.txt", 1, "text")]

    stats = run_text_updates(
        client, updates, TARGET, console=quiet_console(), require_confirmation=False,
    )

    assert stats["not_found"] == 1
    assert stats["updated"] == 1


def test_empty_text_is_skipped_not_written():
    client = fake_client({"o:id": 1})

    stats = run_text_updates(
        client, [TextUpdate("1.txt", 1, "   ")], TARGET,
        console=quiet_console(), require_confirmation=False,
    )

    assert stats["empty"] == 1
    client.update_item.assert_not_called()


def test_dry_run_reports_without_writing():
    client = fake_client({"o:id": 1})

    stats = run_text_updates(
        client, [TextUpdate("1.txt", 1, "text")], TARGET,
        console=quiet_console(), require_confirmation=False, dry_run=True,
    )

    assert stats["would_update"] == 1
    client.update_item.assert_not_called()
