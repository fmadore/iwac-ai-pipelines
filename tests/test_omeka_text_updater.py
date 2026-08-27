"""Tests for the shared Omeka text-update runner.

The behaviours pinned here are the ones that used to differ between the four
pipelines that each had their own copy: skipping unchanged items, honouring
--dry-run, and never PATCHing when the operator declines.
"""

import io
import json
from unittest.mock import MagicMock

from rich.console import Console

from common.omeka_text_updater import (
    apply_text_value,
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


def test_closed_stdin_declines_rather_than_crashing(monkeypatch):
    """An EOF is not consent, and must not surface as a write failure.

    Before this, ``console.input`` raised straight out of the gate on a piped or
    cron run — no writes happened, but the caller saw an exception rather than a
    clean decline.
    """
    client = fake_client({"o:id": 1})

    def raise_eof(self, *a, **k):
        raise EOFError

    monkeypatch.setattr(Console, "input", raise_eof)

    stats = run_text_updates(client, [TextUpdate("1.txt", 1, "text")], TARGET,
                             console=quiet_console())

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


# ---------------------------------------------------------------------------
# extra_values — several values on one item, one PATCH
# ---------------------------------------------------------------------------

# Mirrors AI_summary/03: French adopts the untagged legacy literal, English never does.
FRENCH = PropertyTarget(
    term="bibo:shortDescription", property_id=116, language="fr", adopt_untagged=True,
)
ENGLISH = PropertyTarget(term="bibo:shortDescription", property_id=116, language="en")


def test_extra_values_land_in_a_single_patch():
    """Two PATCHes per item would double the round trips over a 12k corpus and
    leave a window where an item holds one language and not the other."""
    client = fake_client({"o:id": 1})
    update = TextUpdate("1.txt", 1, "Résumé.", extra_values=[(ENGLISH, "Summary.")])

    stats = run_text_updates(
        client, [update], FRENCH, console=quiet_console(), require_confirmation=False,
    )

    assert stats["updated"] == 1
    client.update_item.assert_called_once()
    written = client.update_item.call_args[0][1]["bibo:shortDescription"]
    assert {v["@language"]: v["@value"] for v in written} == {
        "fr": "Résumé.", "en": "Summary.",
    }


def test_item_is_skipped_only_when_every_value_is_empty():
    client = fake_client({"o:id": 1})
    updates = [
        TextUpdate("1.txt", 1, "  ", extra_values=[(ENGLISH, "Summary.")]),
        TextUpdate("2.txt", 2, "  ", extra_values=[(ENGLISH, "  ")]),
    ]

    stats = run_text_updates(
        client, updates, FRENCH, console=quiet_console(), require_confirmation=False,
    )

    assert (stats["updated"], stats["empty"]) == (1, 1)


# ---------------------------------------------------------------------------
# Pre-write backup — the only route back from a bulk overwrite
# ---------------------------------------------------------------------------

def _backup_lines(tmp_path):
    files = list(tmp_path.glob("_pre_write_*.jsonl"))
    assert len(files) == 1, f"expected one backup file, found {files}"
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_backup_captures_the_state_before_the_patch(tmp_path):
    """What the backup must contain is the OLD summary, not the new one."""
    existing = {"o:id": 1, "bibo:shortDescription": [
        {"@value": "ancien résumé", "property_id": 116, "type": "literal"}
    ]}
    client = fake_client(existing)

    run_text_updates(
        client, [TextUpdate("1.txt", 1, "nouveau résumé")], FRENCH,
        console=quiet_console(), require_confirmation=False, backup_dir=tmp_path,
    )

    captured = _backup_lines(tmp_path)
    assert len(captured) == 1
    assert captured[0]["bibo:shortDescription"][0]["@value"] == "ancien résumé"
    # And the PATCH really did send the new one.
    assert client.update_item.call_args[0][1]["bibo:shortDescription"][0]["@value"] == "nouveau résumé"


def test_backup_is_flushed_before_each_patch(tmp_path):
    """A crash mid-run must leave a backup of everything already overwritten.

    ``WriteGuard.dump_backup`` buffers and writes once at the end, so an
    interrupted corpus pass would produce nothing at all — which is exactly when
    the backup is needed.
    """
    seen = []

    def blow_up_on_the_third(item_id, data):
        seen.append(item_id)
        if len(seen) == 3:
            raise RuntimeError("connection reset")
        return True

    client = fake_client({"o:id": 0})
    client.get_item.side_effect = lambda i: {"o:id": i, "bibo:shortDescription": [
        {"@value": f"old {i}", "property_id": 116, "type": "literal"}]}
    client.update_item.side_effect = blow_up_on_the_third

    stats = run_text_updates(
        client, [TextUpdate(f"{i}.txt", i, f"new {i}") for i in (1, 2, 3, 4)], FRENCH,
        console=quiet_console(), require_confirmation=False, backup_dir=tmp_path,
    )

    captured = _backup_lines(tmp_path)
    assert [c["o:id"] for c in captured] == [1, 2, 3, 4]
    assert [c["bibo:shortDescription"][0]["@value"] for c in captured[:2]] == ["old 1", "old 2"]
    assert stats["failed"] == 1


def test_dry_run_writes_no_backup(tmp_path):
    client = fake_client({"o:id": 1})

    run_text_updates(
        client, [TextUpdate("1.txt", 1, "text")], FRENCH, console=quiet_console(),
        require_confirmation=False, dry_run=True, backup_dir=tmp_path,
    )

    assert list(tmp_path.iterdir()) == []


def test_unchanged_items_are_not_backed_up(tmp_path):
    """The backup records what was overwritten, not everything inspected."""
    same = {"o:id": 1, "bibo:shortDescription": [
        {"@value": "identique", "property_id": 116, "type": "literal", "@language": "fr"}
    ]}
    client = fake_client(same)

    run_text_updates(
        client, [TextUpdate("1.txt", 1, "identique")], FRENCH,
        console=quiet_console(), require_confirmation=False, backup_dir=tmp_path,
    )

    assert _backup_lines(tmp_path) == []


# ---------------------------------------------------------------------------
# is_public — per-value visibility
# ---------------------------------------------------------------------------
#
# Omeka's ``is_public`` sits on the value, not the item, and the Hugging Face
# export reads it as ``OCR_is_public`` to decide whether to mask a row's full
# text. A pipeline whose sources are copyrighted has to state it: a value
# created without one defaults to public.


def test_new_value_is_public_by_default():
    """The historical behaviour every pipeline had before the flag existed."""
    item = {"o:id": 1}
    apply_text_value(item, TARGET, "texte")
    assert item["bibo:content"][0]["is_public"] is True


def test_new_value_honours_an_explicit_private_target():
    item = {"o:id": 1}
    target = PropertyTarget(
        term="bibo:content", property_id=91, property_label="content", is_public=False
    )
    apply_text_value(item, target, "texte de thèse sous droits")
    assert item["bibo:content"][0]["is_public"] is False


def test_existing_public_value_is_made_private_when_the_target_says_so():
    item = {
        "o:id": 1,
        "bibo:content": [
            {"type": "literal", "property_id": 91, "@value": "ancien", "is_public": True}
        ],
    }
    target = PropertyTarget(
        term="bibo:content", property_id=91, property_label="content", is_public=False
    )
    assert apply_text_value(item, target, "nouveau")
    assert item["bibo:content"][0]["is_public"] is False


def test_a_visibility_change_alone_counts_as_a_change():
    """Otherwise a re-run would report 'unchanged' and leave the text public."""
    item = {
        "o:id": 1,
        "bibo:content": [
            {"type": "literal", "property_id": 91, "@value": "texte", "is_public": True}
        ],
    }
    target = PropertyTarget(
        term="bibo:content", property_id=91, property_label="content", is_public=False
    )
    assert apply_text_value(item, target, "texte") is True


def test_default_target_never_republishes_a_private_value():
    """A curator's decision to hide a value outranks a pipeline with no opinion.

    This is the reason the field is ``Optional[bool]`` rather than ``bool``:
    defaulting to ``True`` would have every existing pipeline quietly publish
    the values it touches.
    """
    item = {
        "o:id": 1,
        "bibo:content": [
            {"type": "literal", "property_id": 91, "@value": "ancien", "is_public": False}
        ],
    }
    apply_text_value(item, TARGET, "nouveau")
    assert item["bibo:content"][0]["is_public"] is False


def test_private_write_keeps_every_other_property():
    """The PATCH carries the whole item; nothing outside the target is touched."""
    item = {
        "o:id": 1,
        "dcterms:title": [{"type": "literal", "property_id": 1, "@value": "Titre"}],
        "dcterms:subject": [{"type": "resource:item", "property_id": 3, "value_resource_id": 42}],
        "o:item_set": [{"o:id": 2212}],
    }
    target = PropertyTarget(
        term="bibo:content", property_id=91, property_label="content", is_public=False
    )
    apply_text_value(item, target, "texte")
    assert item["dcterms:title"][0]["@value"] == "Titre"
    assert item["dcterms:subject"][0]["value_resource_id"] == 42
    assert item["o:item_set"] == [{"o:id": 2212}]
