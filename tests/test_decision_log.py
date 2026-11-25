"""The curation registry: what it refuses, what it records, and that the shipped file parses."""
import csv
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "AI_authority_curation"
sys.path.insert(0, str(PIPELINE))

import decision_log as dl  # noqa: E402

D = dl.Decision


def test_the_shipped_registry_parses():
    decisions = dl.load()
    assert decisions, "decisions.csv should not be empty"
    with dl.REGISTRY.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == dl.COLUMNS


def test_unknown_action_or_status_is_an_error():
    with pytest.raises(ValueError):
        D("2026-11-25", "rename", (1,))
    with pytest.raises(ValueError):
        D("2026-11-25", "merge", (1, 2), status="maybe")


def test_a_refused_alternative_is_taken_out_and_the_rest_kept():
    decisions = [D("2026-11-20", "alternative", (248,), "Fête nationale", "rejected")]
    row = {"authority_id": "248", "alternatives": "Fete Nationale | Fête du 13 janvier"}
    assert dl.refusal("plan_alternatives", row, decisions) is None
    assert row["alternatives"] == "Fête du 13 janvier"
    lone = {"authority_id": "248", "alternatives": "Fête nationale"}
    assert "Fête nationale" in dl.refusal("plan_alternatives", lone, decisions)
    other = {"authority_id": "249", "alternatives": "Fête nationale"}
    assert dl.refusal("plan_alternatives", other, decisions) is None


def test_a_refused_merge_is_refused_either_way_round_and_a_loser_cannot_merge_twice():
    decisions = [D("2026-11-25", "merge", (65, 15171), status="rejected"),
                 D("2026-11-24", "merge", (23504, 46268))]
    assert dl.merge_refusal(15171, 65, decisions)
    assert "already merged" in dl.merge_refusal(12, 46268, decisions)
    assert dl.merge_refusal(23504, 99, decisions) is None


def test_an_identifier_refusal_without_a_value_covers_every_match():
    decisions = [D("2026-11-24", "identifier", (15065,), "", "rejected"),
                 D("2026-11-24", "identifier", (7,), "Q42", "rejected")]
    assert dl.identifier_refusal(15065, "Q1", decisions)
    assert dl.identifier_refusal(7, "q42", decisions)
    assert dl.identifier_refusal(7, "Q43", decisions) is None


def test_screen_plan_unapproves_and_explains_in_problems_or_note():
    decisions = [D("2026-11-25", "create", (), "Forever (Lomé)", "rejected")]
    rows = [{"approve": "yes", "status": "ready", "title": "Forever (Lome)", "note": ""},
            {"approve": "", "status": "ready", "title": "Tchamba 1", "note": ""}]
    assert dl.screen_plan("plan_create", rows, decisions) == 1
    assert rows[0]["approve"] == "" and rows[0]["status"] == "rejected"
    assert rows[0]["note"].startswith("registry: record refused")
    assert rows[1]["status"] == "ready"


def test_append_skips_what_is_already_recorded_and_pending_drops_closed_rows(tmp_path):
    path = tmp_path / "decisions.csv"
    first = [D("2026-11-24", "delete", (46268,), "Diaby Moustapha", "pending"),
             D("2026-11-24", "delete", (188541,), "La Haye", "pending")]
    assert dl.append(first, path) == 2
    assert dl.append(first, path) == 0
    dl.append([D("2026-11-25", "delete", (46268,), "Diaby Moustapha", "done")], path)
    decisions = dl.load(path)
    assert [d.records for d in dl.pending(decisions)] == [(188541,)]
    assert len(dl.about(46268, decisions)) == 2


def test_approve_no_rows_become_rejected_decisions(tmp_path):
    plan = tmp_path / "plan_alternatives.csv"
    with plan.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["approve", "authority_id", "alternatives", "note"])
        writer.writeheader()
        writer.writerow({"approve": "no", "authority_id": "248", "alternatives": "Fête nationale | 13 janvier",
                         "note": "generic"})
        writer.writerow({"approve": "yes", "authority_id": "532", "alternatives": "Klouto", "note": ""})
    found = dl.refusals_from_plan(plan, "output/run")
    assert [(d.action, d.records, d.value, d.status) for d in found] == [
        ("alternative", (248,), "Fête nationale", "rejected"),
        ("alternative", (248,), "13 janvier", "rejected"),
    ]
