"""AI_NER/02b: the context check may only choose among the ids it was offered."""

import importlib.util
import json
import sys
from pathlib import Path

from common.reconciliation import build_authority_dict
from common.reconciliation_cli import AuthorityDicts

_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ner_context_check", _ROOT / "AI_NER" / "02b_context_check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


class FakeClient:
    def __init__(self, items_by_set):
        self.items_by_set = items_by_set

    def get_items(self, item_set_id):
        return self.items_by_set.get(int(item_set_id), [])


def _item(item_id, title, *alternatives):
    data = {"o:id": item_id, "dcterms:title": [{"@value": title}]}
    if alternatives:
        data["dcterms:alternative"] = [{"@value": a} for a in alternatives]
    return data


def _dicts():
    places = FakeClient({268: [_item(444, "Université de Ouagadougou"), _item(14061, "Yirgou")]})
    subjects = FakeClient({266: [_item(1, "Abdoulaye Ouédraogo"), _item(2, "Abdoulaye Ouédraogo")],
                           854: [_item(579, "Commission Nationale d'Organisation du Pèlerinage (Burkina Faso)")]})
    spatial = build_authority_dict(places, ["268"])
    subject = build_authority_dict(subjects, ["266", "854"], country="Burkina Faso")
    return AuthorityDicts(*spatial, *subject)


def test_entries_follow_step_02_including_redirects_and_ambiguity():
    row = {"Spatial AI": "Yirgou|Mars", "Subject AI": "Abdoulaye Ouédraogo|Université de Ouagadougou|"
           "Commission Nationale d'Organisation du Pèlerinage|CNOP"}
    entries = check.article_entries(row, _dicts(), {"cnop": [579, 753, 654, 1]})
    got = [(e.term, e.where, e.status, e.candidates) for e in entries]
    assert got == [
        ("Yirgou", "place", "match", [14061]),
        ("Abdoulaye Ouédraogo", "subject", "ambiguous", [1, 2]),
        ("Université de Ouagadougou", "place", "match", [444]),
        ("Commission Nationale d'Organisation du Pèlerinage", "subject", "match", [579]),
        ("CNOP", "subject", "approximate", [579, 753, 654]),
    ]


def test_choices_outside_the_candidates_are_refused():
    entries = [check.Entry(1, "Yirgou", "place", "match", [14061]),
               check.Entry(2, "Abdoulaye Ouédraogo", "subject", "ambiguous", [1, 2]),
               check.Entry(3, "CNOP", "subject", "approximate", [579]),
               check.Entry(4, "AIB", "subject", "match", [99])]
    decisions = [check.EntryDecision(n=1, choice=13757, reason="the clashes"),
                 check.EntryDecision(n=2, choice=2, reason="the imam of the text"),
                 check.EntryDecision(n=4, choice=None, reason="byline")]
    outcomes = check.apply_decisions(entries, decisions)
    assert [(o["choice"], o["change"]) for o in outcomes] == [
        (14061, "keep"), (2, "add"), (None, "none"), (None, "drop")]
    assert outcomes[0]["reason"].startswith("refused id 13757")
    assert outcomes[2]["reason"] == "no decision returned"
    assert check.checked_ids(outcomes) == {"place": [14061], "subject": [2]}


def test_checked_file_keeps_the_provenance_of_the_ner_run(tmp_path):
    from common.omeka_link_updater import provenance_model_key
    source = tmp_path / "country_x_processed_claude_haiku_5_5.csv"
    source.write_text("o:id\n", encoding="utf-8")
    (tmp_path / (source.name + ".checkpoint.json")).write_text(json.dumps(
        {"version": 1, "context": {"model_key": "claude-haiku-5-5"}, "entries": {}}), encoding="utf-8")
    checked = tmp_path / "country_x_processed_claude_haiku_5_5_reconciled_checked.csv"
    assert provenance_model_key(checked) == "claude-haiku-5-5"
