"""Tests for common.reconciliation normalization and matching."""

import csv

from common.reconciliation import (
    build_authority_dict,
    calculate_similarity,
    find_potential_matches,
    normalize_location_name,
    reconcile_column_values,
)


def test_normalize_strips_accents_and_spacing():
    assert normalize_location_name("Côte d'Ivoire") == normalize_location_name("Cote d'Ivoire")
    assert normalize_location_name("Ouaga-dougou") == normalize_location_name("Ouagadougou")
    assert normalize_location_name("") == ""


def test_exact_match_scores_one():
    assert calculate_similarity("Bamako", "Bamako") == 1.0


def test_unrelated_terms_score_zero():
    assert calculate_similarity("Bamako", "Cotonou") == 0.0


def test_reversed_name_matches_strongly():
    assert calculate_similarity("Madore Frédérick", "Frédérick Madore") >= 0.9


class FakeOmekaClient:
    """Minimal stand-in returning canned items per item set."""

    def __init__(self, items_by_set):
        self.items_by_set = items_by_set

    def get_items(self, item_set_id):
        return self.items_by_set.get(int(item_set_id), [])


def _item(item_id, title, alternatives=()):
    data = {
        "o:id": item_id,
        "dcterms:title": [{"@value": title}],
    }
    if alternatives:
        data["dcterms:alternative"] = [{"@value": a} for a in alternatives]
    return data


def test_build_authority_dict_resolves_unique_terms():
    client = FakeOmekaClient({1: [_item(10, "Bamako"), _item(11, "Cotonou")]})
    authority, ambiguous, metadata = build_authority_dict(client, ["1"])
    assert authority["bamako"] == "10"
    assert not ambiguous
    assert metadata["10"]["primary_title"] == "Bamako"


def test_build_authority_dict_flags_cross_set_ambiguity():
    # The same title in two different sets maps to two different items —
    # it must land in the ambiguous dict, not silently resolve to either.
    client = FakeOmekaClient({
        1: [_item(10, "Union")],
        2: [_item(20, "Union")],
    })
    authority, ambiguous, _ = build_authority_dict(client, ["1", "2"])
    assert "union" not in authority
    assert sorted(ambiguous["union"]) == ["10", "20"]


def test_build_authority_dict_skips_linked_resource_titles():
    # Linked-resource values have no '@value'; they used to crash the build.
    client = FakeOmekaClient({
        1: [{
            "o:id": 30,
            "dcterms:title": [
                {"value_resource_id": 99},  # linked resource, no @value
                {"@value": "Niamey"},
            ],
        }],
    })
    authority, _, metadata = build_authority_dict(client, ["1"])
    assert authority["niamey"] == "30"
    assert metadata["30"]["primary_title"] == "Niamey"


def test_potential_matches_keep_best_title_per_authority_item():
    metadata = {
        "10": {"primary_title": "Bamako", "alternatives": ["Ville de Bamako"]},
        "20": {"primary_title": "Cotonou", "alternatives": []},
    }
    matches = find_potential_matches("Bamako", metadata, max_candidates=5)
    assert matches == [("10", "Bamako", 1.0, "primary_title")]


def test_reconcile_column_preserves_ambiguity_and_counts(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["o:id", "Subject AI"])
        writer.writeheader()
        writer.writerow({"o:id": "1", "Subject AI": "Bamako|Union|Unknown"})

    result = reconcile_column_values(
        str(source),
        str(output),
        {"bamako": "10"},
        "Subject AI",
        "Subject IDs",
        str(tmp_path / "input"),
        "subject",
        {"union": ["20", "30"]},
    )

    assert result == (str(output), 1, 3, 2)
    with output.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["Subject IDs"] == "10"
    unreconciled = (tmp_path / "input_unreconciled_subject.csv").read_text(
        encoding="utf-8"
    )
    assert "Union (Ambiguous)" in unreconciled
    assert "Unknown" in unreconciled
