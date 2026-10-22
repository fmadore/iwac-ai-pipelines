"""Tests for common.reconciliation normalization and matching."""

import csv

import pytest

from common.reconciliation import (
    Redirect,
    build_authority_dict,
    calculate_similarity,
    find_potential_matches,
    normalize_location_name,
    plural_fold_key,
    reconcile_column_values,
    token_order_key,
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


def test_build_authority_dict_stops_when_a_set_cannot_be_fetched():
    # Skipping the set would hide cross-set ambiguity: "Union" would resolve
    # to item 10 alone and be linked with a model annotation.
    class FailingClient(FakeOmekaClient):
        def get_items(self, item_set_id):
            if int(item_set_id) == 2:
                raise ConnectionError("read timed out")
            return super().get_items(item_set_id)

    client = FailingClient({1: [_item(10, "Union")], 2: [_item(20, "Union")]})
    with pytest.raises(RuntimeError, match="item set 2"):
        build_authority_dict(client, ["1", "2"])


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


def _reconcile_one(tmp_path, value, authority, ambiguous, redirect=None, extra=None):
    """Reconcile one Subject AI cell; return (output row, result tuple)."""
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    row = {"o:id": "1", "Subject AI": value, **(extra or {})}
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    result = reconcile_column_values(
        str(source), str(output), authority, "Subject AI", "Subject IDs",
        str(tmp_path / "input"), "subject", ambiguous, redirect=redirect,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle)), result


def test_token_order_matches_reversed_person_name(tmp_path):
    client = FakeOmekaClient({266: [_item(67772, "Coulibaly Kassoum")]})
    authority, ambiguous, _ = build_authority_dict(client, ["266"], token_order=True)
    row, _ = _reconcile_one(tmp_path, "Kassoum Coulibaly", authority, ambiguous)
    assert row["Subject IDs"] == "67772"


def test_token_order_is_off_unless_asked(tmp_path):
    # Reference indexing builds its dicts without the looser keys.
    client = FakeOmekaClient({266: [_item(67772, "Coulibaly Kassoum")]})
    authority, ambiguous, _ = build_authority_dict(client, ["266"])
    row, _ = _reconcile_one(tmp_path, "Kassoum Coulibaly", authority, ambiguous)
    assert row["Subject IDs"] == ""


def test_token_order_is_exact_never_fuzzy():
    assert token_order_key("Kassoum Coulibaly") == token_order_key("Coulibaly Kassoum")
    assert token_order_key("Kassoum Coulibali") != token_order_key("Coulibaly Kassoum")
    # Same words is not enough: the multiset must be equal.
    assert token_order_key("Kassoum Kassoum Coulibaly") != token_order_key("Coulibaly Kassoum")


def test_token_order_gives_no_key_to_initials_or_single_words():
    # "A. S." and "S. A." are two different signatures.
    assert token_order_key("A. S.") == ""
    assert token_order_key("Sawadogo A.") == ""
    assert token_order_key("Ouagadougou") == ""
    # Elided articles are not initials.
    assert token_order_key("Institut Musulman d'Éducation et d'Enseignement") == token_order_key(
        "Institut Musulman d'Enseignement et d'Éducation"
    )


def test_exact_match_wins_over_a_reordered_one(tmp_path):
    # Two records whose words are the same in another order: each exact title
    # still reaches its own record, and only an unseen order is ambiguous.
    client = FakeOmekaClient({266: [
        _item(1, "Moussa Ibrahim"),
        _item(2, "Ibrahim Moussa"),
        _item(3, "Ibrahim Moussa Diallo"),
        _item(4, "Diallo Ibrahim Moussa"),
    ]})
    authority, ambiguous, _ = build_authority_dict(client, ["266"], token_order=True)
    row, _ = _reconcile_one(tmp_path, "Moussa Ibrahim|Ibrahim Moussa", authority, ambiguous)
    assert row["Subject IDs"] == "1|2"
    row, _ = _reconcile_one(tmp_path, "Moussa Diallo Ibrahim", authority, ambiguous)
    assert row["Subject IDs"] == ""
    assert "Moussa Diallo Ibrahim (Ambiguous)" in (
        tmp_path / "input_unreconciled_subject.csv"
    ).read_text(encoding="utf-8")


def test_plural_fold_matches_singular_and_plural(tmp_path):
    client = FakeOmekaClient({854: [
        _item(571, "Cercle d'Études, de Recherches et de Formation Islamiques"),
    ]})
    authority, ambiguous, _ = build_authority_dict(client, ["854"], plural_fold_sets=["854"])
    row, _ = _reconcile_one(
        tmp_path, "Cercle D'Étude, de Recherche et de Formation Islamique", authority, ambiguous,
    )
    assert row["Subject IDs"] == "571"


def test_plural_fold_keeps_accents_and_short_words():
    # Stripping accents too would give "Jeunes" (youth) the key of "Jeûne" (fasting).
    assert plural_fold_key("Jeunes") != plural_fold_key("Jeûne")
    assert plural_fold_key("Jeunes") == plural_fold_key("Jeune")
    # Words of four letters or fewer keep their final letter.
    assert plural_fold_key("Unis") != plural_fold_key("Uni")
    assert plural_fold_key("Lieux de culte") == plural_fold_key("Lieu de culte")
    # Decomposed accents fold like composed ones.
    assert plural_fold_key("Études") == plural_fold_key("Étude")


def test_plural_fold_does_not_link_jeunes_to_jeune(tmp_path):
    client = FakeOmekaClient({1: [_item(76041, "Jeûne")]})
    authority, ambiguous, _ = build_authority_dict(
        client, ["1"], token_order=True, plural_fold_sets=["1"],
    )
    row, _ = _reconcile_one(tmp_path, "Jeunes", authority, ambiguous)
    assert row["Subject IDs"] == ""


def test_redirect_sends_a_subject_naming_a_place_to_spatial(tmp_path):
    place = {"universite de ouagadougou": "444"}
    row, result = _reconcile_one(
        tmp_path, "Université de Ouagadougou|Islam", {"islam": "5"}, {},
        redirect=Redirect(place, {}, "Spatial IDs"),
        extra={"Spatial IDs": "268|444"},
    )
    assert row["Subject IDs"] == "5"
    assert row["Spatial IDs"] == "268|444"  # deduplicated, order kept
    assert result[1] == 2  # both values matched


def test_redirect_never_overrides_a_subject_or_its_ambiguity(tmp_path):
    place = {"sahel": "60540", "union": "99"}
    row, _ = _reconcile_one(
        tmp_path, "Sahel|Union", {"sahel": "7"}, {"union": ["20", "30"]},
        redirect=Redirect(place, {}, "Spatial IDs"),
    )
    assert row["Subject IDs"] == "7"
    assert row["Spatial IDs"] == ""


def test_plural_fold_skips_sets_left_out(tmp_path):
    # "Prophète" is an alias of Mahomet; "Prophètes" are the prophets.
    client = FakeOmekaClient({
        266: [_item(2013, "Mahomet", ["Prophète"])],
        1: [_item(78445, "Confréries")],
    })
    authority, ambiguous, _ = build_authority_dict(
        client, ["266", "1"], token_order=True, plural_fold_sets=["1"],
    )
    row, _ = _reconcile_one(tmp_path, "Prophètes|Confrérie|Prophète", authority, ambiguous)
    assert row["Subject IDs"] == "78445|2013"


def test_country_qualifier_matches_bare_name_for_the_runs_country_only():
    from common.reconciliation import strip_country_qualifier
    assert strip_country_qualifier("Ministère de la Santé (Burkina Faso)", "Burkina Faso") == "Ministère de la Santé"
    assert strip_country_qualifier("Région des Savanes (Togo)", "Burkina Faso") == ""
    assert strip_country_qualifier("Ministère de la Santé", "Burkina Faso") == ""
    assert strip_country_qualifier("Bénin (pays)", "Benin") == ""

    client = FakeOmekaClient({854: [
        _item(1, "Ministère de la Santé (Burkina Faso)"),
        _item(2, "Ministère de la Santé (Togo)"),
        _item(3, "Gendarmerie nationale (Burkina Faso)"),
    ]})
    authority, ambiguous, _ = build_authority_dict(client, ["854"], country="Burkina Faso")
    assert authority["ministère de la santé"] == "1"
    assert authority["gendarmerie nationale"] == "3"
    assert "ministère de la santé (togo)" in authority
    unqualified, _, _ = build_authority_dict(client, ["854"])
    assert "ministère de la santé" not in unqualified


def test_country_qualifier_never_resolves_a_name_another_record_carries_bare():
    client = FakeOmekaClient({854: [
        _item(1, "Conseil constitutionnel (Burkina Faso)"),
        _item(4, "Conseil constitutionnel"),
    ]})
    authority, ambiguous, _ = build_authority_dict(client, ["854"], country="Burkina Faso")
    assert "conseil constitutionnel" not in authority
    assert ambiguous["conseil constitutionnel"] == ["1", "4"]


def test_acronyms_match_only_in_capitals_when_asked():
    from common.reconciliation import resolve_term
    client = FakeOmekaClient({854: [
        _item(23699, "Front de la Oumma Islamique", ["FOI"]),
        _item(578, "Communauté Musulmane du Burkina Faso", ["C.M.B.F"]),
        _item(579, "Commission Nationale d'Organisation du Pèlerinage (Burkina Faso)", ["CNOPM (Burkina Faso)"]),
    ]})
    authority, ambiguous, _ = build_authority_dict(client, ["854"], country="Burkina Faso",
                                                   acronyms_need_capitals=True)
    assert resolve_term("FOI", authority, ambiguous) == ("23699", [])
    assert resolve_term("Foi", authority, ambiguous) == (None, [])
    assert resolve_term("foi", authority, ambiguous) == (None, [])
    assert resolve_term("CMBF", authority, ambiguous) == ("578", [])
    assert resolve_term("CNOPM", authority, ambiguous) == ("579", [])
    assert resolve_term("Cnopm", authority, ambiguous) == (None, [])
    assert resolve_term("Front de la Oumma Islamique", authority, ambiguous) == ("23699", [])

    lenient, lenient_ambiguous, _ = build_authority_dict(client, ["854"])
    assert resolve_term("Foi", lenient, lenient_ambiguous) == ("23699", [])
