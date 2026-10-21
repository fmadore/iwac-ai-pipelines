"""Authority curation: the checks between a Haiku verdict and the archive."""

import importlib.util
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "AI_authority_curation"
sys.path.insert(0, str(PIPELINE))

import curation_core as cc  # noqa: E402


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(rid, title, *alternatives):
    return {"o:id": rid, "o:title": title,
            "dcterms:alternative": [{"type": "literal", "@value": a} for a in alternatives]}


# --- names and evidence ------------------------------------------------------

def test_fold_keeps_length_so_snippets_quote_the_source():
    text = "L'Union Islamique du Bénin (UIB) à Porto-Novo"
    assert len(cc.fold(text)) == len(text)
    assert cc.normalize_name("El Hadj Ibrahima Niassé", strip_honorifics=True) == "ibrahima niasse"


def test_snippets_ignore_accents_and_respect_word_boundaries():
    text = "Hier, l'UNION ISLAMIQUE DU BENIN a réuni ses membres. Tuiba n'était pas là."
    found = cc.find_snippets(text, ["Union islamique du Bénin"], width=10)
    assert found and "UNION ISLAMIQUE DU BENIN" in found[0]
    assert cc.find_snippets(text, ["UIB"]) == []


def test_duplicate_candidates_find_variants_and_acronyms_only():
    records = [
        record(1, "Tidjaniya"), record(2, "Tidjaniyya"),
        record(3, "Union Islamique du Bénin"), record(4, "UIB"),
        record(5, "Mouvement Sunnite du Burkina Faso"),
    ]
    pairs = {(p.a, p.b): p.reason for p in cc.duplicate_candidates(records)}
    assert (1, 2) in pairs and "similar spelling" in pairs[(1, 2)]
    assert pairs[(3, 4)].startswith("acronym")
    assert not any(5 in pair for pair in pairs)


def test_a_shared_acronym_is_weaker_than_a_shared_name():
    records = [record(1, "Association des Jeunes Musulmans de Tanghin", "AJMT"),
               record(2, "Association de la Jeunesse Musulmane du Togo", "AJMT")]
    [pair] = cc.duplicate_candidates(records)
    assert pair.reason == "shared acronym: ajmt" and pair.score < 1.0


def test_person_duplicates_ignore_honorifics():
    people = [record(10, "El Hadj Moussa Traoré"), record(11, "Moussa Traore")]
    assert [(p.a, p.b) for p in cc.duplicate_candidates(people, strip_honorifics=True)] == [(10, 11)]


def test_sentence_count_does_not_split_at_abbreviations():
    assert cc.sentence_count("Fondée par M. Diallo en 1985. Active à Lomé.") == 2
    assert cc.sentence_count("Cheikh soufi sénégalais (1900-1975), fondateur de la Fayda.") == 1
    assert cc.sentence_count("Un. Deux. Trois.") == 3


# --- describe verdicts -------------------------------------------------------

def describe_batch(needs=("description",)):
    return {"batch": "describe_001", "task": "describe",
            "role_options": [{"id": 8874, "title": "Imam"}],
            "authorities": [{"authority": {"id": 100, "title": "X"}, "needs": list(needs),
                             "evidence": [{"item_id": 7}, {"item_id": 8}]}]}


def verdict_file(*results, model=cc.REVIEWER_MODEL_KEY):
    return {"batch": "describe_001", "model": model, "results": list(results)}


GOOD = {"authority_id": 100, "status": "ready", "description_fr": "Association béninoise.",
        "description_en": "Beninese association.", "evidence": [7]}


def checked(batch, verdicts):
    [row] = cc.check_describe_batch(batch, verdicts)
    return row


def test_a_good_description_is_ready():
    row = checked(describe_batch(), verdict_file(GOOD))
    assert (row.description_status, row.role_status) == ("ready", "none")


def test_a_verdict_from_another_model_is_never_ready():
    row = checked(describe_batch(), verdict_file(GOOD, model="claude-sonnet-5-5"))
    assert row.description_status == "invalid"
    assert "claude-sonnet-5-5" in row.description_problems[0]


@pytest.mark.parametrize("change, problem", [
    ({"description_fr": "Un. Deux. Trois."}, "3 sentences"),
    ({"description_en": ""}, "empty"),
    ({"evidence": [7, 99]}, "not given"),
    ({"evidence": []}, "no evidence"),
])
def test_rule_breaking_descriptions_are_invalid(change, problem):
    row = checked(describe_batch(), verdict_file({**GOOD, **change}))
    assert row.description_status == "invalid"
    assert problem in "; ".join(row.description_problems)


def test_an_unsure_reviewer_is_flagged_not_invalid():
    row = checked(describe_batch(), verdict_file({"authority_id": 100, "status": "unsure",
                                                  "note": "two men of this name"}))
    assert row.description_status == "flagged"


def test_missing_verdicts_are_invalid():
    assert checked(describe_batch(), None).description_status == "invalid"
    assert checked(describe_batch(), verdict_file()).description_status == "invalid"


def test_roles_must_come_from_the_offered_list_with_evidence():
    batch = describe_batch(needs=("description", "role"))
    ok = checked(batch, verdict_file({**GOOD, "role_id": 8874, "role_evidence": [8]}))
    assert ok.role_status == "ready"
    stray = checked(batch, verdict_file({**GOOD, "role_id": 9999, "role_evidence": [8]}))
    assert stray.role_status == "invalid"
    bare = checked(batch, verdict_file({**GOOD, "role_id": 8874}))
    assert bare.role_status == "invalid"
    proposal = checked(batch, verdict_file({**GOOD, "new_role": "Chef coutumier", "role_evidence": [8]}))
    assert proposal.role_status == "flagged"


def test_a_role_for_a_record_that_has_one_is_refused():
    row = checked(describe_batch(), verdict_file({**GOOD, "role_id": 8874, "role_evidence": [8]}))
    assert row.role_status == "invalid"


def test_a_missing_verdict_in_a_larger_batch_is_reported_not_crashed():
    batch = describe_batch()
    batch["authorities"].append({"authority": {"id": 101, "title": "Y"}, "needs": ["description"],
                                 "evidence": [{"item_id": 9}]})
    rows = {r.authority_id: r for r in cc.check_describe_batch(batch, verdict_file(GOOD))}
    assert rows[100].description_status == "ready"
    assert rows[101].description_status == "invalid"


def test_general_knowledge_is_allowed_but_always_shown():
    claim = {**GOOD, "description_fr": "Parti politique ivoirien fondé en 1946.",
             "general_knowledge": ["fondé en 1946"]}
    row = checked(describe_batch(), verdict_file(claim))
    assert row.description_status == "flagged"
    assert row.description_problems == ["from general knowledge, check: fondé en 1946"]


# --- triage ------------------------------------------------------------------

def triage_entry(title, candidates=()):
    return {"authority": {"id": 13795, "title": title}, "needs": ["sort", "description", "role"],
            "candidates": [{"id": c, "title": "x", "url": ""} for c in candidates],
            "evidence": [{"item_id": 7, "quotes": ["le général Kérékou, dit le Caméléon"]}]}


def triage(**fields):
    base = {"authority_id": 13795, "target": "persons", "status": "ready",
            "description_fr": "Homme politique béninois.", "description_en": "Beninese politician.",
            "evidence": [7]}
    return cc.TriageVerdict.model_validate({**base, **fields})


def test_parentheses_hide_names_and_dates():
    assert cc.name_variants(record(1, "FCDB (Forces Cauris pour le Développement du Bénin)")) == [
        "FCDB (Forces Cauris pour le Développement du Bénin)", "FCDB",
        "Forces Cauris pour le Développement du Bénin"]
    assert cc.name_variants(record(2, "Mathieu Kérékou (1933-2015)")) == [
        "Mathieu Kérékou (1933-2015)", "Mathieu Kérékou"]


def test_a_stub_finds_the_authority_it_may_duplicate():
    index = cc.NameIndex([record(500, "Mathieu Kérékou"), record(501, "Mathieu Kérékou Junior")])
    [(best, score, _), *_] = index.candidates(record(13795, "Mathieu Kérékou (1933-2015)"))
    assert best == 500 and score == 1.0


def test_a_cleaned_title_moves_dates_and_acronyms_but_loses_nothing():
    kerekou = triage_entry("Mathieu Kérékou (1933-2015)")
    assert cc.triage_problems(kerekou, triage(title="Mathieu Kérékou", birth_year=1933, death_year=2015)) == []
    fcdb = triage_entry("FCDB (Forces Cauris pour le Développement du Bénin)")
    ok = triage(target="organisations", title="Forces Cauris pour le Développement du Bénin",
                alternatives=["FCDB"])
    assert cc.triage_problems(fcdb, ok) == []
    assert "drops part" in cc.triage_problems(fcdb, triage(target="organisations", title="Forces Cauris"))[0]


@pytest.mark.parametrize("fields, problem", [
    ({"title": "Général Mathieu Kérékou"}, "adds words"),
    ({"title": "Mathieu Kérékou", "birth_year": 1934}, "not in the current title"),
    ({"target": "places", "title": "Mathieu Kérékou", "birth_year": 1933}, "only filed for persons"),
    ({"alternatives": ["Le Bélier"]}, "neither in the title nor in the evidence"),
])
def test_triage_names_are_never_invented(fields, problem):
    assert problem in "; ".join(cc.triage_problems(triage_entry("Mathieu Kérékou (1933-2015)"), triage(**fields)))


def test_an_alternative_written_in_the_evidence_is_allowed():
    entry = triage_entry("Mathieu Kérékou (1933-2015)")
    assert cc.triage_problems(entry, triage(alternatives=["le Caméléon"])) == []


def test_triage_rows_route_to_filing_merging_or_the_curator():
    batch = {"batch": "triage_001", "task": "triage",
             "authorities": [triage_entry("Mathieu Kérékou (1933-2015)", candidates=[500])]}

    def row(**fields):
        verdict = {"authority_id": 13795, "target": "persons", "status": "ready", **fields}
        [checked_row] = cc.check_triage_batch(batch, {"batch": "triage_001", "model": cc.REVIEWER_MODEL_KEY,
                                                      "results": [verdict]})
        return checked_row.status, checked_row.action

    assert row(duplicate_of=500) == ("ready", "merge")
    assert row(duplicate_of=999) == ("invalid", "merge")
    assert row(target="unsure") == ("flagged", "none")
    assert row(title="Mathieu Kérékou", birth_year=1933, death_year=2015) == ("ready", "file")
    [described] = cc.check_describe_batch(batch, {"batch": "triage_001", "model": cc.REVIEWER_MODEL_KEY,
                                                  "results": [{"authority_id": 13795, "target": "persons",
                                                               "status": "ready", "duplicate_of": 500}]})
    assert described.description_status == "none"  # merged away: nothing to describe


def test_filing_follows_the_template():
    sort = _load("05_sort.py")
    props = {"dcterms:type": 8, "dcterms:title": 1, "dcterms:alternative": 17, "foaf:birthday": 182,
             "dcterms:date": 7}
    stub = {"o:id": 14585, "o:title": "FCDB (Forces Cauris pour le Développement du Bénin)",
            "o:item_set": [{"o:id": cc.UNSORTED_SET_ID}],
            "dcterms:title": [{"type": "literal", "property_id": 1,
                               "@value": "FCDB (Forces Cauris pour le Développement du Bénin)"}]}
    row = {"target": "organisations", "new_title": "Forces Cauris pour le Développement du Bénin",
           "alternatives": "FCDB", "birth_year": "", "death_year": "", "model": cc.REVIEWER_MODEL_KEY}
    assert sort.file_stub(stub, row, props) == 2  # type + alternative
    assert stub["o:item_set"] == [{"o:id": 854}]
    assert stub["o:resource_class"] == {"o:id": 96} and stub["o:resource_template"] == {"o:id": 7}
    assert stub["dcterms:title"][0]["@value"] == "Forces Cauris pour le Développement du Bénin"
    assert stub["dcterms:type"][0]["value_resource_id"] == 67568

    person = {"o:id": 13795, "o:title": "Mathieu Kérékou (1933-2015)", "o:item_set": [{"o:id": cc.UNSORTED_SET_ID}],
              "dcterms:title": [{"type": "literal", "property_id": 1, "@value": "Mathieu Kérékou (1933-2015)"}]}
    sort.file_stub(person, {"target": "persons", "new_title": "Mathieu Kérékou", "alternatives": "",
                            "birth_year": "1933", "death_year": "2015", "model": cc.REVIEWER_MODEL_KEY}, props)
    assert person["foaf:birthday"][0]["@value"] == "1933" and person["dcterms:date"][0]["@value"] == "2015"

    filed = {"o:id": 1, "o:item_set": [{"o:id": cc.UNSORTED_SET_ID}, {"o:id": 266}],
             "o:resource_class": {"o:id": 94}}
    assert sort.file_stub(filed, {"target": "persons", "model": "", "note": "already in the persons set"},
                          props) == 0
    assert filed["o:item_set"] == [{"o:id": 266}] and "dcterms:type" not in filed


class CountingClient:
    def __init__(self):
        self.calls = 0

    def count_items(self, **filters):
        self.calls += 1
        return int(filters["property[0][text]"]) % 3  # stub 3 → 0 links, 4 → 1, 5 → 2


def test_link_counts_are_cached_and_only_new_stubs_are_counted(tmp_path):
    collect = _load("01_collect.py")
    client, cache, props = CountingClient(), tmp_path / "counts.json", {"dcterms:subject": 3}
    first = collect.stub_link_counts(client, [record(3, "a"), record(4, "b")], props, cache, recount=False)
    assert first == {3: {"dcterms:subject": 0}, 4: {"dcterms:subject": 1}} and client.calls == 2
    collect.stub_link_counts(client, [record(3, "a"), record(4, "b"), record(5, "c")], props, cache, False)
    assert client.calls == 3  # only the new stub
    collect.stub_link_counts(client, [record(3, "a")], props, cache, recount=True)
    assert client.calls == 4


def test_a_rerun_skips_stubs_an_earlier_run_reviewed(tmp_path):
    import json

    collect = _load("01_collect.py")
    for run, ids in (("unsorted_1", [10, 11]), ("unsorted_2", [12])):
        (tmp_path / run / "batches").mkdir(parents=True)
        (tmp_path / run / "batches" / "triage_001.json").write_text(json.dumps(
            {"authorities": [{"authority": {"id": i}} for i in ids]}), encoding="utf-8")
    assert collect.previously_batched(tmp_path, tmp_path / "unsorted_2") == {10, 11}


# --- Wikidata ----------------------------------------------------------------

def wikidata_batch(**candidate):
    batch = describe_batch(needs=("wikidata",))
    batch["authorities"][0]["wikidata_candidates"] = [
        {"qid": "Q7164", "label": "Banque mondiale", "url": "https://www.wikidata.org/wiki/Q7164", **candidate}]
    return batch


@pytest.mark.parametrize("pick, candidate, status", [
    ("Q7164", {}, "ready"),
    (None, {}, "none"),
    ("Q999", {}, "invalid"),
    ("Q7164", {"used_by": {"id": 42, "title": "World Bank"}}, "flagged"),
    (None, {"used_by": {"id": 42, "title": "World Bank"}}, "flagged"),  # a duplicate signal, kept
])
def test_a_wikidata_pick_must_be_offered_and_unused(pick, candidate, status):
    row = checked(wikidata_batch(**candidate),
                  verdict_file({"authority_id": 100, "status": "ready", "wikidata": pick}))
    assert row.wikidata_status == status
    assert row.description_status == "none"  # a wikidata-only batch asks for nothing else


def test_a_record_that_has_wikidata_gets_no_second_one():
    row = checked(describe_batch(), verdict_file({**GOOD, "wikidata": "Q7164"}))
    assert row.wikidata_status == "invalid"


def test_wikidata_identifiers_are_read_and_written_in_the_archive_shape():
    from wikidata import qid_from, record_qids

    stored = {"type": "uri", "property_id": 10, "@id": "https://www.wikidata.org/wiki/Q357779",
              "o:label": "Q357779"}
    assert qid_from(stored) == "Q357779"
    assert record_qids({"dcterms:identifier": [stored, {"@value": "iwac-index-0003056"}]}) == ["Q357779"]
    apply = _load("03_apply.py")
    assert apply.wikidata_value("Q357779", 10) == {**stored, "is_public": True}


def test_a_candidate_already_in_the_archive_is_marked_as_a_duplicate_signal():
    collect = _load("01_collect.py")

    class Search:
        def candidates(self, names):
            return [{"qid": "Q7164"}, {"qid": "Q1"}]

    entry = {"needs": ["description"]}
    used = collect.used_qids([{"o:id": 42, "o:title": "World Bank",
                               "dcterms:identifier": [{"@id": "https://www.wikidata.org/wiki/Q7164"}]}])
    collect.attach_wikidata(entry, record(100, "Banque Mondiale"), Search(), used)
    assert entry["needs"] == ["description", "wikidata"]
    assert entry["wikidata_candidates"][0]["used_by"] == {"id": 42, "title": "World Bank"}
    assert "used_by" not in entry["wikidata_candidates"][1]


# --- duplicate verdicts ------------------------------------------------------

def duplicate_batch():
    return {"batch": "duplicates_001", "task": "duplicates",
            "pairs": [{"a": {"id": 2}, "b": {"id": 1}, "score": 0.9, "candidate_reason": "x"}]}


@pytest.mark.parametrize("verdict, status", [
    ({"a": 1, "b": 2, "verdict": "same", "survivor_id": 1}, "ready"),
    ({"a": 2, "b": 1, "verdict": "same", "survivor_id": 3}, "invalid"),
    ({"a": 1, "b": 2, "verdict": "different", "reason": "father and son"}, "rejected"),
    ({"a": 1, "b": 2, "verdict": "unsure"}, "flagged"),
])
def test_duplicate_verdicts(verdict, status):
    [row] = cc.check_duplicate_batch(duplicate_batch(),
                                     {"batch": "duplicates_001", "model": cc.REVIEWER_MODEL_KEY,
                                      "results": [verdict]})
    assert row.status == status


# --- merging -----------------------------------------------------------------

BASE = "https://islam.zmo.de/api"


def link(rid, **extra):
    return {"type": "resource:item", "property_id": 3, "value_resource_id": rid,
            "@id": f"{BASE}/items/{rid}", "display_title": "old", **extra}


def test_repoint_keeps_the_annotation_and_drops_display_keys():
    annotation = {"iwac:nerModel": [{"value_resource_id": 125443}]}
    item = {"o:id": 50, "dcterms:subject": [link(2, **{"@annotation": annotation})]}
    assert cc.repoint_links(item, loser=2, survivor=1, base_url=BASE) == (1, 0)
    [value] = item["dcterms:subject"]
    assert value["value_resource_id"] == 1 and value["@id"].endswith("/items/1")
    assert value["@annotation"] == annotation and "display_title" not in value


def test_repoint_never_creates_a_second_link_or_a_self_link():
    item = {"o:id": 50, "dcterms:subject": [link(1), link(2)]}
    assert cc.repoint_links(item, loser=2, survivor=1, base_url=BASE) == (0, 1)
    assert [v["value_resource_id"] for v in item["dcterms:subject"]] == [1]
    survivor = {"o:id": 1, "dcterms:relation": [link(2)]}
    assert cc.repoint_links(survivor, loser=2, survivor=1, base_url=BASE) == (0, 1)
    assert survivor["dcterms:relation"] == []


def test_an_existing_link_to_the_survivor_wins_over_a_retargeted_one():
    """The hand-made link keeps its place and its (absent) annotation."""
    ai = {"iwac:nerModel": [{"value_resource_id": 125443}]}
    item = {"o:id": 50, "dcterms:subject": [link(2, **{"@annotation": ai}), link(9), link(1)]}
    assert cc.repoint_links(item, loser=2, survivor=1, base_url=BASE) == (0, 1)
    assert [v["value_resource_id"] for v in item["dcterms:subject"]] == [9, 1]
    assert "@annotation" not in item["dcterms:subject"][1]


def test_no_property_ends_up_naming_the_survivor_twice():
    item = {"o:id": 50, "dcterms:subject": [link(1), link(2), link(1), link(2)],
            "dcterms:spatial": [link(2)]}
    cc.repoint_links(item, loser=2, survivor=1, base_url=BASE)
    assert [v["value_resource_id"] for v in item["dcterms:subject"]] == [1]
    assert [v["value_resource_id"] for v in item["dcterms:spatial"]] == [1]  # other properties keep theirs
    assert cc.repeated_links(item, [1]) == []
    assert cc.repeated_links({"dcterms:subject": [link(1), link(1)]}, [1]) == [("dcterms:subject", 1)]


def test_an_untouched_property_is_left_alone():
    item = {"o:id": 50, "dcterms:subject": [link(7), link(7)]}  # not ours to clean
    assert cc.repoint_links(item, loser=2, survivor=1, base_url=BASE) == (0, 0)
    assert len(item["dcterms:subject"]) == 2


def test_the_losers_names_become_alternatives_once():
    survivor = record(1, "Tidjaniya", "Tijaniyya")
    loser = record(2, "Tidjaniyya", "Tijaniyya", "Tidjanya")
    assert cc.missing_alternatives(survivor, loser) == ["Tidjaniyya", "Tidjanya"]


def test_merging_a_stub_adds_variants_not_dates():
    assert cc.missing_alternatives(record(1, "Mathieu Kérékou"), record(2, "Mathieu Kérékou (1933-2015)")) == []
    fcdb = record(2, "FCDB (Forces Cauris pour le Développement du Bénin)")
    assert cc.missing_alternatives(record(1, "Forces Cauris pour le Développement du Bénin"), fcdb) == ["FCDB"]


# --- apply -------------------------------------------------------------------

def test_an_edited_description_loses_the_model_annotation():
    apply = _load("03_apply.py")
    row = {"status": "ready", "model": cc.REVIEWER_MODEL_KEY, "description_fr": "A.",
           "draft_fr": "A.", "description_en": "B.", "draft_en": "B."}
    assert apply.is_reviewer_text(row)
    assert apply.is_reviewer_text({**row, "status": "flagged"})  # checked, approved unchanged
    assert not apply.is_reviewer_text({**row, "description_fr": "A, corrected."})
    assert not apply.is_reviewer_text({**row, "status": "flagged", "draft_fr": "", "draft_en": ""})
    assert not apply.is_reviewer_text({**row, "model": "claude-opus-5-5"})


def test_a_year_missing_from_the_evidence_flags_the_draft():
    batch = describe_batch()
    batch["authorities"][0]["evidence"][0]["quotes"] = ["L'association, créée en 1985, …"]
    supported = checked(batch, verdict_file({**GOOD, "description_fr": "Association fondée en 1985."}))
    assert supported.description_status == "ready"
    invented = checked(batch, verdict_file({**GOOD, "description_fr": "Association fondée en 1963."}))
    assert invented.description_status == "flagged"
    assert "1963" in invented.description_problems[0]


def test_an_acronym_matches_as_written_not_as_a_word():
    text = "Leur foi est grande. Le FOI a organisé le pèlerinage."
    [snippet] = cc.find_snippets(text, ["FOI"], width=5)
    assert "FOI" in snippet and "foi est" not in snippet


# --- NER leftovers: plans and new records --------------------------------------

PROPS = {"dcterms:title": 1, "dcterms:alternative": 17, "dcterms:description": 4, "dcterms:type": 8,
         "dcterms:isPartOf": 33, "dcterms:spatial": 40}
ANNOTATION = {"type": "resource:item", "value_resource_id": 125443}


def _payload(**row):
    base = {"kind": "places", "title": "Région du Nord (Burkina Faso)", "alternatives": "",
            "description_fr": "", "category_id": "", "country_id": ""}
    return cc.new_record_payload({**base, **row}, PROPS, type_value={"value_resource_id": 67568},
                                 description_annotation=ANNOTATION, link_annotation=ANNOTATION,
                                 link_value=lambda rid: {"type": "resource:item", "value_resource_id": rid})


def test_new_place_is_filed_with_its_country_as_parent():
    item = _payload(alternatives="Yaadga | yaadga | Région du Nord (Burkina Faso)", country_id="546",
                    description_fr="Région administrative du nord du Burkina Faso.")
    assert item["o:item_set"] == [{"o:id": 268}] and item["o:resource_class"] == {"o:id": 9}
    assert [v["@value"] for v in item["dcterms:alternative"]] == ["Yaadga"]
    assert [v["value_resource_id"] for v in item["dcterms:isPartOf"]] == [546]
    assert "dcterms:spatial" not in item
    assert item["dcterms:isPartOf"][0]["@annotation"] == {"iwac:nerModel": [ANNOTATION]}
    description = item["dcterms:description"][0]
    assert description["@language"] == "fr" and description["@annotation"] == {"iwac:descriptionModel": [ANNOTATION]}


def test_new_person_gets_country_as_spatial_and_category_as_parent():
    item = _payload(kind="persons", title="Mohamed Bazoum", category_id="8990", country_id="540")
    assert item["o:item_set"] == [{"o:id": 266}]
    assert [v["value_resource_id"] for v in item["dcterms:isPartOf"]] == [8990]
    assert [v["value_resource_id"] for v in item["dcterms:spatial"]] == [540]


def test_curator_rewrite_is_not_stamped_with_the_model():
    create = _load("07_create.py")
    row = {"model": "claude-haiku-5-5", "description_fr": "Texte revu.", "draft_fr": "Texte du modèle."}
    assert not create.is_model_text(row)
    assert create.is_model_text({**row, "draft_fr": "Texte revu."})


def test_plan_holds_back_what_may_exist_and_never_preapproves_merges():
    plan = _load("06_ner_plan.py")
    categories, countries = plan.label_maps([
        {"o:id": 546, "o:title": "Burkina Faso",
         "dcterms:isPartOf": [{"value_resource_id": 15835, "display_title": "Pays"}]},
        {"o:id": 901, "o:title": "Roch Marc Christian Kaboré",
         "dcterms:isPartOf": [{"value_resource_id": 8990, "display_title": "Politicien.ne"}]},
    ])
    assert countries == {"burkina faso": 546} and categories["politicien.ne"] == 8990
    new = [{"type": "person", "titre proposé": "Ousmane Bougouma", "alternatifs": "", "catégorie": "Politicien.ne",
            "pays": "Burkina Faso", "mentions": 59, "confiance": "high", "existe déjà ?": ""},
           {"type": "place", "titre proposé": "Turquie", "alternatifs": "", "catégorie": "Pays", "pays": "",
            "mentions": 16, "confiance": "high", "existe déjà ?": "545 Turquie (place)"}]
    rows = plan.create_rows(new, categories, countries)
    assert rows[0]["approve"] == "yes" and rows[0]["category_id"] == 8990 and rows[0]["country_id"] == 546
    assert rows[1]["approve"] == "" and rows[1]["status"].startswith("check")
    merges = plan.merge_rows([{"id A": 15171, "titre A": "Barkhane", "type A": "stub",
                               "id B": 65, "titre B": "Opération Barkhane", "type B": "subject",
                               "vraisemblance": "plausible", "note": ""}])
    assert merges[0]["survivor_id"] == 65 and merges[0]["loser_id"] == 15171 and merges[0]["approve"] == ""
