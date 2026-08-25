"""Tests for the generation-2 sentiment panel, its Omeka write path and cache.

Three things here are worth guarding, all of them lessons from generation 1:

- **Every value must land in a MODEL-KEYED property.** Generation 1's stored
  corpus could not say which model produced it, and recovering that took a dig
  through git history. The property name is what closes that hole — and it is
  the only thing that can, because Omeka does not index value annotations, so a
  model recorded only in an annotation is a model no query can find.
- **The panel must not collide with generation 1.** The two generations share
  an item and a vocabulary; a property term that overlapped would silently
  overwrite 12,286 articles' worth of existing annotations.
- **The cache must survive being killed.** It exists so a multi-day run can be
  interrupted, so a crash mid-write must cost one record, not the file.
"""

import json
import logging

import pytest

from common.iwac_config import AI_MODEL_ITEMS
from common.llm_provider import MODEL_REGISTRY, PROVIDER_SELFHOSTED
from sentiment_cache import CACHE_FORMAT_VERSION, SentimentCache
from sentiment_core import (
    CENTRALITE_ITEM_IDS,
    ITEM_ID_TO_SUBJECTIVITE,
    PANEL,
    PILOT_CANDIDATES,
    PANEL_REASONING_EFFECTIVE,
    POLARITE_ITEM_IDS,
    RESULT_FIELD_SUFFIXES,
    SENTIMENT_MODEL_ANNOTATION_TERM,
    SUBJECTIVITE_ITEM_IDS,
    request_timeout_for_budget,
)

# Imported from the script under its real name; `01_...` is not an identifier.
import importlib.util
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent / "AI_sentiment_analysis"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, _PIPELINE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sentiment_run = _load("sentiment_run", "01_sentiment_analysis.py")
pilot = _load("sentiment_pilot", "02_pilot_new_panel.py")
pilot_report = _load("sentiment_pilot_report", "03_pilot_report.py")

ANNOTATION_PID = 337
BASE_URL = "https://example.org/api"


@pytest.fixture
def property_ids():
    """A plausible id per term — the real ones are resolved from Omeka."""
    ids = {SENTIMENT_MODEL_ANNOTATION_TERM: ANNOTATION_PID}
    next_id = ANNOTATION_PID + 1
    for member in PANEL.values():
        for term in member.terms:
            ids[term] = next_id
            next_id += 1
    return ids


@pytest.fixture
def good_result():
    return {
        "centralite_islam_musulmans": "Central",
        "centralite_justification": "L'islam est un thème important de l'article.",
        "polarite": "Neutre",
        "polarite_justification": "Le ton reste factuel.",
        "subjectivite_score": "Plutôt objectif",
        "subjectivite_justification": "Peu d'opinions exprimées.",
        "analysis_error": None,
    }


# ---------------------------------------------------------------------------
# Panel definition
# ---------------------------------------------------------------------------

def test_every_member_has_an_authority_item():
    for key, member in PANEL.items():
        assert member.registry_key in AI_MODEL_ITEMS, (
            f"{key} annotates with registry key {member.registry_key!r}, which is "
            f"not in AI_MODEL_ITEMS — the annotation would raise at write time"
        )
        assert isinstance(member.model_item_id, int)


def test_panel_terms_are_unique():
    seen = {}
    for member in PANEL.values():
        for term in member.terms:
            assert term not in seen, f"{term} claimed by both {seen[term]} and {member.key}"
            seen[term] = member.key
    assert len(seen) == 6 * len(PANEL)


#: Property prefixes emptied and abandoned on 2026-08-07 — the vendor-keyed
#: generation-1 campaign and the retired April preview. No longer installed
#: anywhere, so they can no longer be *overwritten*; they are pinned because
#: reusing one would resurrect what generation 2 exists to prevent.
ABANDONED_PREFIXES = (
    "iwac:gemini", "iwac:chatgpt", "iwac:mistral", "iwac:deepseekV4Flash",
)


def test_panel_does_not_reuse_an_abandoned_property():
    """A vendor slot names a vendor, and a vendor is not a model.

    Generation 1 could not say which model produced a stored value, and
    recovering that took a dig through git history. Naming a member ``mistral``
    or ``gemini`` would generate exactly those terms again.

    Compared as exact terms, not by prefix: ``iwac:mistralSmall2603*`` does begin
    with ``iwac:mistral`` and ``iwac:deepseekV4Flash0731*`` with
    ``iwac:deepseekV4Flash``, and reading either as a collision would be a false
    alarm — both are live members whose terms are distinct from the slot's.
    """
    abandoned = {
        f"{prefix}{suffix}"
        for prefix in ABANDONED_PREFIXES
        for suffix in RESULT_FIELD_SUFFIXES.values()
    }
    active = {term for member in PANEL.values() for term in member.terms}

    assert len(abandoned) == 24
    assert not (abandoned & active), sorted(abandoned & active)


def test_every_member_has_a_declared_effective_reasoning_depth():
    # Candidates included: a pilot manifest has to record the depth its numbers
    # were produced at, exactly as a production manifest does.
    assert set(PANEL_REASONING_EFFECTIVE) == set(PANEL) | set(PILOT_CANDIDATES)
    # Mistral has no middle setting; recording it as "medium" would misreport
    # the run rather than merely simplify it.
    assert PANEL_REASONING_EFFECTIVE["mistral_small_2603"].startswith("high")
    assert PANEL_REASONING_EFFECTIVE["deepseek_v4_flash_0731"].startswith("high")
    # Qwen3.8's ladder has a real middle rung, so its "medium" is the requested
    # depth rather than a rounding of it. That is most of the reason it is being
    # piloted at all, and the manifest should not blur the distinction.
    assert PANEL_REASONING_EFFECTIVE["qwen3_8_27b"] == "medium"


# ---------------------------------------------------------------------------
# Pilot candidates
# ---------------------------------------------------------------------------

def test_pilot_candidates_cannot_be_written_to_omeka():
    """Staging is the safety property, and it is structural.

    ``01_sentiment_analysis.py`` iterates ``PANEL``, so a model absent from it
    cannot be selected however the script is invoked. Promotion is a deliberate
    edit that moves a member across — at which point
    ``test_every_member_has_an_authority_item`` starts demanding the Omeka
    record that makes the annotation citable.
    """
    assert not set(PILOT_CANDIDATES) & set(PANEL)
    for key in PILOT_CANDIDATES:
        assert key not in sentiment_run.PANEL, (
            f"{key} is reachable from the write path; it would annotate live items"
        )


def test_pilot_candidate_terms_collide_with_nothing():
    """A candidate's six properties must not shadow a live or abandoned slot.

    Checked before promotion rather than after, because the failure mode is
    silent: an overlapping term overwrites annotations instead of erroring.
    """
    live = {term for member in PANEL.values() for term in member.terms}
    abandoned = {
        f"{prefix}{suffix}"
        for prefix in ABANDONED_PREFIXES
        for suffix in RESULT_FIELD_SUFFIXES.values()
    }
    seen = {}
    for member in PILOT_CANDIDATES.values():
        for term in member.terms:
            assert term not in live, f"{term} is already a live panel property"
            assert term not in abandoned, f"{term} resurrects an abandoned slot"
            assert term not in seen, f"{term} claimed by both {seen[term]} and {member.key}"
            seen[term] = member.key


def test_the_two_qwen_routes_stay_on_opposite_sides_of_the_panel():
    """Two routes to one set of weights, told apart by registry key.

    The property prefix names the model and the registry key names the route —
    the split ``gemma_4_31b_it`` → ``gemma-4-openrouter`` already makes. Running
    both is what turns "self-hosting is cheaper" into something measured on one
    sample rather than assumed.

    Since 2026-08-25 the self-hosted route is a panel member and the OpenRouter
    twin is still a candidate, which is the arrangement this pins. Promoting the
    twin as well would put one model in the panel twice — a panel that counts
    one reading of the construct as two is measuring its own routing.
    """
    selfhosted = PANEL["qwen3_8_27b"]
    hosted = PILOT_CANDIDATES["qwen3_8_27b_openrouter"]

    assert selfhosted.registry_key == "qwen3.8-27b-selfhosted"
    assert hosted.registry_key == "qwen3.8-27b-openrouter"
    assert MODEL_REGISTRY[selfhosted.registry_key].provider == PROVIDER_SELFHOSTED
    # Same weights on both routes: the comparison is only meaningful if the
    # served model is the same one.
    assert MODEL_REGISTRY[hosted.registry_key].model.lower() \
        == MODEL_REGISTRY[selfhosted.registry_key].model.lower()
    # ...which is exactly why only one of them may be writable.
    assert "qwen3_8_27b_openrouter" not in PANEL
    assert selfhosted.property_prefix != hosted.property_prefix


def test_the_pilot_runs_candidates_alongside_the_live_panel():
    """Agreement is only interesting against the annotators already in use, so
    one payload has to carry both halves."""
    assert set(pilot.V2_PANEL) == set(PANEL) | set(PILOT_CANDIDATES)
    assert pilot.selected_models("qwen3_8_27b") == ["qwen3_8_27b"]


def test_retry_attempts_fit_inside_model_timeout():
    per_attempt = request_timeout_for_budget(120)
    assert per_attempt == pytest.approx(112 / 3)
    assert per_attempt * 3 + 1 + 2 + 5 == pytest.approx(120)


def test_pilot_model_subset_validation():
    assert pilot.selected_models("deepseek_v4_flash_0731, gemma_4_31b_it") == [
        "deepseek_v4_flash_0731",
        "gemma_4_31b_it",
    ]
    with pytest.raises(ValueError, match="Unknown model"):
        pilot.selected_models("not_a_model")
    with pytest.raises(ValueError, match="No models"):
        pilot.selected_models(",")


def test_pilot_payload_records_exact_deepseek_model_and_prompt():
    models = pilot.PilotModels(
        clients={"deepseek_v4_flash_0731": object()},
        labels={"deepseek_v4_flash_0731": "DeepSeek V4 Flash 0731"},
        model_ids={
            "deepseek_v4_flash_0731": "deepseek/deepseek-v4-flash-0731",
        },
        skipped=[],
    )

    payload = pilot.build_payload(
        timestamp="20260801T000000Z",
        articles=[{"o:id": 1}],
        results={"1": {"v2_runs": []}},
        models=models,
        seed=42,
        repeats=1,
        system_prompt="prompt text",
        prompt_id="abc123",
    )

    manifest = payload["manifest"]
    assert manifest["v2_models"]["deepseek_v4_flash_0731"]["model_id"] \
        == "deepseek/deepseek-v4-flash-0731"
    assert manifest["prompt_fingerprint"] == "abc123"


def test_pilot_report_selects_newest_timestamped_file(tmp_path, monkeypatch):
    older = tmp_path / "pilot_20260701.json"
    newer = tmp_path / "pilot_20260801.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pilot_report, "PILOT_DIR", tmp_path)

    assert pilot_report.resolve_pilot_path(None) == newer


def test_pilot_report_majority_requires_more_than_half():
    assert pilot_report.majority(["positive", "positive", "negative"]) == "positive"
    assert pilot_report.majority(["positive", "negative"]) is None
    assert pilot_report.majority(["positive", None]) is None


# ---------------------------------------------------------------------------
# Building Omeka values
# ---------------------------------------------------------------------------

def test_no_value_carries_a_provenance_annotation(property_ids, good_result):
    """Dropped 2026-07-31, and the reason is worth pinning.

    Omeka S does not index value annotations: verified live, a query for
    ``iwac:sentimentModel = 79613`` returned 0 items while 498 carried exactly
    that annotation, and ``/api/value_annotations`` is a 500. So the annotation
    could not be searched, filtered or aggregated — it was an unreachable second
    copy of what the property name already states, written six times per model
    per item. Generation 1's hole is closed by the model-keyed property names,
    which ARE queryable (see the two tests above).
    """
    member = PANEL["gemma_4_31b_it"]
    values = sentiment_run.build_property_values(
        member, good_result, property_ids
    )

    assert len(values) == 6
    for term, [value] in values.items():
        assert "@annotation" not in value, (
            f"{term} carries an @annotation; Omeka cannot query it, so it is "
            f"payload with no retrieval value"
        )


def test_controlled_vocabulary_fields_are_resource_links(property_ids, good_result):
    member = PANEL["gpt_5_6_luna"]
    values = sentiment_run.build_property_values(
        member, good_result, property_ids
    )

    assert values[member.term("centralite_islam_musulmans")][0]["value_resource_id"] \
        == CENTRALITE_ITEM_IDS["Central"]
    assert values[member.term("polarite")][0]["value_resource_id"] \
        == POLARITE_ITEM_IDS["Neutre"]
    assert values[member.term("subjectivite_score")][0]["value_resource_id"] \
        == SUBJECTIVITE_ITEM_IDS["Plutôt objectif"]
    # Justifications are French literals, not links.
    just = values[member.term("polarite_justification")][0]
    assert just["type"] == "literal"
    assert just["@language"] == "fr"


def test_unmappable_and_blank_fields_are_omitted(property_ids):
    """A property that is absent can be filled in later; one written blank or
    wrong looks like a real annotation."""
    member = PANEL["deepseek_v4_flash_0731"]
    partial = {
        "centralite_islam_musulmans": "Central",
        "centralite_justification": "   ",           # whitespace only
        "polarite": "ERREUR_ANALYSE",                # not in the vocabulary
        "polarite_justification": "",
        "subjectivite_score": None,                  # model declined to score
        "subjectivite_justification": "Sujet non abordé.",
    }
    values = sentiment_run.build_property_values(
        member, partial, property_ids
    )

    assert member.term("centralite_islam_musulmans") in values
    assert member.term("subjectivite_justification") in values
    for absent in ("centralite_justification", "polarite",
                   "polarite_justification", "subjectivite_score"):
        assert member.term(absent) not in values, absent


def test_write_path_preserves_unrelated_properties(monkeypatch, property_ids, good_result):
    """PATCH sends the whole item; anything dropped is deleted by Omeka."""
    member = PANEL["deepseek_v4_flash_0731"]
    item = {
        "o:id": 42,
        "dcterms:title": [{"@value": "Un titre", "property_id": 1}],
        "bibo:content": [{"@value": "x" * 5000, "property_id": 91}],
        "iwac:geminiCentralite": [{"value_resource_id": 78049, "property_id": 319}],
    }
    patched = {}

    class FakeClient:
        base_url = BASE_URL

        def get_item(self, item_id):
            return dict(item)

        def update_item(self, item_id, data):
            patched.update(data)
            return True

    status = sentiment_run.update_item_sentiment(
        FakeClient(), 42, {member.key: good_result}, property_ids,
    )

    assert status == "updated"
    assert patched["dcterms:title"] == item["dcterms:title"]
    assert patched["bibo:content"] == item["bibo:content"]
    # Generation 1 must come back out exactly as it went in.
    assert patched["iwac:geminiCentralite"] == item["iwac:geminiCentralite"]
    assert patched[member.probe_term][0]["value_resource_id"] == CENTRALITE_ITEM_IDS["Central"]


def test_rewriting_identical_values_is_a_no_op(property_ids, good_result):
    """Skipping unchanged items is what makes a resume cheap."""
    member = PANEL["gemma_4_31b_it"]
    values = sentiment_run.build_property_values(
        member, good_result, property_ids
    )
    stored = {"o:id": 7, **values}

    class FakeClient:
        base_url = BASE_URL

        def get_item(self, item_id):
            return dict(stored)

        def update_item(self, item_id, data):  # pragma: no cover - must not run
            raise AssertionError("PATCHed an item that had not changed")

    assert sentiment_run.update_item_sentiment(
        FakeClient(), 7, {member.key: good_result}, property_ids,
    ) == "unchanged"


def test_errored_results_are_never_written(property_ids):
    member = PANEL["mistral_small_2603"]
    errored = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "polarite": "ERREUR_ANALYSE",
        "analysis_error": "RateLimitError: 429",
    }

    class FakeClient:
        base_url = BASE_URL

        def get_item(self, item_id):
            return {"o:id": 9}

        def update_item(self, item_id, data):  # pragma: no cover - must not run
            raise AssertionError("wrote an errored result to Omeka")

    assert sentiment_run.update_item_sentiment(
        FakeClient(), 9, {member.key: errored}, property_ids,
    ) == "unchanged"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_round_trips(tmp_path, good_result):
    path = tmp_path / "c.jsonl"
    with SentimentCache(path=path) as cache:
        cache.put(123, "gemma_4_31b_it", good_result, model_id="google/gemma-4-31b-it")

    reloaded = SentimentCache(path=path)
    report = reloaded.load()
    assert report.records == 1
    assert reloaded.get(123, "gemma_4_31b_it") == good_result
    # Item ids are normalised, so an int and a str reach the same record.
    assert reloaded.has("123", "gemma_4_31b_it")


def test_cache_is_granular_to_the_model(tmp_path, good_result):
    """One model failing must not force re-running the rest of the panel."""
    path = tmp_path / "c.jsonl"
    with SentimentCache(path=path) as cache:
        for key in ("gemma_4_31b_it", "gpt_5_6_luna"):
            cache.put(1, key, good_result)

    reloaded = SentimentCache(path=path)
    reloaded.load()
    assert reloaded.missing_models(1, PANEL) == [
        "mistral_small_2603", "deepseek_v4_flash_0731", "qwen3_8_27b",
    ]


def test_cache_survives_a_torn_final_line(tmp_path, good_result):
    """A process killed mid-write costs one record, not the run."""
    path = tmp_path / "c.jsonl"
    with SentimentCache(path=path) as cache:
        cache.put(1, "gemma_4_31b_it", good_result)
        cache.put(2, "gpt_5_6_luna", good_result)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"v": 2, "item_id": "3", "model": "deepseek_v4_flash_0731", "resu')

    reloaded = SentimentCache(path=path)
    report = reloaded.load()
    assert report.records == 2
    assert report.skipped_malformed == 1
    assert report.torn_at is not None
    assert reloaded.get(1, "gemma_4_31b_it") == good_result
    assert reloaded.get(2, "gpt_5_6_luna") == good_result


def test_cache_last_record_wins(tmp_path, good_result):
    path = tmp_path / "c.jsonl"
    revised = {**good_result, "polarite": "Positif"}
    with SentimentCache(path=path) as cache:
        cache.put(1, "gemma_4_31b_it", good_result)
        cache.put(1, "gemma_4_31b_it", revised)

    reloaded = SentimentCache(path=path)
    reloaded.load()
    assert reloaded.get(1, "gemma_4_31b_it")["polarite"] == "Positif"
    # Both lines are kept: the file doubles as an audit trail.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_cache_ignores_records_from_another_format_version(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({
        "v": CACHE_FORMAT_VERSION - 1, "item_id": "1",
        "model": "gemma_4_31b_it", "result": {"polarite": "Neutre"},
    }) + "\n", encoding="utf-8")

    cache = SentimentCache(path=path)
    report = cache.load()
    assert report.records == 0
    assert report.skipped_version == 1
    assert cache.get(1, "gemma_4_31b_it") is None


def test_cache_records_the_model_that_answered(tmp_path, good_result):
    """The gap that made generation 1 unattributable."""
    path = tmp_path / "c.jsonl"
    with SentimentCache(path=path) as cache:
        cache.put(1, "gemma_4_31b_it", good_result,
                  model_id="google/gemma-4-31b-it", reasoning="medium")

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["model_id"] == "google/gemma-4-31b-it"
    assert record["reasoning"] == "medium"
    assert record["ts"].endswith("+00:00")


def test_cache_provenance_is_part_of_reuse_identity(tmp_path, good_result):
    """A new snapshot or prompt must miss even when the panel slot is unchanged."""
    path = tmp_path / "c.jsonl"
    with SentimentCache(path=path) as cache:
        cache.put(
            1,
            "deepseek_v4_flash_0731",
            good_result,
            model_id="deepseek/deepseek-v4-flash-0731",
            reasoning="high",
            prompt="prompt-a",
        )

    cache = SentimentCache(path=path)
    cache.load()
    expected = {
        "model_id": "deepseek/deepseek-v4-flash-0731",
        "reasoning": "high",
        "prompt": "prompt-a",
    }
    assert cache.has(1, "deepseek_v4_flash_0731", **expected)
    assert not cache.has(1, "deepseek_v4_flash_0731", **{**expected, "prompt": "prompt-b"})
    assert not cache.has(
        1,
        "deepseek_v4_flash_0731",
        **{**expected, "model_id": "deepseek/deepseek-v4-flash"},
    )
    assert cache.results_for(
        1, expected={"deepseek_v4_flash_0731": expected}
    ) == {"deepseek_v4_flash_0731": good_result}
    assert cache.count_matching({"deepseek_v4_flash_0731": expected}) == 1


def test_missing_cache_file_is_not_an_error(tmp_path):
    report = SentimentCache(path=tmp_path / "nope.jsonl").load()
    assert report.records == 0


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------

def test_already_written_models_are_detected():
    members = list(PANEL.values())
    item = {
        "o:id": 1,
        members[0].probe_term: [{"value_resource_id": 78049}],
        members[2].probe_term: [{"value_resource_id": 78050}],
        # An empty list is Omeka's "property absent", not "annotated".
        members[1].probe_term: [],
    }
    assert sentiment_run.models_already_written(item, members) == [
        members[0].key, members[2].key,
    ]


def test_a_scoped_run_leaves_other_members_untouched(property_ids, good_result):
    """Running the panel one model at a time must be additive.

    An item already annotated by Gemma gets DeepSeek's six properties added and
    nothing else disturbed — including DeepSeek's own neighbours in the same
    vocabulary.
    """
    deepseek = PANEL["deepseek_v4_flash_0731"]
    gemma = PANEL["gemma_4_31b_it"]
    existing = sentiment_run.build_property_values(
        gemma, good_result, property_ids
    )
    item = {"o:id": 5, "dcterms:title": [{"@value": "T"}], **existing}
    patched = {}

    class FakeClient:
        base_url = BASE_URL

        def get_item(self, item_id):
            return dict(item)

        def update_item(self, item_id, data):
            patched.update(data)
            return True

    status = sentiment_run.update_item_sentiment(
        FakeClient(), 5, {deepseek.key: good_result}, property_ids,
    )

    assert status == "updated"
    for term in gemma.terms:
        assert patched[term] == existing[term], f"scoped run disturbed {term}"
    for term in deepseek.terms:
        assert term in patched
    assert (patched[deepseek.probe_term][0]["value_resource_id"]
            == CENTRALITE_ITEM_IDS["Central"])


def test_subjectivite_is_a_label_not_a_number():
    """The 2026-07-31 change, and the reason the cache format was bumped.

    Subjectivité was the one dimension asked for as an integer and by a
    distance the least reliable (pilot pairwise kappa 0.093-0.470). A stale
    integer must not quietly map to an item — it should produce nothing, which
    is what the version bump exists to make impossible in the first place.
    """
    from sentiment_core import SUBJECTIVITE_LABELS, SUBJECTIVITE_ORDER

    assert set(SUBJECTIVITE_ITEM_IDS) == set(SUBJECTIVITE_LABELS)
    assert SUBJECTIVITE_ORDER["Très objectif"] == 1
    assert SUBJECTIVITE_ORDER["Très subjectif"] == 5
    # Generation-1 links read back as labels, so both generations compare on
    # one scale rather than one being ints and the other strings.
    assert ITEM_ID_TO_SUBJECTIVITE[78043] == "Très objectif"
    for legacy_int in range(1, 6):
        assert SUBJECTIVITE_ITEM_IDS.get(legacy_int) is None


def _sentiment_base(centralite="Central"):
    return dict(
        centralite_islam_musulmans=centralite, centralite_justification="x",
        polarite="Neutre", polarite_justification="x",
        subjectivite_justification="x",
    )


def test_schema_rejects_a_numeric_subjectivite():
    from pydantic import ValidationError
    from sentiment_core import SentimentAnalysisOutput

    base = _sentiment_base()
    assert SentimentAnalysisOutput(subjectivite_score="Mixte", **base)
    with pytest.raises(ValidationError):
        SentimentAnalysisOutput(subjectivite_score=3, **base)


def test_null_subjectivite_allowed_only_when_islam_is_not_addressed():
    """A schema-compliant non-answer is still a non-answer.

    ``subjectivite_score`` is nullable so ``Non abordé`` articles have somewhere
    to go, which means provider-side ``strict`` validation accepts a null on any
    article. DeepSeek V4 Flash 0731 returned one on 1,485 articles it had itself
    marked as discussing Islam (2026-08), and because
    ``build_property_values`` omits missing fields, those items were written with
    four properties instead of six and the run reported success.
    """
    from pydantic import ValidationError
    from sentiment_core import SentimentAnalysisOutput

    # The one case a null is a real answer.
    assert SentimentAnalysisOutput(
        subjectivite_score=None, **_sentiment_base("Non abordé")
    )

    for centralite in ("Très central", "Central", "Secondaire", "Marginal"):
        with pytest.raises(ValidationError):
            SentimentAnalysisOutput(
                subjectivite_score=None, **_sentiment_base(centralite)
            )

    # A real label is always fine, including on Non abordé.
    for centralite in ("Non abordé", "Marginal", "Très central"):
        assert SentimentAnalysisOutput(
            subjectivite_score="Plutôt objectif", **_sentiment_base(centralite)
        )


def test_parse_item_ids_accepts_a_list_or_a_file(tmp_path):
    """A repair set is thousands of ids, which does not fit on a command line."""
    parse = sentiment_run.parse_item_ids

    assert parse("2233,2234") == [2233, 2234]
    assert parse(" 2233 , 2234 ") == [2233, 2234]

    listing = tmp_path / "ids.txt"
    listing.write_text("2233\n2234\n2235\n", encoding="utf-8")
    assert parse(f"@{listing}") == [2233, 2234, 2235]


def test_parse_item_ids_rejects_what_would_silently_misfire():
    parse = sentiment_run.parse_item_ids

    # A duplicate would be annotated twice, the second write a no-op PATCH.
    assert parse("2233,2233,2234") == [2233, 2234]

    with pytest.raises(ValueError):
        parse("")
    with pytest.raises(ValueError):
        parse("2233,not-an-id")
    with pytest.raises(ValueError):
        parse("@/no/such/file/anywhere.txt")


def test_item_ids_may_not_be_combined_with_a_listing():
    """--item-ids replaces the listing rather than filtering it.

    Accepting both would silently ignore one of them, and the mistake is only
    visible after a full corpus walk.
    """
    parser = sentiment_run.build_argument_parser()
    args = parser.parse_args(["--item-ids", "2233", "--resource-class-id", "36"])
    with pytest.raises(ValueError):
        sentiment_run.validate_arguments(args)

    ok = parser.parse_args(["--item-ids", "2233"])
    assert sentiment_run.validate_arguments(ok) == []


def test_validator_does_not_change_the_wire_schema():
    """The provider contract and the prompt fingerprint must be untouched.

    A cross-field rule lives in Python, not in JSON Schema, so the body sent to
    the model is byte-identical — the 12,305 values already stored under
    fingerprint d14ace9ac192 stay comparable with anything produced after it.
    """
    from sentiment_core import SentimentAnalysisOutput

    schema = SentimentAnalysisOutput.model_json_schema()
    subjectivite = json.dumps(schema["properties"]["subjectivite_score"])
    assert "null" in subjectivite, "field must stay nullable on the wire"


def test_prompt_carries_its_load_bearing_sections():
    """The prompt is the instrument; a silent truncation would not be visible.

    Worked examples were removed on 2026-08-03 after the A/B pilot measured them
    anchoring the label distribution onto the labels they demonstrated. What is
    left is definitions plus boundary rules, and every one of these headings is
    depended on by a field of ``SentimentAnalysisOutput``.
    """
    from sentiment_core import load_system_prompt, prompt_fingerprint

    prompt = load_system_prompt()

    for rule in ("## Centralité", "## Subjectivité", "## Polarité",
                 "Coopération avec les pays arabes", "## Cohérence"):
        assert rule in prompt, rule
    assert prompt_fingerprint(prompt) == prompt_fingerprint()
    assert prompt_fingerprint(prompt) != prompt_fingerprint(prompt + "\n")


def test_pilot_resumes_from_its_partial_file(tmp_path):
    """A 50x3 pilot is hours of calls; writing only at the end lost all of them.

    The prompt check is the load-bearing part: resuming across a prompt edit
    would silently mix two instruments inside one pilot, which is exactly what
    a pilot exists to rule out.
    """
    logger = logging.getLogger("test")
    path = tmp_path / "p.partial.jsonl"
    path.write_text("\n".join([
        json.dumps({"prompt": "abc", "item_id": "1", "run": 0,
                    "result": {"deepseek_v4_flash_0731": {"polarite": "Neutre"}}}),
        json.dumps({"prompt": "abc", "item_id": "1", "run": 1,
                    "result": {"deepseek_v4_flash_0731": {"polarite": "Positif"}}}),
        json.dumps({"prompt": "DIFFERENT", "item_id": "2", "run": 0, "result": {}}),
        '{"prompt": "abc", "item_id": "3", "ru',  # torn by a kill mid-write
    ]), encoding="utf-8")

    done = pilot.load_partial(path, "abc", logger)

    assert sorted(done["1"]) == [0, 1]
    assert done["1"][1]["deepseek_v4_flash_0731"]["polarite"] == "Positif"
    assert "2" not in done, "reused a repeat produced by a different prompt"
    assert "3" not in done, "torn line should be skipped, not fatal"
    assert pilot.load_partial(tmp_path / "absent.jsonl", "abc", logger) == {}


def test_prompt_fingerprint_tracks_the_prompt():
    """Provenance for the prompt, which nothing else in a stored record captures."""
    from sentiment_core import prompt_fingerprint

    live = prompt_fingerprint()
    assert len(live) == 12
    assert live == prompt_fingerprint(), "fingerprint must be stable"
    assert prompt_fingerprint("something else") != live


def test_language_gate_accepts_french_and_english_only():
    """A French-prompted model scores an Ewé article confidently and wrongly,
    and the result is indistinguishable from a real annotation once stored."""
    from sentiment_core import ANALYSABLE_LANGUAGES, get_item_language

    def item(label):
        return {"dcterms:language": [
            {"type": "resource:item", "value_resource_id": 1, "display_title": label}
        ]}

    assert get_item_language(item("Français")) in ANALYSABLE_LANGUAGES
    assert get_item_language(item("Anglais")) in ANALYSABLE_LANGUAGES
    for dropped in ("Ewé", "Kabyè", "Dendi"):
        assert get_item_language(item(dropped)) not in ANALYSABLE_LANGUAGES
    # 6 of 12,356 articles carry no language value; untagged must stay
    # distinguishable from "tagged with a language we skip".
    assert get_item_language({}) is None
    assert get_item_language({"dcterms:language": []}) is None


def test_result_field_suffixes_cover_the_schema():
    """A schema field with no property would be analysed and silently dropped."""
    from sentiment_core import SentimentAnalysisOutput

    assert set(RESULT_FIELD_SUFFIXES) == set(SentimentAnalysisOutput.model_fields)


# ---------------------------------------------------------------------------
# Terminal provider errors
# ---------------------------------------------------------------------------

class _Status402(Exception):
    """Shaped like an OpenAI-SDK APIStatusError, which is what OpenRouter raises."""
    status_code = 402

    def __str__(self):
        return ("Error code: 402 - {'error': {'message': 'Insufficient credits. "
                "Add more using https://openrouter.ai/settings/credits', 'code': 402}}")


class _Status429Transient(Exception):
    code = 429

    def __str__(self):
        return "429 Too Many Requests: rate limit exceeded, please slow down"


def test_a_dead_account_stops_the_run_instead_of_being_retried():
    """A 402 cost a real run 823 identical failures and several hours.

    ``analyze_with_model`` retried every terminal error three times with
    backoff, so an OpenRouter balance that ran dry at article ~11,500 produced
    2,469 pointless requests and walked the rest of the corpus collecting the
    same message. A 402 is never transient and must raise on the first hit.
    """
    from common.rate_limiter import QuotaExhaustedError, is_quota_exhausted
    from sentiment_core import analyze_with_model

    assert is_quota_exhausted(_Status402())
    assert not is_quota_exhausted(_Status429Transient()), (
        "a transient per-minute 429 must still be retried, not treated as terminal"
    )

    calls = []

    class DeadAccount:
        def generate_structured(self, *a, **kw):
            calls.append(1)
            raise _Status402()

    with pytest.raises(QuotaExhaustedError):
        analyze_with_model(DeadAccount(), "texte", "prompt", "DeepSeek",
                           logging.getLogger("test"))
    assert len(calls) == 1, f"retried a 402 {len(calls)} times; it is never transient"


def test_a_transient_error_is_still_retried():
    """The guard above must not turn every failure into a full stop."""
    from sentiment_core import analyze_with_model

    calls = []

    class Flaky:
        def generate_structured(self, *a, **kw):
            calls.append(1)
            raise _Status429Transient()

    result = analyze_with_model(Flaky(), "texte", "prompt", "DeepSeek",
                                logging.getLogger("test"), max_retries=3)
    assert len(calls) == 3
    assert result["analysis_error"]
