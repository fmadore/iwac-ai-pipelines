"""Tests for the value annotation written by common.omeka_text_updater.

These guard a real regression. OmekaClient.upsert_property_value() has two paths:
it mutates an existing literal in place (which keeps a sibling ``@annotation``), or
it APPENDS a freshly built value object made only of
type/property_id/property_label/is_public/@value — with no ``@annotation``.

The append path is the dangerous one: writing a summary to a property that has no
literal yet produces an un-annotated value, silently losing the record of which AI
model produced it. That is how six articles lost their iwac:summaryModel provenance
when their summaries were moved from dcterms:abstract to bibo:shortDescription.

The logic now lives in ``common/omeka_text_updater.apply_text_value`` and is shared
by the summary, OCR, OCR-correction and transcription updaters, so these tests cover
all four rather than just AI_summary/03.
"""

import pytest

from common.iwac_config import (
    AI_MODEL_ITEMS,
    IWAC_SUMMARY_MODEL_PROPERTY_ID,
    RETIRED_AI_MODEL_ITEM_IDS,
    model_annotation_value,
)
from common.omeka_text_updater import (
    PropertyTarget,
    apply_text_value,
    apply_text_values,
)

SUMMARY_PID = 116
BASE_URL = "https://example.org/api"


@pytest.fixture
def model_value():
    return model_annotation_value(
        BASE_URL, "gpt-5.6-luna", IWAC_SUMMARY_MODEL_PROPERTY_ID, "AI Model - Summary"
    )


@pytest.fixture
def target(model_value):
    return PropertyTarget(
        term="bibo:shortDescription",
        property_id=SUMMARY_PID,
        property_label="shortDescription",
        annotation_term="iwac:summaryModel",
        annotation_value=model_value,
    )


def _annotation(item_data):
    return item_data["bibo:shortDescription"][0].get("@annotation")


def test_writes_summary_with_model_annotation(target):
    """The core guard: the APPEND path must still produce an annotated value."""
    item = {"o:id": 1}
    assert apply_text_value(item, target, "Résumé.") is True

    value = item["bibo:shortDescription"][0]
    assert value["@value"] == "Résumé."
    assert value["property_id"] == SUMMARY_PID
    linked = _annotation(item)["iwac:summaryModel"][0]
    assert linked["value_resource_id"] == AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"]
    assert linked["property_id"] == IWAC_SUMMARY_MODEL_PROPERTY_ID


def test_preserves_other_properties(target):
    # bibo:content is 64KB of OCR on real items — it must survive untouched.
    item = {"o:id": 1, "bibo:content": [{"@value": "ocr", "property_id": 91, "type": "literal"}]}
    apply_text_value(item, target, "Résumé.")
    assert item["bibo:content"] == [{"@value": "ocr", "property_id": 91, "type": "literal"}]


def test_idempotent_when_text_and_model_unchanged(target):
    item = {"o:id": 1}
    assert apply_text_value(item, target, "Résumé.") is True
    # Second pass must report no change so the caller skips a needless PATCH.
    assert apply_text_value(item, target, "Résumé.") is False


def test_server_added_annotation_keys_are_not_a_difference(target, model_value):
    """The regression that re-PATCHed a whole corpus.

    Omeka echoes a ``resource:item`` annotation back with ``"url": null`` — a key
    no client sends. An exact dict comparison therefore never matched a value
    that had just been written, so every annotated item reported as changed and
    the unchanged-skip never fired: a resumed 12,305-article summary run wrote
    the entire corpus again instead of only its remainder.
    """
    item = {"o:id": 1}
    apply_text_value(item, target, "Résumé.")

    # Simulate the round trip: Omeka returns what we sent, plus its own keys.
    stored = item["bibo:shortDescription"][0]["@annotation"]["iwac:summaryModel"][0]
    stored["url"] = None
    stored["thumbnail_display_urls"] = {"square": None}

    assert apply_text_value(item, target, "Résumé.") is False, (
        "a server-added key must not count as a change"
    )


def test_a_genuinely_different_model_still_counts_as_a_change(target, model_value):
    """The loosened comparison must not stop noticing a real difference."""
    item = {"o:id": 1}
    apply_text_value(item, target, "Résumé.")
    stored = item["bibo:shortDescription"][0]["@annotation"]["iwac:summaryModel"][0]
    stored["url"] = None
    stored["value_resource_id"] = 78053  # annotated by a different model

    assert apply_text_value(item, target, "Résumé.") is True
    assert (_annotation(item)["iwac:summaryModel"][0]["value_resource_id"]
            == AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"])


def test_missing_annotation_is_still_a_change(target):
    item = {
        "o:id": 1,
        "bibo:shortDescription": [{
            "@value": "Résumé.", "property_id": SUMMARY_PID,
            "type": "literal", "@language": "fr",
        }],
    }
    assert apply_text_value(item, target, "Résumé.") is True


def test_refreshes_a_stale_model_annotation(target):
    item = {
        "o:id": 1,
        "bibo:shortDescription": [{
            "@value": "Résumé.",
            "property_id": SUMMARY_PID,
            "type": "literal",
            "@annotation": {"iwac:summaryModel": [
                {"type": "resource:item", "property_id": IWAC_SUMMARY_MODEL_PROPERTY_ID,
                 "value_resource_id": 78053, "display_title": "Gemini 3.0 flash"}
            ]},
        }],
    }
    assert apply_text_value(item, target, "Résumé.") is True
    assert (_annotation(item)["iwac:summaryModel"][0]["value_resource_id"]
            == AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"])


def test_annotation_survives_a_text_rewrite(target):
    """The in-place path: replacing the text must keep exactly one annotated value."""
    item = {"o:id": 1}
    apply_text_value(item, target, "Ancien résumé.")
    assert apply_text_value(item, target, "Nouveau résumé.") is True

    values = item["bibo:shortDescription"]
    assert len(values) == 1, "rewrite should replace the literal, not append a second one"
    assert values[0]["@value"] == "Nouveau résumé."
    assert _annotation(item) is not None, "annotation was dropped by the rewrite"


# ---------------------------------------------------------------------------
# Bilingual writes: two language-tagged literals on one property
# ---------------------------------------------------------------------------

@pytest.fixture
def french(model_value):
    return PropertyTarget(
        term="bibo:shortDescription", property_id=SUMMARY_PID,
        property_label="shortDescription",
        annotation_term="iwac:summaryModel", annotation_value=model_value,
        language="fr", adopt_untagged=True,
    )


@pytest.fixture
def english(model_value):
    return PropertyTarget(
        term="bibo:shortDescription", property_id=SUMMARY_PID,
        property_label="shortDescription",
        annotation_term="iwac:summaryModel", annotation_value=model_value,
        language="en",
    )


def _by_language(item):
    return {
        value.get("@language"): value["@value"]
        for value in item["bibo:shortDescription"]
    }


def test_writes_one_literal_per_language(french, english):
    """The core bilingual guard.

    ``upsert_property_value`` matches the first literal on a property whatever
    its language, so routing both writes through it would make the English
    summary overwrite the French one.
    """
    item = {"o:id": 1}
    assert apply_text_values(item, [(french, "Résumé."), (english, "Summary.")]) is True

    assert _by_language(item) == {"fr": "Résumé.", "en": "Summary."}
    for value in item["bibo:shortDescription"]:
        linked = value["@annotation"]["iwac:summaryModel"][0]
        assert linked["value_resource_id"] == AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"]


def test_rewrite_replaces_each_language_in_place(french, english):
    item = {"o:id": 1}
    apply_text_values(item, [(french, "Ancien."), (english, "Old.")])
    assert apply_text_values(item, [(french, "Nouveau."), (english, "New.")]) is True

    assert len(item["bibo:shortDescription"]) == 2, "rewrite appended instead of replacing"
    assert _by_language(item) == {"fr": "Nouveau.", "en": "New."}


def test_bilingual_write_is_idempotent(french, english):
    item = {"o:id": 1}
    assert apply_text_values(item, [(french, "Résumé."), (english, "Summary.")]) is True
    assert apply_text_values(item, [(french, "Résumé."), (english, "Summary.")]) is False


def test_french_adopts_the_untagged_legacy_summary(french, english):
    """~12,300 articles carry a French summary written before the tag existed.

    Appending a second, ``fr``-tagged French value beside the untagged one would
    give the item two French summaries and pipe-join them in the HF export.
    """
    item = {
        "o:id": 1,
        "bibo:shortDescription": [{
            "@value": "Ancien résumé.",
            "property_id": SUMMARY_PID,
            "property_label": "shortDescription",
            "type": "literal",
            "is_public": True,
        }],
    }
    assert apply_text_values(item, [(french, "Résumé."), (english, "Summary.")]) is True

    assert len(item["bibo:shortDescription"]) == 2
    assert _by_language(item) == {"fr": "Résumé.", "en": "Summary."}


def test_english_never_claims_an_untagged_value(english):
    """Only the language that owns the legacy values may adopt them.

    Without ``adopt_untagged=False`` on the English target, an English-first
    write would overwrite a French summary and label it ``en``.
    """
    item = {
        "o:id": 1,
        "bibo:shortDescription": [{
            "@value": "Ancien résumé.",
            "property_id": SUMMARY_PID,
            "type": "literal",
        }],
    }
    assert apply_text_value(item, english, "Summary.") is True

    assert _by_language(item) == {None: "Ancien résumé.", "en": "Summary."}


def test_empty_half_never_blanks_a_stored_value(french, english):
    """A missing translation must leave Omeka's value alone, not erase it."""
    item = {"o:id": 1}
    apply_text_values(item, [(french, "Résumé."), (english, "Summary.")])

    assert apply_text_values(item, [(french, "Résumé."), (english, "   ")]) is False
    assert _by_language(item) == {"fr": "Résumé.", "en": "Summary."}


def test_language_blind_target_is_unchanged_by_the_bilingual_support():
    """OCR, correction and transcription pass no language — first literal wins.

    Pinned because ``apply_text_value`` no longer delegates to
    ``upsert_property_value``; these three pipelines must keep the exact
    behaviour they had when it did.
    """
    plain = PropertyTarget(term="bibo:content", property_id=91, property_label="content")
    item = {
        "o:id": 1,
        "bibo:content": [
            {"@value": "first", "property_id": 91, "type": "literal", "@language": "fr"},
            {"@value": "second", "property_id": 91, "type": "literal"},
        ],
    }
    assert apply_text_value(item, plain, "corrected") is True

    assert [v["@value"] for v in item["bibo:content"]] == ["corrected", "second"]
    assert item["bibo:content"][0]["@language"] == "fr", "language-blind write retagged a value"


# ---------------------------------------------------------------------------
# Targets without provenance (OCR correction rewrites text another model OCR'd)
# ---------------------------------------------------------------------------

def test_unannotated_target_preserves_an_existing_annotation():
    """Correcting OCR text must not erase who did the original OCR.

    upsert_property_value mutates the existing literal in place, so a sibling
    ``@annotation`` survives — this pins that behaviour for the correction
    pipeline, which deliberately writes no annotation of its own.
    """
    ocr_provenance = {"iwac:ocrModel": [{"value_resource_id": 79611}]}
    item = {
        "o:id": 1,
        "bibo:content": [{
            "@value": "raw ocr",
            "property_id": 91,
            "type": "literal",
            "@annotation": ocr_provenance,
        }],
    }
    plain = PropertyTarget(term="bibo:content", property_id=91, property_label="content")

    assert apply_text_value(item, plain, "corrected ocr") is True
    assert item["bibo:content"][0]["@value"] == "corrected ocr"
    assert item["bibo:content"][0]["@annotation"] == ocr_provenance


def test_registry_points_at_the_current_authority_items():
    """Pins the 2026-07-31 correction: 79609 was deleted upstream and 79608 was
    a duplicate, so both had to be repointed. An annotation aimed at a dead item
    is worse than none — it looks like provenance and resolves to nothing."""
    assert AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"] == 79610
    assert AI_MODEL_ITEMS["gemini-3.6-flash"]["item_id"] == 79611
    assert AI_MODEL_ITEMS["gemini-3.5-flash-lite"]["item_id"] == 79617
    assert AI_MODEL_ITEMS["deepseek-v4-flash-0731"]["item_id"] == 83261


def test_no_registry_key_is_a_rolling_alias():
    """A model whose version cannot be stated cannot be cited.

    ``gemini-flash-lite`` was in this registry claiming item 78631, "Gemini 3.1
    flash lite", while resolving to ``gemini-flash-lite-latest`` — which pointed
    at 3.1 when the entry was written and at 3.5 by 2026-07-31. Every annotation
    written through it after 3.5 shipped names the wrong model. The same bug was
    fixed for ``gemini-flash`` earlier; this guards both.
    """
    from common.llm_provider import MODEL_REGISTRY

    for key in AI_MODEL_ITEMS:
        option = MODEL_REGISTRY.get(key)
        if option is None:
            continue
        assert "latest" not in option.model, (
            f"AI_MODEL_ITEMS[{key!r}] annotates as "
            f"{AI_MODEL_ITEMS[key]['display_title']!r} but its registry entry "
            f"resolves to the rolling alias {option.model!r} — the stored "
            f"provenance would assert a version the run cannot confirm"
        )


def test_no_registry_entry_points_at_a_retired_item():
    retired = set(RETIRED_AI_MODEL_ITEM_IDS)
    live = {key: model["item_id"] for key, model in AI_MODEL_ITEMS.items()}
    assert not (set(live.values()) & retired), (
        f"registry still points at retired item(s): "
        f"{ {k: v for k, v in live.items() if v in retired} }"
    )
