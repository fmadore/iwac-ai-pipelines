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
    model_annotation_value,
)
from common.omeka_text_updater import PropertyTarget, apply_text_value

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
    assert _annotation(item)["iwac:summaryModel"][0]["value_resource_id"] == 79609


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
# Targets without provenance (OCR correction rewrites text another model OCR'd)
# ---------------------------------------------------------------------------

def test_unannotated_target_preserves_an_existing_annotation():
    """Correcting OCR text must not erase who did the original OCR.

    upsert_property_value mutates the existing literal in place, so a sibling
    ``@annotation`` survives — this pins that behaviour for the correction
    pipeline, which deliberately writes no annotation of its own.
    """
    ocr_provenance = {"iwac:ocrModel": [{"value_resource_id": 79608}]}
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
    assert AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"] == 79609
    assert AI_MODEL_ITEMS["gemini-flash"]["item_id"] == 79608
