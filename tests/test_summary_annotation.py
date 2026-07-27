"""Tests for the iwac:summaryModel annotation written by AI_summary/03.

These guard a real regression. OmekaClient.upsert_property_value() has two paths:
it mutates an existing literal in place (which keeps a sibling ``@annotation``), or
it APPENDS a freshly built value object made only of
type/property_id/property_label/is_public/@value — with no ``@annotation``.

The append path is the dangerous one: writing a summary to a property that has no
literal yet produces an un-annotated value, silently losing the record of which AI
model produced it. That is how six articles lost their iwac:summaryModel provenance
when their summaries were moved from dcterms:abstract to bibo:shortDescription.
"""

import importlib.util
from pathlib import Path

import pytest

from common.iwac_config import (
    AI_MODEL_ITEMS,
    IWAC_SUMMARY_MODEL_PROPERTY_ID,
    model_annotation_value,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PID = 116
BASE_URL = "https://example.org/api"


def _load_updater():
    """Import the numbered script by path (module names can't start with a digit)."""
    path = REPO_ROOT / "AI_summary" / "03_omeka_update_summaries.py"
    spec = importlib.util.spec_from_file_location("omeka_update_summaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def updater():
    return _load_updater()


@pytest.fixture
def model_value():
    return model_annotation_value(
        BASE_URL, "gpt-5.6-luna", IWAC_SUMMARY_MODEL_PROPERTY_ID, "AI Model - Summary"
    )


def _annotation(item_data):
    return item_data["bibo:shortDescription"][0].get("@annotation")


def test_writes_summary_with_model_annotation(updater, model_value):
    """The core guard: the APPEND path must still produce an annotated value."""
    item = {"o:id": 1}
    assert updater.apply_summary(item, "Résumé.", SUMMARY_PID, model_value) is True

    value = item["bibo:shortDescription"][0]
    assert value["@value"] == "Résumé."
    assert value["property_id"] == SUMMARY_PID
    linked = _annotation(item)["iwac:summaryModel"][0]
    assert linked["value_resource_id"] == AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"]
    assert linked["property_id"] == IWAC_SUMMARY_MODEL_PROPERTY_ID


def test_preserves_other_properties(updater, model_value):
    # bibo:content is 64KB of OCR on real items — it must survive untouched.
    item = {"o:id": 1, "bibo:content": [{"@value": "ocr", "property_id": 91, "type": "literal"}]}
    updater.apply_summary(item, "Résumé.", SUMMARY_PID, model_value)
    assert item["bibo:content"] == [{"@value": "ocr", "property_id": 91, "type": "literal"}]


def test_idempotent_when_text_and_model_unchanged(updater, model_value):
    item = {"o:id": 1}
    assert updater.apply_summary(item, "Résumé.", SUMMARY_PID, model_value) is True
    # Second pass must report no change so the caller skips a needless PATCH.
    assert updater.apply_summary(item, "Résumé.", SUMMARY_PID, model_value) is False


def test_refreshes_a_stale_model_annotation(updater, model_value):
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
    assert updater.apply_summary(item, "Résumé.", SUMMARY_PID, model_value) is True
    assert _annotation(item)["iwac:summaryModel"][0]["value_resource_id"] == 79609


def test_annotation_survives_a_text_rewrite(updater, model_value):
    """The in-place path: replacing the text must keep exactly one annotated value."""
    item = {"o:id": 1}
    updater.apply_summary(item, "Ancien résumé.", SUMMARY_PID, model_value)
    assert updater.apply_summary(item, "Nouveau résumé.", SUMMARY_PID, model_value) is True

    values = item["bibo:shortDescription"]
    assert len(values) == 1, "rewrite should replace the literal, not append a second one"
    assert values[0]["@value"] == "Nouveau résumé."
    assert _annotation(item) is not None, "annotation was dropped by the rewrite"


def test_registry_points_at_the_current_authority_items():
    assert AI_MODEL_ITEMS["gpt-5.6-luna"]["item_id"] == 79609
    assert AI_MODEL_ITEMS["gemini-flash"]["item_id"] == 79608
