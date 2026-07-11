"""
IWAC-instance configuration shared across pipelines.

Everything in this module is specific to the IWAC Omeka S instance
(https://islam.zmo.de). It exists so that instance-coupled constants live in
ONE place: several of them used to be copy-pasted across three or more
scripts, where they could silently drift apart.

Property IDs are stable for a given installation but differ between
installations — resolve unknown terms at runtime with
``OmekaClient.get_property_id()`` instead of guessing.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Authority item sets (used by NER reconciliation and reference indexing)
# ---------------------------------------------------------------------------

SPATIAL_AUTHORITY_ITEM_SETS: List[str] = ["268"]
SUBJECT_AUTHORITY_ITEM_SETS: List[str] = ["854", "2", "266"]
TOPIC_AUTHORITY_ITEM_SETS: List[str] = ["1"]

# ---------------------------------------------------------------------------
# Property IDs on the IWAC instance (verified live against islam.zmo.de)
# ---------------------------------------------------------------------------

DCTERMS_TITLE_PROPERTY_ID = 1
DCTERMS_SUBJECT_PROPERTY_ID = 3
DCTERMS_TYPE_PROPERTY_ID = 8
DCTERMS_IDENTIFIER_PROPERTY_ID = 10
DCTERMS_TABLE_OF_CONTENTS_PROPERTY_ID = 18
DCTERMS_SPATIAL_PROPERTY_ID = 40
BIBO_CONTENT_PROPERTY_ID = 91
IWAC_OCR_MODEL_PROPERTY_ID = 312      # iwac:ocrModel ("AI Model - OCR")
IWAC_SUMMARY_MODEL_PROPERTY_ID = 313  # iwac:summaryModel ("AI Model - Summary")

# ---------------------------------------------------------------------------
# Authority items
# ---------------------------------------------------------------------------

# "Notice d'autorité" authority-record type item (linked via customvocab:6)
AUTHORITY_RECORD_TYPE_ITEM_ID = 67568

# AI model annotation items (class 244, "Notice d'autorité").
# display_title mirrors the actual Omeka item title.
AI_MODEL_ITEMS: Dict[str, Dict] = {
    "claude-opus": {"item_id": 78528, "display_title": "Claude Opus 4.6"},
    "gemini-pro": {"item_id": 78536, "display_title": "Gemini 3.1 pro"},
    "gemini-flash": {"item_id": 78630, "display_title": "Gemini 3.5 flash"},
    "gemini-flash-lite": {"item_id": 78631, "display_title": "Gemini 3.1 flash lite"},
}


def item_api_url(base_url: str, item_id: int) -> str:
    """Build an item's API ``@id`` from the client's base URL.

    Use this instead of hardcoding absolute ``https://islam.zmo.de/...``
    URLs so that a changed ``OMEKA_BASE_URL`` keeps working.
    """
    return f"{base_url.rstrip('/')}/items/{item_id}"


def model_annotation_value(
    base_url: str,
    model_key: str,
    property_id: int,
    property_label: str,
) -> Dict:
    """Build the ``resource:item`` value object annotating which AI model
    produced a piece of content (OCR, summary, ...)."""
    model = AI_MODEL_ITEMS[model_key]
    return {
        "type": "resource:item",
        "property_id": property_id,
        "property_label": property_label,
        "is_public": True,
        "@id": item_api_url(base_url, model["item_id"]),
        "value_resource_id": model["item_id"],
        "value_resource_name": "items",
        "display_title": model["display_title"],
    }
