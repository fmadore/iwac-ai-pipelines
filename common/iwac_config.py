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

from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Authority item sets (used by NER reconciliation and reference indexing)
# ---------------------------------------------------------------------------

SPATIAL_AUTHORITY_ITEM_SETS: List[str] = ["268"]
SUBJECT_AUTHORITY_ITEM_SETS: List[str] = ["854", "2", "266"]
TOPIC_AUTHORITY_ITEM_SETS: List[str] = ["1"]

# ---------------------------------------------------------------------------
# Scholarly references
# ---------------------------------------------------------------------------

#: The nine resource classes that make up the "references" population — the
#: scholarly literature *about* the collection's subject, as opposed to the
#: archival material itself. Counts verified live on 2026-08-18 (867 items).
#:
#: Class, not template, is the key. These nine classes share templates in both
#: directions — template 10 carries both ``Book`` (40) and ``EditedBook`` (52) —
#: so a template filter would both over- and under-select. The same rule governs
#: the ``references`` subset of the Hugging Face export.
REFERENCE_RESOURCE_CLASSES: Dict[int, str] = {
    35: "Academic Article",
    43: "Chapter",
    88: "Thesis",
    40: "Book",
    82: "Report",
    178: "Book review",
    52: "Edited Book",
    77: "Personal Communication",
    305: "Blog post",
}

# ---------------------------------------------------------------------------
# YouTube-hosted audiovisual items
# ---------------------------------------------------------------------------

#: Resource template of the embedded-YouTube items ingested from public channels
#: since 2026-08-12. They share resource class 38 (``bibo:AudioVisualDocument``)
#: with the deposited recordings on template 19, so the class alone cannot tell
#: the two apart — and the difference matters: a YouTube media carries no file
#: (null ``o:original_url`` / ``o:media_type`` / ``o:size``), so anything that
#: assumes "primary media ⇒ bytes to download" is wrong for these.
YOUTUBE_VIDEO_TEMPLATE_ID = 23

#: Item sets holding YouTube videos, one per country. Used as the default scope
#: of ``AI_youtube_transcription/01``; new channels add new sets here.
YOUTUBE_VIDEO_ITEM_SETS: List[str] = ["108260"]  # YouTube videos Burkina Faso

# ---------------------------------------------------------------------------
# Property IDs on the IWAC instance (verified live against islam.zmo.de)
# ---------------------------------------------------------------------------

DCTERMS_TITLE_PROPERTY_ID = 1
DCTERMS_SUBJECT_PROPERTY_ID = 3
DCTERMS_TYPE_PROPERTY_ID = 8
DCTERMS_IDENTIFIER_PROPERTY_ID = 10
DCTERMS_LANGUAGE_PROPERTY_ID = 12
DCTERMS_TABLE_OF_CONTENTS_PROPERTY_ID = 18
DCTERMS_SPATIAL_PROPERTY_ID = 40
BIBO_CONTENT_PROPERTY_ID = 91
#: bibo:cites ("cites"). Declared on all four reference templates — Thesis (14),
#: Journal article (18), Book chapter (11) and Book (10) — and populated on zero
#: items until ``AI_publication_extraction/04`` started writing the works a
#: publication cites. Verified live 2026-08-18.
BIBO_CITES_PROPERTY_ID = 60
FABIO_HAS_URL_PROPERTY_ID = 278       # fabio:hasURL — a ``uri`` value, read from ``@id``
IWAC_OCR_MODEL_PROPERTY_ID = 312      # iwac:ocrModel ("AI Model - OCR")
IWAC_SUMMARY_MODEL_PROPERTY_ID = 313  # iwac:summaryModel ("AI Model - Summary")
#: iwac:transcriptionModel ("AI Model - Transcription"). Declared in the IWAC
#: vocabulary since it was first uploaded but unused until 2026-08-12, when
#: ``AI_youtube_transcription/03`` became the first step to write it; before
#: that a transcription's model was recorded nowhere but the file header on
#: disk. ``AI_audio_summary/03`` writes it too since 2026-08-27 — but only for
#: the models that have an authority item, which is neither Voxtral nor either
#: of the rolling Gemini aliases that step 02 offers.
IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID = 315
#: iwac:nerModel ("AI Model - NER"). Declared with the vocabulary but written by
#: nothing until 2026-09-02, when ``AI_NER/03`` and ``AI_reference_indexing/05``
#: started annotating every ``dcterms:subject`` / ``dcterms:spatial`` link they
#: append. Links added before then carry no provenance and cannot be told apart
#: from hand-catalogued ones. Verified live 2026-09-02.
IWAC_NER_MODEL_PROPERTY_ID = 314

#: The IWAC ontology (``iwac:``) in this instance. Its property IDs are
#: contiguous but not guaranteed to be — resolve terms with
#: :func:`resolve_property_ids` rather than assuming a range.
IWAC_VOCABULARY_ID = 10

#: Slug of the public site an item page is served from. The same items are
#: published under ``westafrica`` for the English UI; the French site is the
#: one cited, so it is the default.
IWAC_SITE_SLUG = "afrique_ouest"

# ---------------------------------------------------------------------------
# Authority items
# ---------------------------------------------------------------------------

# "Notice d'autorité" authority-record type item (linked via customvocab:6)
AUTHORITY_RECORD_TYPE_ITEM_ID = 67568

#: Item set behind custom vocab 6, where the ``dcterms:language`` authority
#: records live alongside the other "Notice d'autorité" items.
AUTHORITY_ITEM_SET_ID = 267

#: ``dcterms:language`` links to an authority item whose title is a FRENCH
#: language name — there is no ISO code anywhere in the record. This maps the
#: codes an AI language-detection pass returns to the label to look up.
#:
#: Deliberately labels, not item IDs: the IDs are assigned per installation and
#: the ones in use here are not contiguous (Français 8355, Ewé 66720, Kabyè
#: 79081, Espagnol 26353 — added years apart), so they are resolved by title at
#: runtime instead.
#:
#: This is exactly the languages the instance holds an authority record for,
#: verified live — no more. Peul, Bambara, Zarma, Yoruba and Wolof are spoken in
#: this material and deliberately absent: adding a code here without its Omeka
#: record only moves the failure later, and creating the record is a curatorial
#: act, not something a pipeline should infer. A detected language outside this
#: map is reported under its own name so an operator can decide, and never linked.
#:
#: ``dyu`` shows how the two halves are meant to move together: transcribing the
#: 46 YouTube videos on 2026-08-13 turned up Dioula on one item, the pipeline
#: reported it as unlinkable, the authority record was created (item 108359), and
#: the entry was added here afterwards. Record first, then code.
LANGUAGE_LABELS_BY_CODE: Dict[str, str] = {
    "fr": "Français", "en": "Anglais", "ar": "Arabe", "ha": "Haoussa",
    "mos": "Mooré", "ee": "Ewé", "kbp": "Kabyè", "ddn": "Dendi",
    "de": "Allemand", "it": "Italien", "es": "Espagnol", "sl": "Slovène",
    "dyu": "Dioula",
}

# AI model annotation items (class 244, "Notice d'autorité", item set 267).
# display_title mirrors the actual Omeka item title — it is what the pre-write
# dump and the confirmation panel show.
#
# Keys match the ``common.llm_provider`` registry keys where one exists, so a
# pipeline can map the model it just ran straight to its annotation item.
# Superseded models keep their items while an annotation still needs them
# (``gemini-3.6-flash`` finishes runs that started on it) and leave this dict
# once nothing does. When each entry was created and why: CHANGELOG.md.
AI_MODEL_ITEMS: Dict[str, Dict] = {
    "claude-opus-5": {"item_id": 79615, "display_title": "Claude Opus 5.0"},
    "claude-opus-4.6": {"item_id": 78528, "display_title": "Claude Opus 4.6"},
    "gemini-3.1-pro": {"item_id": 78536, "display_title": "Gemini 3.1 Pro"},
    "gemini-3.7-flash": {"item_id": 111774, "display_title": "Gemini 3.7 Flash"},
    "gemini-3.6-flash": {"item_id": 79611, "display_title": "Gemini 3.6 Flash"},
    "gemini-3.5-flash-lite": {"item_id": 79617, "display_title": "Gemini 3.5 Flash-Lite"},
    "gemini-3.1-flash-lite": {"item_id": 78631, "display_title": "Gemini 3.1 Flash Lite"},
    # Keyed on the OpenRouter route, which is the only one that may see archive
    # text: Gemma is free-of-charge on the Gemini API and its pricing page states
    # that free-tier content is used to improve Google's products. The registry's
    # ``gemma-4`` (Gemini-routed) key is deliberately absent here.
    "gemma-4-openrouter": {"item_id": 111663, "display_title": "Gemma 4 31B"},
    "gpt-5.6-luna": {"item_id": 79610, "display_title": "GPT-5.6 Luna"},
    "mistral-small": {"item_id": 79614, "display_title": "Mistral Small 4"},
    "qwen3.5-moe": {"item_id": 79616, "display_title": "Qwen3.5 122B-A10B"},
    "qwen3.5-moe-small": {"item_id": 79612, "display_title": "Qwen3.5 35B-A3B"},
    # Keyed on the self-hosted route, which is the one that produced the
    # annotations: they were generated on university hardware, and the OpenRouter
    # twin (``qwen3.8-27b-openrouter``) is deliberately absent because it has
    # written nothing. The route is half of what the provenance record claims,
    # so the day the twin annotates anything it needs its own authority item
    # rather than a share of this one. Created 2026-08-25.
    "qwen3.8-27b-selfhosted": {"item_id": 111933, "display_title": "Qwen3.8 27B"},
    "deepseek-v4-flash-0731": {
        "item_id": 83261,
        "display_title": "DeepSeek V4 Flash 0731",
    },
    # Not an LLM registry key: Mistral's OCR endpoint is a dedicated model with
    # no ``MODEL_REGISTRY`` entry, and the key is the pinned API id rather than
    # ``mistral-ocr-latest``, which is a rolling alias — see
    # ``common/mistral_ocr.py``. Created 2026-08-18 for
    # ``AI_publication_extraction``; the 425 ``bibo:content`` values already on
    # the reference corpus carry no provenance annotation at all.
    "mistral-ocr-4-1": {"item_id": 111889, "display_title": "Mistral OCR 4.1"},
    # Also not an LLM registry key, and for the same reason as Mistral's OCR
    # endpoint: ``gemini-3.5-transcribe`` is a dedicated speech-to-text model
    # reached through the Interactions API, not a chat model — it rejects a
    # system instruction outright ("Developer instruction is not enabled for
    # this model"), so there is nothing for ``llm_provider`` to route. The key
    # mirrors the item's ``dcterms:alternative``. Created 2026-08-27 for
    # ``AI_audio_summary/02c``; the audio pipeline stamped no transcription
    # provenance at all before it.
    "gemini-3.5-transcribe": {"item_id": 113077, "display_title": "Gemini 3.5 Transcribe"},
}

#: Superseded authority items — never annotate with these. Why each was
#: retired, and the three rolling-alias keys that were re-pinned before
#: them, is in CHANGELOG.md under "AI model authority items".
#:
#:   79608  "Gemini 3.6 flash"  duplicate of 79611
#:   79609  "GPT-5.6 Luna"      deleted upstream; replaced by 79610
#:
#: The rule those incidents left behind: a key here must name a pinned
#: release. A rolling alias (``gemini-flash-latest``, ``gemini-pro-latest``)
#: reports its own version as "… Latest", so a run through it cannot confirm
#: the model the annotation would claim — and a model whose version cannot
#: be stated is a model that cannot be cited.
RETIRED_AI_MODEL_ITEM_IDS = (79608, 79609)


def resolve_property_ids(client, terms: Iterable[str]) -> Dict[str, int]:
    """Resolve ``iwac:`` terms to property IDs in one request.

    The whole IWAC vocabulary fits inside a single page, so a run that needs
    thirty-odd property IDs costs one GET rather than thirty. Resolving instead
    of hardcoding matters here because these IDs are assigned by Omeka when the
    vocabulary is updated, and differ between installations.

    Raises:
        KeyError: if any term is missing, naming all of them. A pipeline that
            silently skipped an unresolved property would write a partial
            annotation set and look successful.
    """
    url = f"{client.base_url}/properties?vocabulary_id={IWAC_VOCABULARY_ID}&per_page=100"
    result = client.get_resource(url)
    if not isinstance(result, list):
        raise RuntimeError(f"Could not list vocabulary {IWAC_VOCABULARY_ID}")

    available = {p["o:term"]: p["o:id"] for p in result}
    wanted = list(terms)
    missing = [term for term in wanted if term not in available]
    if missing:
        raise KeyError(
            f"{len(missing)} property term(s) not in the IWAC vocabulary: "
            f"{', '.join(missing)}. Update the vocabulary first — see "
            f"AI_sentiment_analysis/00_setup_properties.py."
        )
    return {term: available[term] for term in wanted}


def item_api_url(base_url: str, item_id: int) -> str:
    """Build an item's API ``@id`` from the client's base URL.

    Use this instead of hardcoding absolute ``https://islam.zmo.de/...``
    URLs so that a changed ``OMEKA_BASE_URL`` keeps working.
    """
    return f"{base_url.rstrip('/')}/items/{item_id}"


def item_page_url(base_url: str, item_id: int, site_slug: str = IWAC_SITE_SLUG) -> str:
    """Build an item's *public page* URL from the client's base URL.

    ``OmekaClient.base_url`` always ends in ``/api``, which serves JSON; the
    human-readable record a citation should point at lives on the site path
    instead. Pass ``site_slug="westafrica"`` for the English UI.
    """
    root = base_url.rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")]
    return f"{root}/s/{site_slug}/item/{item_id}"


def select_model_key(default: Optional[str] = None) -> Optional[str]:
    """Prompt for the AI model that produced the content being written.

    Returns the chosen ``AI_MODEL_ITEMS`` key, or ``None`` if the input was
    invalid. Plain print/input keeps this module dependency-free, matching
    ``llm_provider.prompt_for_model_choice``.
    """
    keys = list(AI_MODEL_ITEMS)
    default_index = keys.index(default) + 1 if default in AI_MODEL_ITEMS else 1

    print("Select the AI model that produced this content:")
    for number, key in enumerate(keys, start=1):
        model = AI_MODEL_ITEMS[key]
        print(f"  {number}) {model['display_title']} (item {model['item_id']})")

    choice = input(f"Enter choice [{default_index}]: ").strip() or str(default_index)
    if choice.isdigit() and 1 <= int(choice) <= len(keys):
        return keys[int(choice) - 1]
    print("Invalid choice.")
    return None


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
