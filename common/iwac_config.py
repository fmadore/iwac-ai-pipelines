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
# display_title mirrors the actual Omeka item title.
#
# Keys match the ``common.llm_provider`` registry keys where one exists, so a
# pipeline can map the model it just ran straight to its annotation item.
#
# Superseded models keep their Omeka items so historical annotations still
# resolve (e.g. 78053 "Gemini 3.0 flash", 78630 "Gemini 3.5 flash"); they are
# dropped from this dict once no longer offered for new writes.
#
# ``gemini-3.6-flash`` is superseded by 3.7 everywhere a run may be *started*,
# but stays here because it is still needed to *finish* one: 46 YouTube
# transcripts already on the archive carry an ``iwac:transcriptionModel``
# annotation naming item 79611, and OCR text produced by 3.6 that has not
# reached step 03 yet must be stamped for the model that actually read it, not
# for whatever is current on upload day.
#
# Three ``display_title`` values were corrected on 2026-08-14 to match the live
# item titles they claim to mirror: "Gemini 3.1 pro" -> "Gemini 3.1 Pro",
# "Gemini 3.6 flash" -> "Gemini 3.6 Flash", "Gemini 3.1 flash lite" -> "Gemini
# 3.1 Flash Lite". Omeka regenerates this key from the linked item on read, so
# nothing stored was ever wrong — but it is what the pre-write payload dump and
# the confirmation panel show, which is where an operator checks that the run is
# about to stamp the model they think it is.
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
}

#: Superseded authority items, kept only so this file records why an id that
#: appears in git history no longer resolves. Do not annotate with these.
#:
#:   79608  "Gemini 3.6 flash"  duplicate of 79611, created 2026-07-27
#:   79609  "GPT-5.6 Luna"      deleted upstream; replaced by 79610
#:
#: The key for Gemini was ``gemini-flash`` until 2026-07-31. That is the
#: registry key for ``gemini-flash-latest``, a rolling alias which reports its
#: version as literally "Gemini Flash Latest" — so annotating such a run as
#: "Gemini 3.6 flash" asserted a model version the run could not confirm. The
#: key is now the pinned ``gemini-3.6-flash``.
#:
#: ``gemini-flash-lite`` was the same bug and outlived the fix: it is the
#: registry key for the *rolling* ``gemini-flash-lite-latest`` while claiming
#: item 78631, "Gemini 3.1 flash lite". That alias resolved to 3.1 when the
#: entry was written and resolves to 3.5 now, so every annotation written
#: through it since Gemini 3.5 Flash-Lite shipped names the wrong model. Both
#: Flash-Lite generations now have pinned registry entries and their own items,
#: and the rolling key is deliberately absent: a model whose version cannot be
#: stated is a model that cannot be cited.
#:
#: ``gemini-pro`` was the third case, found the same day by the guard added for
#: the second (``test_no_registry_key_is_a_rolling_alias``). It resolved to
#: ``gemini-pro-latest`` while claiming item 78536, "Gemini 3.1 pro"; the alias
#: reports its own version as the string "Gemini Pro Latest", so a run through
#: it could not have confirmed that claim even if asked. Re-keyed to the pinned
#: ``gemini-3.1-pro``. The rolling ``gemini-pro`` option stays in
#: ``MODEL_REGISTRY`` — it is the right choice for a pipeline that wants
#: whatever Pro is current and does not stamp provenance — it simply cannot be
#: an annotation key.
#:
#: ``claude-opus`` was the same ambiguity in a different shape and was split on
#: 2026-08-07 into ``claude-opus-4.6`` and ``claude-opus-5``. There is no
#: registry entry for either — the Claude keys exist only so ``AI_summary_issue``
#: can stamp which model read the PDFs when the index came from the
#: ``issue-indexing`` agent rather than from a provider API. (The other
#: agent-driven pipeline, ``AI_reference_indexing``, writes resource links and
#: stamps nothing.) That model is whatever Claude Code is running, which no code
#: here observes, so the operator asserts it — and an unversioned key invited
#: them to assert a name that had quietly come to mean the wrong release. 78528
#: keeps the existing Opus 4.6 annotations resolving; new runs take 79615.
#:
#: ``deepseek-v4-flash`` (item 79613, "DeepSeek V4 Flash") left this dict on
#: 2026-08-07: the April preview is superseded everywhere by the dated 0731
#: release. Its sentiment values were deleted from Omeka the same day, so
#: nothing links to 79613 any more. The item itself is still live and is left
#: out of RETIRED_AI_MODEL_ITEM_IDS, which means "this id no longer resolves".
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
