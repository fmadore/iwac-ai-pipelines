"""
sentiment_core.py
=================

Shared sentiment-analysis primitives: the output schema, the prompt loader,
the Omeka content extractor and the per-model / all-model analysis calls.

Both the production run (``01_sentiment_analysis.py``) and the pilot
(``02_pilot_new_panel.py``) import from here. That matters more than the usual
DRY argument: a pilot whose schema or prompt has drifted from production is not
comparable to it, and the whole point of the pilot is comparability.

Nothing in this module writes to Omeka.
"""
import time
import hashlib
import logging
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel, Field, model_validator

from common.iwac_config import AI_MODEL_ITEMS
from common.llm_provider import BaseLLMClient
from common.rate_limiter import QuotaExhaustedError, is_quota_exhausted

# The production runner owns this retry budget. Provider SDK retries are
# disabled there so the item-level timeout is not multiplied invisibly by a
# second retry loop inside the SDK.
MODEL_MAX_ATTEMPTS = 3
MODEL_RETRY_DELAYS = (1, 2)

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

CENTRALITE_LABELS = ("Très central", "Central", "Secondaire", "Marginal", "Non abordé")
POLARITE_LABELS = (
    "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable",
)

#: Subjectivité, weakest→strongest. Generation 1 asked the models for the
#: integer 1-5 and mapped it to these labels on the way into Omeka; generation 2
#: asks for the label directly.
#:
#: The reason is measured, not aesthetic. Subjectivité was the one dimension
#: requested as a number and is by a distance the least reliable: in the
#: 2026-07-29 pilot its pairwise kappa within the panel ran 0.093-0.470, against
#: 0.248-0.478 for polarité and up to 0.725 for centralité, and one model
#: reproduced its own answer just 47% of the time. Numeric scales are a known
#: cause of exactly this — "prompting for numerical scores instead of labels
#: reduces all LLMs' compliance and accuracy" (arXiv:2406.11980) — and the
#: storage was always a link to a labelled item anyway, so nothing downstream
#: needed the integer.
SUBJECTIVITE_LABELS = (
    "Très objectif", "Plutôt objectif", "Mixte", "Plutôt subjectif", "Très subjectif",
)

#: Ordinal rank of each label, weakest→strongest. Used by the pilot's agreement
#: maths; ``Non applicable`` has no place on the scale and is deliberately absent.
POLARITE_ORDER = {
    "Très négatif": 1, "Négatif": 2, "Neutre": 3, "Positif": 4, "Très positif": 5,
}
CENTRALITE_ORDER = {
    "Non abordé": 1, "Marginal": 2, "Secondaire": 3, "Central": 4, "Très central": 5,
}
#: Doubles as the generation-1 bridge: the integer a v1 model returned is this
#: label's rank, so both generations compare on one scale.
SUBJECTIVITE_ORDER = {label: rank for rank, label in enumerate(SUBJECTIVITE_LABELS, 1)}

# Omeka items backing the controlled vocabulary. Centralité, polarité and the
# subjectivité score are all stored as resource:item links, not literals.
# (Instance-specific, so arguably common/iwac_config.py — kept here because
# only the sentiment scripts use them.)
CENTRALITE_ITEM_IDS = {
    "Très central": 78048,
    "Central": 78049,
    "Secondaire": 78050,
    "Marginal": 78051,
    "Non abordé": 78052,
}

POLARITE_ITEM_IDS = {
    "Très positif": 78031,
    "Positif": 78038,
    "Neutre": 78039,
    "Négatif": 78040,
    "Très négatif": 78041,
    "Non applicable": 78042,
}

#: Keyed by label since 2026-07-31 (was 1-5). The Omeka items are unchanged —
#: only what the model is asked to return.
SUBJECTIVITE_ITEM_IDS = {
    "Très objectif": 78043,
    "Plutôt objectif": 78044,
    "Mixte": 78045,
    "Plutôt subjectif": 78046,
    "Très subjectif": 78047,
}

#: Reverse map for reading a stored subjectivité link back to its label. Used to
#: read generation-1 annotations, which is why it returns a label rather than
#: the integer those runs produced: both generations then land on one scale and
#: the pilot report can compare them by equality.
ITEM_ID_TO_SUBJECTIVITE = {v: k for k, v in SUBJECTIVITE_ITEM_IDS.items()}


class SentimentAnalysisOutput(BaseModel):
    """Schema for sentiment analysis output - used for structured outputs with AI APIs."""
    centralite_islam_musulmans: Literal[
        "Très central", "Central", "Secondaire", "Marginal", "Non abordé"
    ] = Field(description="Importance accordée aux thèmes liés à l'islam et aux musulmans dans l'article")
    centralite_justification: str = Field(description="Courte justification en 1 phrase sur la centralité de l'islam/des musulmans")
    # Named ``_score`` for continuity: it is the HF column name
    # (``{model}_subjectivite_score``) and the Omeka property suffix
    # (``SubjectiviteScore``), both already live. It holds a label, not a number.
    subjectivite_score: Optional[
        Literal["Très objectif", "Plutôt objectif", "Mixte",
                "Plutôt subjectif", "Très subjectif"]
    ] = Field(
        default=None,
        description="Niveau de subjectivité du traitement, ou null si le sujet n'est pas abordé",
    )
    subjectivite_justification: str = Field(description="Justification en 1-2 phrases pour le score de subjectivité")
    polarite: Literal[
        "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable"
    ] = Field(description="Sentiment général exprimé dans l'article envers l'islam et/ou les musulmans")
    polarite_justification: str = Field(description="Justification en 1-2 phrases pour la polarité")

    @model_validator(mode="after")
    def _subjectivite_required_unless_unaddressed(self) -> "SentimentAnalysisOutput":
        """Reject a null subjectivité on an article that does discuss Islam.

        ``subjectivite_score`` is nullable so that ``Non abordé`` articles have
        somewhere to go, and ``strict`` JSON-schema validation therefore accepts
        a null on *any* article — the schema cannot express "null only when
        centralité is Non abordé". A model may be perfectly schema-compliant and
        still answer nothing.

        DeepSeek V4 Flash 0731 did exactly that on its first full pass, 2026-08:
        1,937 of 12,305 articles came back with no subjectivité, and only 452 of
        those were ``Non abordé``. The other 1,485 were articles it had itself
        marked as discussing Islam — and the omission was not random, rising from
        11% on ``Très central`` to 22% on ``Marginal``. GPT-5.6 Luna returned a
        label on 100% of the same corpus, so this is a model behaviour, not a
        property of the articles.

        Nothing caught it: ``build_property_values`` drops a field whose value is
        missing, so the item was written with four properties instead of six and
        the run reported success. Raising here instead routes the answer into
        ``analyze_with_model``'s retry loop, and a still-null third attempt
        becomes a *recorded failure* the ordinary re-run path picks up. A silent
        gap becomes a loud one.

        The retry is only a resample — provider-side ``strict`` validation has
        already passed, so the model is never told why its answer was rejected.
        That is enough in practice because these models run at their vendor
        temperature (1.0 for DeepSeek V4), so attempts differ.
        """
        if (
            self.subjectivite_score is None
            and self.centralite_islam_musulmans != "Non abordé"
        ):
            raise ValueError(
                "subjectivite_score is null but centralité is "
                f"{self.centralite_islam_musulmans!r}; a null subjectivité is "
                "only valid when centralité is 'Non abordé'"
            )
        return self


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

#: Schema field -> Omeka property-name suffix. The six properties of a model's
#: set are ``iwac:<prefix><suffix>``, so this map plus a prefix is the whole
#: property layout.
RESULT_FIELD_SUFFIXES: Dict[str, str] = {
    "centralite_islam_musulmans": "Centralite",
    "centralite_justification": "CentraliteJustification",
    "polarite": "Polarite",
    "polarite_justification": "PolariteJustification",
    "subjectivite_score": "SubjectiviteScore",
    "subjectivite_justification": "SubjectiviteJustification",
}

#: The three fields stored as ``resource:item`` links into the controlled
#: vocabulary rather than as literals. The other three are French literals.
RESOURCE_FIELDS = frozenset(
    {"centralite_islam_musulmans", "polarite", "subjectivite_score"}
)

#: Value-annotation property recording which model produced a sentiment value
#: (``iwac:sentimentModel``, the sibling of ``iwac:ocrModel`` /
#: ``iwac:summaryModel``). Generation 1 had no such annotation, which is why
#: reconstructing its models needed a dig through git history.
SENTIMENT_MODEL_ANNOTATION_TERM = "iwac:sentimentModel"


@dataclass(frozen=True)
class PanelMember:
    """One annotator model: where it comes from and where its answers go."""

    #: Hugging Face column prefix — the provider model id with ``-``/``.``/``/``
    #: folded to ``_``. Also the key used in the cache and the pilot output.
    key: str
    #: ``common.llm_provider`` registry key.
    registry_key: str
    #: Human label for progress output and Omeka property labels.
    label: str
    #: Omeka property-name stem: ``iwac:<prefix>Centralite`` and friends.
    property_prefix: str

    @property
    def model_item_id(self) -> int:
        """Authority item (class 244, item set 267) the annotation links to.

        Delegated rather than stored: ``AI_MODEL_ITEMS`` is the registry every
        pipeline annotates from, and a second copy here would be free to drift
        from it silently.
        """
        return AI_MODEL_ITEMS[self.registry_key]["item_id"]

    def term(self, result_field: str) -> str:
        """Omeka term for one schema field, e.g. ``iwac:gemini36FlashPolarite``."""
        return f"iwac:{self.property_prefix}{RESULT_FIELD_SUFFIXES[result_field]}"

    @property
    def terms(self) -> List[str]:
        return [self.term(field) for field in RESULT_FIELD_SUFFIXES]

    @property
    def probe_term(self) -> str:
        """The property whose presence means "this model has already run"."""
        return self.term("centralite_islam_musulmans")


#: Generation 2, live from 2026-07-31. Each member owns SIX properties named
#: for the model, never for its vendor: reusing a vendor slot is what made
#: generation 1 impossible to attribute without a git archaeology session.
#:
#: Three of the four are open-weights releases (Mistral Small 4 and Gemma 4
#: under Apache-2.0, DeepSeek V4 Flash under MIT), so those annotations can be
#: regenerated from weights that are archivable alongside them. Their active
#: parameter counts — 6.5B, 13B and 31B (Gemma is dense, so all of it) — sit
#: inside a factor of five, wider than the pair that preceded them.
#:
#: Every member is its vendor's high-volume tier. That is the property that
#: makes this a panel rather than a quality ladder, and it is why the Gemini
#: slot moved off ``gemini-3.6-flash`` on 2026-07-31: at $1.50/$7.50 per 1M it
#: cost five to seventeen times the others, so any disagreement it had with
#: them could be read as "the expensive model knows better" rather than as two
#: readings of the construct. Nothing was ever written to ``iwac:gemini36Flash*``.
#:
#: **The Google slot became Gemma 4 31B on 2026-08-14, replacing
#: ``gemini-3.5-flash-lite``.** Flash-Lite held the slot from 2026-07-31 but
#: never annotated anything — ``iwac:gemini35FlashLiteCentralite`` was verified
#: at 0 items on the live archive the day of the swap — so this fills an empty
#: slot rather than mixing two models into one column, and generation 2 is
#: unchanged by it. Flash-Lite was still the panel's cost outlier at $0.30/$2.50
#: per 1M, against $0.09–0.20 in and $0.18–1.20 out for the rest, and the only
#: member whose weights could not be archived beside its annotations.
#:
#: Gemma replaces the Google slot rather than joining as a fifth voice. It comes
#: out of the same lab and pretraining-pipeline family as Gemini, so running both
#: would buy correlated annotator error — the preference-leakage effect in the
#: LLM-as-judge literature — which inflates agreement for reasons that have
#: nothing to do with the construct, while making the panel 2/5 Google.
#:
#: **It is routed through OpenRouter, not ``GEMINI_API_KEY``, and that is not an
#: implementation detail.** Gemma is free-of-charge on the Gemini API with no
#: paid tier, and Google's pricing page states that free-tier content is used to
#: improve its products; this pipeline ships whole archival articles, which is
#: exactly what ``OPENROUTER_PROVIDER_PREFS``' ``data_collection: "deny"`` exists
#: to prevent. The Gemini route is also capped at 16,000 input tokens per minute
#: for this model (measured 2026-08-14, quota
#: ``GenerateContentInputTokensPerModelPerMinute``), i.e. ~4 articles a minute
#: and ~51 h for the corpus — no faster than OpenRouter, for the privacy cost.
#:
#: **Qwen3.5 122B-A10B was dropped on 2026-08-05, before it annotated
#: anything.** It was never a model-quality decision — it was a serving one.
#: OpenRouter lists five endpoints for it against DeepSeek's 22, and one of
#: those does not support structured outputs, so ``require_parameters`` leaves
#: four. That queueing put its median call at 104 s where the rest of the panel
#: sat at 4–6 s, i.e. a corpus pass measured in days rather than hours, and the
#: bottleneck was never the prompt or the reasoning level (it is marginally
#: *faster* at ``medium`` than at ``low``). ``iwac:qwen35A10b*`` holds zero
#: values, so it follows ``qwen35A3b`` and ``gemini36Flash`` out of the emitted
#: ontology, which ``00_setup_properties.py --verify`` confirms is empty before
#: any upload.
#:
#: Gemma was checked against that precedent before it was added, because it is the
#: same shape of risk: OpenRouter lists 19 endpoints for ``google/gemma-4-31b-it``
#: and 16 of them support structured outputs, so ``require_parameters`` leaves a
#: deep pool rather than Qwen's four (verified 2026-08-14). Its ~72 s median is
#: still the slowest in the panel — see the throughput table in the README — but
#: it is queueing behind reasoning, not behind a starved endpoint list.
PANEL: Dict[str, PanelMember] = {
    m.key: m
    for m in (
        PanelMember("gemma_4_31b_it", "gemma-4-openrouter",
                    "Gemma 4 31B", "gemma431bIt"),
        PanelMember("gpt_5_6_luna", "gpt-5.6-luna", "GPT-5.6 Luna",
                    "gpt56Luna"),
        PanelMember("mistral_small_2603", "mistral-small", "Mistral Small 4",
                    "mistralSmall2603"),
        PanelMember("deepseek_v4_flash_0731", "deepseek-v4-flash-0731",
                    "DeepSeek V4 Flash 0731", "deepseekV4Flash0731"),
    )
}

#: Models under evaluation, which ``02_pilot_new_panel.py`` runs and nothing
#: else does. Membership of :data:`PANEL` is what makes a model *writable* —
#: ``01_sentiment_analysis.py`` iterates that dict alone — so a candidate parked
#: here is one that cannot reach Omeka however it is invoked. That separation is
#: the point: issue #12 defers the add-or-replace decision until the pilot
#: report exists, and a candidate sitting in ``PANEL`` in the meantime would be a
#: single ``--models`` flag away from writing six properties across 12,300 items.
#:
#: Everything else about a candidate is production: same ``PanelMember`` type,
#: same prompt, same schema, same :func:`panel_reasoning`, same call path
#: through ``analyze_with_all_models``. Only the membership is staged, so the
#: pilot measures the model rather than a lookalike of the pipeline.
#:
#: :attr:`PanelMember.model_item_id` raises ``KeyError`` for a candidate until
#: its Omeka authority item exists, and that is the correct order — the record
#: comes first, then the code that cites it (see ``AI_MODEL_ITEMS``). Nothing in
#: the pilot path asks for it; only the write path does.
#:
#: **Qwen3.8 27B appears twice, once per route.** The property prefix names the
#: model and the registry key carries the route, exactly as ``gemma_4_31b_it``
#: does for ``gemma-4-openrouter``. Running both on one sample is what turns
#: "self-hosting is cheaper" into a measurement: same weights, same articles,
#: with latency, structured-output reliability and reasoning depth read off each
#: route rather than assumed to match. Cost is not comparable in one unit —
#: OpenRouter bills tokens, a cluster bills GPU-hours and queue time — so record
#: both rather than converting one into the other.
#:
#: Neither needs a :data:`PANEL_REASONING_OVERRIDES` entry. Qwen3.8's ladder is
#: low/medium/xhigh, so the panel's requested ``medium`` is a rung the model
#: actually has: it would be the first member since GPT-5.6 Luna to sit at the
#: requested depth instead of being rounded up to it. Whether that survives the
#: OpenRouter route is precisely what the twin is there to find out — Gemma's
#: graduated levels collapsed to on/off when fanned across third-party backends
#: (see :data:`PANEL_REASONING`), and a self-hosted server is the only route here
#: where the depth can be read off the server's own logs.
PILOT_CANDIDATES: Dict[str, PanelMember] = {
    m.key: m
    for m in (
        PanelMember("qwen3_8_27b", "qwen3.8-27b-selfhosted",
                    "Qwen3.8 27B (self-hosted)", "qwen3827b"),
        PanelMember("qwen3_8_27b_openrouter", "qwen3.8-27b-openrouter",
                    "Qwen3.8 27B (OpenRouter)", "qwen3827bOr"),
    )
}

#: Reasoning depth requested of every panel member.
#:
#: The two knobs are sent together because the vendors split on naming: Gemini
#: takes ``thinking_level``, everyone else ``reasoning_effort``. Each client
#: reads only its own, so setting both is how one config reaches all four.
#:
#: Verified against the live APIs, 2026-07-29/31 and 2026-08-14 for Gemma:
#:   GPT-5.6 Luna           effort none/low/medium/high/xhigh/max  -> medium
#:   Gemma 4 31B            two levels only, MINIMAL|HIGH          -> high
#:   DeepSeek V4 Flash 0731 accepts only low/high/max              -> high
#:   Mistral Small 4        effort ONLY none|high — low/medium 400 -> high
#:
#: **Only GPT-5.6 Luna now sits at a genuine middle setting.** The other three
#: have no middle level, so they are rounded up to ``high`` rather than dropped
#: into a lighter/non-reasoning mode. That is a real limit on comparability and
#: belongs in any write-up of the panel results — and the swap of the Google
#: slot from Gemini 3.5 Flash-Lite (which did have MEDIUM) to Gemma made it
#: worse, taking the panel from an even 2/2 split to 3 of 4 rounded up. It was
#: accepted for the reasons in the ``PANEL`` comment above; it should not pass
#: unstated.
#:
#: Gemma's ``high`` is also the least legible of the three, because OpenRouter
#: fans the request across third-party backends serving the same weights and they
#: do not agree on what an effort means or on how to report it. Measured on one
#: article, 2026-08-14: with no effort sent the answer is ~200 output tokens and
#: no reasoning; at ``medium`` and at ``high`` it is ~1,000-1,200 output tokens
#: with 3.3-4.2k characters of reasoning — but the two levels are
#: indistinguishable from each other in both latency and reasoning length, so
#: Gemma's thinking is on/off through this route rather than graduated. One
#: backend reasoned at ``minimal`` too (897 tokens), and one reports
#: ``reasoning_tokens: 1`` while emitting 3.7k characters of it. Read the depth
#: as requested, never as measured.
PANEL_REASONING = {"reasoning_effort": "medium", "thinking_level": "MEDIUM"}

#: Per-member deviations from the shared middle setting. DeepSeek 0731 and
#: Gemma have no medium level; round up for the annotation panel (as Mistral
#: Small already does) so they stay in the reasoning regime. Stated explicitly
#: rather than left to each client's fallback, so the run manifest records a
#: decision instead of an accident. Other bulk text pipelines use the registry
#: defaults instead.
PANEL_REASONING_OVERRIDES: Dict[str, Dict[str, str]] = {
    "deepseek_v4_flash_0731": {"reasoning_effort": "high"},
    "gemma_4_31b_it": {"reasoning_effort": "high"},
}


def panel_reasoning(member_key: str) -> Dict[str, str]:
    """Requested reasoning config for one member, shared by pilot and production."""
    return {**PANEL_REASONING, **PANEL_REASONING_OVERRIDES.get(member_key, {})}

#: Effective (not merely requested) depth, for run manifests. See above.
_ROUNDED_UP_REASONING: Dict[str, str] = {
    "mistral_small_2603": "high (API accepts only none|high; medium rounded up)",
    "deepseek_v4_flash_0731": "high (API accepts only low|high|max; medium rounded up)",
    "gemma_4_31b_it": "high (model has only MINIMAL|HIGH; medium rounded up)",
}
#: Covers candidates too, so a pilot manifest records the same fact production
#: does. Qwen3.8 is absent from ``_ROUNDED_UP_REASONING`` on purpose: its ladder
#: has a real middle rung, which is most of why it is being piloted.
PANEL_REASONING_EFFECTIVE: Dict[str, str] = {
    key: _ROUNDED_UP_REASONING.get(key, "medium")
    for key in {**PANEL, **PILOT_CANDIDATES}
}

#: Sentiment on Omeka is now generation 2 alone. Deleted from the archive on
#: 2026-08-07, after the values were confirmed present on the Hugging Face full
#: mirror: the vendor-keyed generation-1 campaign (``iwac:gemini*``,
#: ``iwac:chatgpt*``, ``iwac:mistral*``, 12,286 items each) and the retired
#: April preview ``iwac:deepseekV4Flash*`` (11,482). The Hub keeps generation 1
#: as ``gemini_3_flash_preview_*`` / ``gpt_5_mini_*`` / ``ministral_14b_2512_*``,
#: frozen with ``omeka_prefix=None`` in the uploader's panel so ``hub_merge``
#: preserves what the uploader no longer emits — that freeze is the archive, and
#: removing it would drop the only remaining copy. The preview was not on the
#: Hub and was discarded deliberately.
#:
#: Nothing here reads those properties any more. Anything comparing generations
#: reads the Hub, not Omeka.


# ---------------------------------------------------------------------------
# Prompt + content
# ---------------------------------------------------------------------------

PROMPT_FILENAME = "sentiment_prompt.md"


def load_system_prompt() -> str:
    """Load the sentiment analysis prompt from the markdown file.

    Raises rather than returning "" — a silent empty prompt would produce
    plausible-looking but unanchored annotations.
    """
    prompt_path = Path(__file__).resolve().parent / PROMPT_FILENAME
    return prompt_path.read_text(encoding="utf-8")


def prompt_fingerprint(prompt: Optional[str] = None) -> str:
    """Short content hash of the prompt actually in use.

    Recorded in every cache record and pilot manifest. Prompt wording moves
    label distributions in ways that are not predictable from reading the diff
    (arXiv:2406.11980), so "which prompt produced this value" is as much a part
    of provenance as which model did — and unlike the model, nothing else in the
    stored record would capture it. A hash rather than a hand-maintained version
    string because the latter is exactly the kind of thing that gets forgotten
    in the edit that mattered.
    """
    text = load_system_prompt() if prompt is None else prompt
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def create_user_prompt(article_text: str) -> str:
    """Create the user prompt with the article text to analyze."""
    return f"Texte à analyser:\n---\n{article_text}\n---"


#: ``dcterms:language`` on IWAC items is a link to a language authority item,
#: not a literal or an ISO code, so the language is read off ``display_title``.
LANGUAGE_TERM = "dcterms:language"

#: Languages this panel may be run on.
#:
#: ``sentiment_prompt.md`` is written in French and requires French
#: justifications, and the scales were designed against francophone West
#: African press. The corpus also holds Ewé (32), Kabiyè (11) and Dendi (2)
#: articles: a French-prompted model will still emit confident-looking scores
#: for those, which is precisely the problem — the output is unusable but
#: indistinguishable from a real annotation once stored.
#:
#: This is the same reasoning that had the 2026-07 ``ocr_quality`` column
#: reverted: lexicon- and prompt-based measures mis-score the non-French
#: material rather than failing visibly on it.
ANALYSABLE_LANGUAGES = frozenset({"Français", "Anglais"})


def get_item_language(item: Dict[str, Any]) -> Optional[str]:
    """Label of an item's ``dcterms:language``, or ``None`` if untagged.

    ``None`` is a real answer, not an error: 6 of the 12,356 articles carry no
    language value at all, and guessing on their behalf is how the non-French
    material would get annotated anyway.
    """
    values = item.get(LANGUAGE_TERM) or []
    for value in values:
        if isinstance(value, dict):
            label = value.get("display_title") or value.get("@value")
            if label:
                return str(label).strip()
    return None


def get_item_content(item: Dict[str, Any]) -> str:
    """Extract bibo:content text from an Omeka item."""
    content_values = item.get('bibo:content', [])
    if not content_values:
        return ""

    # Prefer French language if available
    for val in content_values:
        if isinstance(val, dict) and val.get('@language') == 'fr':
            return val.get('@value', '')

    # Fall back to first value
    if content_values and isinstance(content_values[0], dict):
        return content_values[0].get('@value', '')

    return ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_with_model(
    llm_client: BaseLLMClient,
    text: str,
    system_prompt: str,
    model_label: str,
    logger: logging.Logger,
    max_retries: int = MODEL_MAX_ATTEMPTS
) -> Dict[str, Any]:
    """Analyze text using any LLM provider via the shared provider."""
    default_error = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": f"Erreur lors de l'analyse {model_label}.",
        "subjectivite_score": None,
        "subjectivite_justification": f"Erreur lors de l'analyse {model_label}.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": f"Erreur lors de l'analyse {model_label}.",
        "analysis_error": "Unknown error"
    }

    if not text or not text.strip():
        return {
            **default_error,
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte non fourni ou vide.",
            "subjectivite_justification": "Non applicable - texte vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable - texte vide.",
            "analysis_error": "Empty text"
        }

    user_prompt = create_user_prompt(text)
    last_error = "Max retries exceeded"

    for attempt in range(max_retries):
        try:
            result = llm_client.generate_structured(
                system_prompt, user_prompt, SentimentAnalysisOutput
            )
            return {**result.model_dump(), "analysis_error": None}
        except Exception as e:
            # An exhausted balance or daily quota is terminal, not transient.
            # Retrying it burns the remaining corpus against a wall: a run that
            # lost its OpenRouter credits at article ~11,500 spent hours
            # producing 823 identical "Insufficient credits" failures because
            # nothing here distinguished a 402 from a flaky backend.
            if is_quota_exhausted(e):
                raise QuotaExhaustedError(
                    f"{model_label}: {type(e).__name__}: {e}"
                ) from e
            last_error = f"{type(e).__name__}: {e}"
            logger.debug(f"{model_label} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = (
                    MODEL_RETRY_DELAYS[attempt]
                    if attempt < len(MODEL_RETRY_DELAYS)
                    else 2 ** attempt
                )
                time.sleep(delay)

    # Keep the real error: "Max retries exceeded" alone makes a bad API key and
    # a malformed response look identical when triaging a failed run.
    return {**default_error, "analysis_error": last_error}


def _timeout_result(reason: str) -> Dict[str, Any]:
    return {
        "analysis_error": reason,
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": f"Erreur: {reason}",
        "subjectivite_score": None,
        "subjectivite_justification": f"Erreur: {reason}",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": f"Erreur: {reason}",
    }


def request_timeout_for_budget(
    total_seconds: float,
    *,
    attempts: int = MODEL_MAX_ATTEMPTS,
    safety_seconds: float = 5.0,
) -> float:
    """Return the per-attempt HTTP timeout inside one model's total budget.

    Retry sleeps and a small scheduling margin are reserved first. This makes
    the outer future timeout enforceable: even the final SDK call must return
    before the future reaches its deadline.
    """
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    retry_sleep = sum(MODEL_RETRY_DELAYS[: max(0, attempts - 1)])
    available = total_seconds - retry_sleep - safety_seconds
    return max(1.0, available / attempts)


def analyze_with_all_models(
    text: str,
    llm_clients: Dict[str, BaseLLMClient],
    system_prompt: str,
    logger: logging.Logger,
    labels: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> Dict[str, Dict[str, Any]]:
    """Run sentiment analysis with all supplied models concurrently.

    Returns:
        Dictionary with model names as keys and results as values
    """
    labels = labels or {}
    results: Dict[str, Dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_clients)) as executor:
        futures = {}

        for model_name, llm_client in llm_clients.items():
            label = labels.get(model_name, model_name.capitalize())
            futures[model_name] = executor.submit(
                analyze_with_model, llm_client, text, system_prompt, label, logger
            )

        for model_name, future in futures.items():
            try:
                results[model_name] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                results[model_name] = _timeout_result("Timeout")
            except QuotaExhaustedError:
                # Must escape: the generic handler below would turn a dead
                # account into an ordinary per-item error and let the run walk
                # the rest of the corpus collecting them.
                raise
            except Exception as e:
                results[model_name] = _timeout_result(f"{type(e).__name__}: {e}")

    return results


def is_valid_result(result: Dict[str, Any]) -> bool:
    """True when a result carries a usable annotation rather than an error."""
    if result.get("analysis_error"):
        return False
    return result.get("centralite_islam_musulmans") != "ERREUR_ANALYSE"
