"""Dependency-free model catalog and generation configuration.

Provider SDK adapters live in :mod:`common.llm_provider`. Keeping this module
free of those optional imports makes model selection cheap to import and keeps
catalog/provenance changes separate from transport implementation changes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional


PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_MISTRAL = "mistral"
PROVIDER_OPENROUTER = "openrouter"
#: An OpenAI-compatible endpoint you run yourself — vLLM on a GPU cluster, or
#: llama.cpp / LM Studio / TGI on anything smaller. Named for the *route*, as
#: ``gemma-4-openrouter`` is: the weights may be identical to a hosted entry,
#: but who sees the text is not, and that is what the provenance record has to
#: say. Unlike every other provider here the endpoint is deployment state, not
#: catalog state, so it is resolved from the environment by the adapter and
#: never written down in this file. See ``serving/README.md``.
PROVIDER_SELFHOSTED = "selfhosted"

OPENAI_SOL_MODEL = "gpt-5.6-sol"
OPENAI_TERRA_MODEL = "gpt-5.6-terra"
OPENAI_LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_MODEL = OPENAI_LUNA_MODEL
OPENAI_FULL_MODEL = OPENAI_SOL_MODEL
DEFAULT_GEMINI_FLASH = "gemini-flash-latest"
DEFAULT_GEMINI_37_FLASH = "gemini-3.7-flash"
DEFAULT_GEMINI_36_FLASH = "gemini-3.6-flash"
DEFAULT_GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
DEFAULT_GEMINI_35_FLASH_LITE = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_31_PRO = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_PRO = "gemini-pro-latest"
DEFAULT_GEMMA_4 = "gemma-4-31b-it"
DEFAULT_MISTRAL_LARGE = "mistral-large-2512"
DEFAULT_MINISTRAL_14B = "ministral-14b-2512"
DEFAULT_MISTRAL_SMALL = "mistral-small-2603"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_QWEN_MOE_MODEL = "qwen/qwen3.5-122b-a10b"
OPENROUTER_QWEN_SMALL_MOE_MODEL = "qwen/qwen3.5-35b-a3b"
OPENROUTER_QWEN_DENSE_MODEL = "qwen/qwen3.5-27b"
OPENROUTER_QWEN38_DENSE_MODEL = "qwen/qwen3.8-27b"
OPENROUTER_GEMMA_4_31B_MODEL = "google/gemma-4-31b-it"
OPENROUTER_DEEPSEEK_FLASH_0731_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_DEEPSEEK_FLASH_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_DEEPSEEK_PRO_MODEL = "deepseek/deepseek-v4-pro"

#: What ``vllm serve Qwen/Qwen3.8-27B`` reports back from ``/v1/models``: the
#: served name defaults to the model path it was launched with, so this is the
#: Hugging Face repo id exactly. Serving under a different ``--served-model-name``
#: means changing this string too.
SELFHOSTED_QWEN38_MODEL = "Qwen/Qwen3.8-27B"

DEFAULT_TEXT_MODEL_KEY = "deepseek-v4-flash-0731"
# Deadline for one transport attempt. It exists to bound a hung socket, not to
# police slow-but-working calls: the longest legitimate text stage here is
# magazine-issue consolidation, which emits a whole table of contents in one
# structured response. 300s sits above that and below the OpenAI SDK's own 600s
# default. Stages wanting a tighter, resumable bound pass
# ``LLMConfig(request_timeout_seconds=...)`` — as the sentiment panel does.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0

OPENROUTER_PROVIDER_PREFS: Dict[str, Any] = {
    "data_collection": "deny",
    "require_parameters": True,
}
OPENROUTER_HEADERS: Dict[str, str] = {
    "HTTP-Referer": "https://github.com/fmadore/iwac-ai-pipelines",
    "X-Title": "IWAC AI Pipelines",
}


#: Google's thinking levels, weakest first. The order is what
#: :func:`clamp_thinking_level` measures distance along, so it is API contract,
#: not decoration.
THINKING_LEVELS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class ModelOption:
    key: str
    provider: str
    model: str
    label: str
    description: str
    default_temperature: Optional[float] = None
    default_reasoning_effort: Optional[str] = "low"
    supported_reasoning_efforts: tuple = ()
    default_text_verbosity: str = "low"
    default_store: bool = False
    default_thinking_level: Optional[str] = None
    #: Which of :data:`THINKING_LEVELS` this model actually accepts. Empty means
    #: all four. Google enforces this per model and rejects the rest with a 400
    #: ``INVALID_ARGUMENT``, so an unstated gap is not a soft preference — it is
    #: a pipeline that cannot make a single call. Verified against the live API,
    #: 2026-08-14.
    supported_thinking_levels: tuple = ()


@dataclass
class LLMConfig:
    """Provider-neutral generation controls inherited over model defaults."""

    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None
    store: Optional[bool] = None
    thinking_level: Optional[str] = None
    request_timeout_seconds: Optional[float] = None
    sdk_max_retries: Optional[int] = None

    def merged_over(self, base: "LLMConfig") -> "LLMConfig":
        """Return a copy where unset fields fall back to ``base``."""
        return LLMConfig(**{
            item.name: (
                getattr(self, item.name)
                if getattr(self, item.name) is not None
                else getattr(base, item.name)
            )
            for item in fields(self)
        })


MODEL_REGISTRY: Dict[str, ModelOption] = {
    # Prices are in/cached-in/out per 1M tokens, short-context tier, verified
    # against developers.openai.com/api/docs/pricing on 2026-08-06. They are
    # DESCRIPTIONS, not something the code computes with — but a stale one is not
    # harmless: the previous "$1/$6" for Luna was 5x the real rate and produced a
    # $50 corpus estimate for a job that costs ~$7. Re-check upstream before
    # quoting any of these; do not infer them from a model's tier name.
    "gpt-5.6-luna": ModelOption(
        "gpt-5.6-luna", PROVIDER_OPENAI, OPENAI_LUNA_MODEL,
        "ChatGPT (GPT-5.6 Luna)",
        "OpenAI Responses API — cost-optimized tier ($0.20/$0.02/$1.20 per 1M)",
    ),
    "gpt-5.6-terra": ModelOption(
        "gpt-5.6-terra", PROVIDER_OPENAI, OPENAI_TERRA_MODEL,
        "ChatGPT (GPT-5.6 Terra)",
        "OpenAI Responses API — balanced tier ($2/$0.20/$12 per 1M)",
    ),
    "gpt-5.6-sol": ModelOption(
        "gpt-5.6-sol", PROVIDER_OPENAI, OPENAI_SOL_MODEL,
        "ChatGPT (GPT-5.6 Sol)",
        "OpenAI Responses API — flagship tier ($5/$0.50/$30 per 1M)",
    ),
    # Gemini 3.7 Flash dropped MINIMAL: the level its two predecessors defaulted
    # to is now a 400 on this model, and because ``gemini-flash-latest`` rolled
    # onto 3.7 the same day, the rolling entry below inherited the break. LOW is
    # the floor for both. See ``supported_thinking_levels`` and
    # ``clamp_thinking_level`` — pipelines still ask for MINIMAL meaning "as
    # little as this model allows", and the clamp is what makes that true.
    "gemini-3.7-flash": ModelOption(
        "gemini-3.7-flash", PROVIDER_GEMINI, DEFAULT_GEMINI_37_FLASH,
        "Gemini 3.7 Flash", "Google Gemini 3.7 Flash — version-pinned Flash",
        default_thinking_level="LOW",
        supported_thinking_levels=("low", "medium", "high"),
    ),
    "gemini-flash": ModelOption(
        "gemini-flash", PROVIDER_GEMINI, DEFAULT_GEMINI_FLASH,
        "Gemini Flash", "Google Gemini Flash — latest stable rolling alias",
        default_thinking_level="LOW",
        supported_thinking_levels=("low", "medium", "high"),
    ),
    "gemini-3.6-flash": ModelOption(
        "gemini-3.6-flash", PROVIDER_GEMINI, DEFAULT_GEMINI_36_FLASH,
        "Gemini 3.6 Flash", "Google Gemini 3.6 Flash — version-pinned Flash",
        default_thinking_level="MINIMAL",
    ),
    "gemini-flash-lite": ModelOption(
        "gemini-flash-lite", PROVIDER_GEMINI, DEFAULT_GEMINI_FLASH_LITE,
        "Gemini Flash-Lite", "Google Gemini Flash-Lite — cheapest rolling alias",
        default_thinking_level="MINIMAL",
    ),
    "gemini-3.5-flash-lite": ModelOption(
        "gemini-3.5-flash-lite", PROVIDER_GEMINI, DEFAULT_GEMINI_35_FLASH_LITE,
        "Gemini 3.5 Flash-Lite", "Google Gemini 3.5 Flash-Lite — version-pinned",
        default_thinking_level="MINIMAL",
    ),
    "gemini-3.1-flash-lite": ModelOption(
        "gemini-3.1-flash-lite", PROVIDER_GEMINI, DEFAULT_GEMINI_31_FLASH_LITE,
        "Gemini 3.1 Flash-Lite", "Google Gemini 3.1 Flash-Lite — version-pinned",
        default_thinking_level="MINIMAL",
    ),
    "gemini-3.1-pro": ModelOption(
        "gemini-3.1-pro", PROVIDER_GEMINI, DEFAULT_GEMINI_31_PRO,
        "Gemini 3.1 Pro", "Google Gemini 3.1 Pro — version-pinned quality tier",
        default_thinking_level="LOW",
        supported_thinking_levels=("low", "medium", "high"),
    ),
    "gemini-pro": ModelOption(
        "gemini-pro", PROVIDER_GEMINI, DEFAULT_GEMINI_PRO,
        "Gemini Pro", "Google Gemini Pro — latest stable rolling alias",
        default_thinking_level="LOW",
        supported_thinking_levels=("low", "medium", "high"),
    ),
    "gemma-4": ModelOption(
        "gemma-4", PROVIDER_GEMINI, DEFAULT_GEMMA_4,
        "Gemma 4 31B", "Google Gemma 4 31B dense open-weights flagship",
        default_thinking_level="HIGH",
        supported_thinking_levels=("minimal", "high"),
    ),
    "mistral-large": ModelOption(
        "mistral-large", PROVIDER_MISTRAL, DEFAULT_MISTRAL_LARGE,
        "Mistral Large 3", "Mistral AI Large 3 — 41B active multimodal MoE",
        default_temperature=0.2,
    ),
    "ministral-14b": ModelOption(
        "ministral-14b", PROVIDER_MISTRAL, DEFAULT_MINISTRAL_14B,
        "Ministral 3 14B", "Mistral Ministral 3 14B — fast and cost-effective",
        default_temperature=0.2,
    ),
    "mistral-small": ModelOption(
        "mistral-small", PROVIDER_MISTRAL, DEFAULT_MISTRAL_SMALL,
        "Mistral Small 4", "Mistral Small 4 (2603) — hybrid reasoning model",
        default_temperature=0.3,
        default_reasoning_effort=None,
        supported_reasoning_efforts=("none", "high"),
    ),
    # The same weights as the ``gemma-4`` entry above, deliberately reached by a
    # different route. Gemma is free on the Gemini API because it is served on
    # the free tier, and free-tier content is used to improve Google's products —
    # precisely what ``OPENROUTER_PROVIDER_PREFS``' ``data_collection: "deny"``
    # exists to prevent when whole archival articles are shipped to a third
    # party. (OpenRouter's own ``:free`` variant has the same problem and is
    # filtered out by that policy anyway.) Use this key for anything that sends
    # archive text; ``gemma-4`` stays for the multimodal document work the
    # OpenAI-shaped chat API cannot do.
    #
    # Gemma 4 has two thinking levels, MINIMAL and HIGH, with nothing in
    # between — the Gemini adapter clamps to the same pair. A caller asking for
    # the panel's "medium" therefore lands on ``high`` rather than dropping to a
    # non-reasoning mode, which is the rounding Mistral Small 4 and DeepSeek V4
    # Flash 0731 already do. Temperature stays unset, as for every Google model.
    "gemma-4-openrouter": ModelOption(
        "gemma-4-openrouter", PROVIDER_OPENROUTER, OPENROUTER_GEMMA_4_31B_MODEL,
        "Gemma 4 31B (OpenRouter)",
        "Gemma 4 31B dense — Apache-2.0, routed under data_collection: deny",
        default_reasoning_effort="high",
        supported_reasoning_efforts=("minimal", "high"),
    ),
    "qwen3.5-moe": ModelOption(
        "qwen3.5-moe", PROVIDER_OPENROUTER, OPENROUTER_QWEN_MOE_MODEL,
        "Qwen3.5 122B-A10B (OpenRouter)",
        "Qwen3.5 122B-A10B — Apache-2.0, MoE 10B active",
        default_temperature=0.7,
        default_reasoning_effort=None,
        supported_reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
    ),
    "qwen3.5-moe-small": ModelOption(
        "qwen3.5-moe-small", PROVIDER_OPENROUTER, OPENROUTER_QWEN_SMALL_MOE_MODEL,
        "Qwen3.5 35B-A3B (OpenRouter)",
        "Qwen3.5 35B-A3B — Apache-2.0, MoE 3B active",
        default_temperature=0.7,
        default_reasoning_effort=None,
        supported_reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
    ),
    "qwen3.5-dense": ModelOption(
        "qwen3.5-dense", PROVIDER_OPENROUTER, OPENROUTER_QWEN_DENSE_MODEL,
        "Qwen3.5 27B (OpenRouter)", "Qwen3.5 27B dense — Apache-2.0",
        default_temperature=0.7,
        default_reasoning_effort=None,
        supported_reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
    ),
    # Qwen3.8 27B, twice: once through an endpoint you run yourself, once
    # through OpenRouter. Same weights, different route — the split
    # ``gemma-4`` / ``gemma-4-openrouter`` already makes, for the same reason:
    # the route is half of what a provenance record claims.
    #
    # Both carry temperature 1.0, which is Qwen's *thinking-mode* recipe
    # (generation_config.json ships 1.0 / top_p 0.95 / top_k 20; the model card's
    # 0.7 is the non-thinking recipe, and it is what the Qwen3.5 entries above
    # inherited). Everything here runs thinking-on, so 1.0 is the applicable
    # number — do not copy 0.7 down from the neighbours. ``top_p``/``top_k`` stay
    # unset as always; a self-hosted vLLM applies the model's own
    # generation_config server-side, which is a quiet bonus of this route.
    #
    # The ladder is genuinely graduated — low / medium / xhigh, verified on the
    # model card — which is the whole reason this model is interesting for the
    # sentiment panel: it would be the first member since GPT-5.6 Luna to sit at
    # the requested middle rather than be rounded up to it. Whether the middle
    # survives the *route* is a separate question, and the one Gemma failed
    # (see ``AI_sentiment_analysis/sentiment_core.py``); ``serving/probe_reasoning.py``
    # is what answers it. Default is ``low``, not the vendor's own ``xhigh``:
    # an unconfigured bulk run reasoning as hard as it can, on GPU hours shared
    # with a whole university, is the expensive accident to design against.
    "qwen3.8-27b-selfhosted": ModelOption(
        "qwen3.8-27b-selfhosted", PROVIDER_SELFHOSTED, SELFHOSTED_QWEN38_MODEL,
        "Qwen3.8 27B (self-hosted)",
        "Qwen3.8 27B dense — Apache-2.0, served from your own vLLM endpoint",
        default_temperature=1.0,
        default_reasoning_effort="low",
        supported_reasoning_efforts=("low", "medium", "xhigh"),
    ),
    # The hosted twin, for measuring one route against the other on the same
    # sample. In no tier on purpose: at $0.45/$3.20 per 1M it is roughly twice
    # the sentiment panel's output-cost band, which is what sent the whole
    # experiment to a GPU cluster in the first place. Reachable by its explicit
    # key and slug alias, so a pilot can ask for it and nothing else will.
    "qwen3.8-27b-openrouter": ModelOption(
        "qwen3.8-27b-openrouter", PROVIDER_OPENROUTER, OPENROUTER_QWEN38_DENSE_MODEL,
        "Qwen3.8 27B (OpenRouter)",
        "Qwen3.8 27B dense — Apache-2.0, routed under data_collection: deny",
        default_temperature=1.0,
        default_reasoning_effort="low",
        supported_reasoning_efforts=("low", "medium", "xhigh"),
    ),
    "deepseek-v4-flash-0731": ModelOption(
        "deepseek-v4-flash-0731", PROVIDER_OPENROUTER,
        OPENROUTER_DEEPSEEK_FLASH_0731_MODEL,
        "DeepSeek V4 Flash 0731 (OpenRouter)",
        "DeepSeek V4 Flash 0731 — official 284B/13B-active MoE, 1M context",
        default_temperature=1.0,
        default_reasoning_effort="low",
        supported_reasoning_efforts=("low", "high", "max"),
    ),
    # ARCHIVE ONLY — absent from every tier, so no pipeline offers it and no
    # `--model` accepts it. The entry survives so its OpenRouter slug still
    # resolves when it turns up in an old pilot payload or a git-archaeology
    # session; the sentiment annotations it wrote were deleted from Omeka on
    # 2026-08-07 and were never on the Hub. New runs take the 0731 release.
    "deepseek-v4-flash": ModelOption(
        "deepseek-v4-flash", PROVIDER_OPENROUTER, OPENROUTER_DEEPSEEK_FLASH_MODEL,
        "DeepSeek V4 Flash Preview (OpenRouter)",
        "DeepSeek V4 Flash April preview — archive only, superseded by 0731",
        default_temperature=1.0,
        default_reasoning_effort=None,
        supported_reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
    ),
    "deepseek-v4-pro": ModelOption(
        "deepseek-v4-pro", PROVIDER_OPENROUTER, OPENROUTER_DEEPSEEK_PRO_MODEL,
        "DeepSeek V4 Pro (OpenRouter)",
        "DeepSeek V4 Pro — 1.6T/49B active MoE quality tier",
        default_temperature=1.0,
        default_reasoning_effort="high",
        supported_reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
    ),
}


MODEL_ALIASES = {
    "gemini": "gemini-3.7-flash",
    "flash": "gemini-3.7-flash",
    "gemini-3.7": "gemini-3.7-flash",
    "flash-lite": "gemini-flash-lite",
    "gemini-flash-lite-latest": "gemini-flash-lite",
    "gemini-flash-lite-3.1": "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    "openai": "gpt-5.6-luna",
    "gpt-5.6": "gpt-5.6-sol",
    "sol": "gpt-5.6-sol",
    "terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
    "openai:gpt-5.6": "gpt-5.6-sol",
    "openai:gpt-5.6-sol": "gpt-5.6-sol",
    "openai:gpt-5.6-terra": "gpt-5.6-terra",
    "openai:gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5-mini": "gpt-5.6-luna",
    "openai:gpt-5-mini": "gpt-5.6-luna",
    "openai-mini": "gpt-5.6-luna",
    "gpt-5-nano": "gpt-5.6-luna",
    "gpt-5.1-mini": "gpt-5.6-luna",
    "openai:gpt-5.1-mini": "gpt-5.6-luna",
    "gpt-5.1": "gpt-5.6-sol",
    "openai:gpt-5.1": "gpt-5.6-sol",
    "gpt-5": "gpt-5.6-sol",
    "openai:gpt-5": "gpt-5.6-sol",
    "openai-5": "gpt-5.6-sol",
    "openai-5.1": "gpt-5.6-sol",
    "gemini-flash-latest": "gemini-flash",
    "gemini-3.5-flash": "gemini-flash",
    "gemini-3-flash-preview": "gemini-flash",
    "gemini-3.7-flash-preview": "gemini-3.7-flash",
    "gemini-pro-latest": "gemini-pro",
    "gemini-3.1-pro-preview": "gemini-pro",
    "gemini-3-pro-preview": "gemini-pro",
    "gemma": "gemma-4",
    "gemma-4-31b": "gemma-4",
    "gemma-4-31b-it": "gemma-4",
    # The OpenRouter slug names the OpenRouter route; the bare ids above keep
    # resolving to the Gemini one they have always meant.
    "google/gemma-4-31b-it": "gemma-4-openrouter",
    "mistral": "mistral-large",
    "mistral-large-latest": "mistral-large",
    "mistral-large-2512": "mistral-large",
    "ministral": "ministral-14b",
    "ministral-3": "ministral-14b",
    "ministral-14b-2512": "ministral-14b",
    "mistral-small-latest": "mistral-small",
    "mistral-small-2603": "mistral-small",
    "mistral-small-4": "mistral-small",
    "qwen": "qwen3.5-moe",
    "qwen3.5": "qwen3.5-moe",
    "qwen3.5-122b-a10b": "qwen3.5-moe",
    "qwen/qwen3.5-122b-a10b": "qwen3.5-moe",
    "qwen3.5-35b-a3b": "qwen3.5-moe-small",
    "qwen/qwen3.5-35b-a3b": "qwen3.5-moe-small",
    "qwen3.5-27b": "qwen3.5-dense",
    "qwen/qwen3.5-27b": "qwen3.5-dense",
    # Qwen3.8 27B resolves by route, and the two names collide once lowercased:
    # the Hugging Face repo id ``Qwen/Qwen3.8-27B`` normalizes to exactly the
    # OpenRouter slug ``qwen/qwen3.8-27b``. The vendor-prefixed form therefore
    # means the hosted route, as it does for ``google/gemma-4-31b-it`` above,
    # and the bare names mean the endpoint you run yourself. Ask for the
    # self-hosted entry by a short name, never by pasting the HF id.
    "qwen3.8": "qwen3.8-27b-selfhosted",
    "qwen3.8-27b": "qwen3.8-27b-selfhosted",
    "qwen/qwen3.8-27b": "qwen3.8-27b-openrouter",
    "deepseek": "deepseek-v4-flash-0731",
    "deepseek-flash": "deepseek-v4-flash-0731",
    "deepseek-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
}


# Every tier offers the *pinned* ``gemini-3.7-flash`` rather than the rolling
# ``gemini-flash``. A tier is what a pipeline picks when it does not name a
# model, so it is also what gets stamped into an ``iwac:*Model`` annotation —
# and a rolling alias reports its own version as the string "Gemini Flash
# Latest", which cannot be cited. The rolling entry stays in MODEL_REGISTRY for
# the pipelines that want whatever Flash is current and stamp nothing.
TEXT_ECONOMY_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gpt-5.6-luna", "gemini-3.7-flash", "ministral-14b",
]
TEXT_OPEN_MODELS: List[str] = [
    "qwen3.5-moe", "qwen3.5-moe-small", "qwen3.5-dense",
    "qwen3.8-27b-selfhosted",
    "deepseek-v4-flash-0731", "deepseek-v4-pro",
]
TEXT_EXTENDED_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gpt-5.6-luna", "gemini-3.7-flash", "gemma-4",
    "mistral-large", "ministral-14b", "mistral-small", "qwen3.5-moe",
]
TEXT_FULL_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gemini-3.7-flash", "gemini-pro", "gpt-5.6-luna",
    "gpt-5.6-sol", "mistral-large", "ministral-14b", "mistral-small",
    "qwen3.5-moe", "qwen3.5-dense", "deepseek-v4-pro",
]
# Both Gemini entries are pinned, unlike the text tiers above, because this is
# the tier ``AI_ocr_extraction/02`` picks from and its step 03 stamps
# ``iwac:ocrModel``. ``gemini-pro`` was the rolling alias here until 2026-08-14 —
# an operator who ran OCR with it had no honest answer at the write step, since
# every Pro authority item names a version the run never confirmed.
GEMINI_DOCUMENT_MODELS: List[str] = ["gemini-3.7-flash", "gemini-3.1-pro", "gemma-4"]
LEGACY_CLI_MODEL_KEYS: List[str] = ["gpt-5-mini", "gpt-5.1", "gpt-5", "gpt-5-nano"]


def supported_thinking_levels_for_model(model_id: str) -> tuple:
    """Levels ``model_id`` accepts, or ``()`` when unconstrained/unknown.

    Keyed on the provider's model id rather than the registry key, because the
    multimodal pipelines never hold a ``ModelOption`` — they pass raw ids such
    as ``gemini-pro-latest`` straight to the SDK. An id absent from the registry
    is reported unconstrained: guessing a restriction would silently downgrade a
    model nobody here has probed.
    """
    for option in MODEL_REGISTRY.values():
        if option.model == model_id and option.supported_thinking_levels:
            return option.supported_thinking_levels
    return ()


def clamp_thinking_level(model_id: str, level: Optional[str]) -> Optional[str]:
    """Snap ``level`` to the nearest level ``model_id`` actually accepts.

    Pipelines ask for a level meaning "roughly this much deliberation", and the
    models disagree about which rungs exist: Gemini 3.7 Flash and every Pro drop
    MINIMAL, Gemma 4 offers only MINIMAL and HIGH. Rejecting the request would
    turn a vendor's ladder change into a dead pipeline — Gemini 3.7 Flash landing
    on ``gemini-flash-latest`` broke OCR, HTR, audio, video and every text tier
    at once, because all of them asked for a MINIMAL that had ceased to exist.

    Ties round *up*: a level the model cannot serve becomes more deliberation,
    never a silent drop to none. Returns lowercase; callers case it as their SDK
    wants.
    """
    if level is None:
        return None
    requested = str(level).strip().lower()
    supported = supported_thinking_levels_for_model(model_id)
    if not supported or requested in supported:
        return requested
    if requested not in THINKING_LEVELS:
        return requested  # unknown name — let the provider report it
    target = THINKING_LEVELS.index(requested)
    return min(
        supported,
        key=lambda name: (abs(THINKING_LEVELS.index(name) - target),
                          -THINKING_LEVELS.index(name)),
    )


def normalize_model_key(model_key: Optional[str]) -> Optional[str]:
    if not model_key:
        return None
    key = model_key.strip().lower()
    return MODEL_ALIASES.get(key, key)


def get_model_option(
    model_key: Optional[str], allowed_keys: Optional[List[str]] = None
) -> ModelOption:
    normalized = normalize_model_key(model_key)
    normalized_allowed = (
        [normalize_model_key(key) for key in allowed_keys] if allowed_keys else None
    )
    if normalized and normalized in MODEL_REGISTRY:
        if normalized_allowed and normalized not in normalized_allowed:
            raise ValueError(
                f"Model '{model_key}' not allowed. Choose from: {', '.join(allowed_keys)}"
            )
        return MODEL_REGISTRY[normalized]
    if normalized:
        raise ValueError(f"Unsupported model key: {model_key}")
    return prompt_for_model_choice(allowed_keys=normalized_allowed)


def prompt_for_model_choice(allowed_keys: Optional[List[str]] = None) -> ModelOption:
    options = (
        [MODEL_REGISTRY[key] for key in allowed_keys if key in MODEL_REGISTRY]
        if allowed_keys else list(MODEL_REGISTRY.values())
    )
    print("Select AI model:")
    for index, option in enumerate(options, start=1):
        print(f"  {index}) {option.label} - {option.description}")
    while True:
        choice = input("Enter choice number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please select a valid option.")


def summary_from_option(option: ModelOption) -> str:
    return f"{option.label} ({option.model})"
