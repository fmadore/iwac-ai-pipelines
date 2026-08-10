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

OPENAI_SOL_MODEL = "gpt-5.6-sol"
OPENAI_TERRA_MODEL = "gpt-5.6-terra"
OPENAI_LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_MODEL = OPENAI_LUNA_MODEL
OPENAI_FULL_MODEL = OPENAI_SOL_MODEL
DEFAULT_GEMINI_FLASH = "gemini-flash-latest"
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
OPENROUTER_DEEPSEEK_FLASH_0731_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_DEEPSEEK_FLASH_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_DEEPSEEK_PRO_MODEL = "deepseek/deepseek-v4-pro"

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
    "gemini-flash": ModelOption(
        "gemini-flash", PROVIDER_GEMINI, DEFAULT_GEMINI_FLASH,
        "Gemini Flash", "Google Gemini Flash — latest stable rolling alias",
        default_thinking_level="MINIMAL",
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
    ),
    "gemini-pro": ModelOption(
        "gemini-pro", PROVIDER_GEMINI, DEFAULT_GEMINI_PRO,
        "Gemini Pro", "Google Gemini Pro — latest stable rolling alias",
        default_thinking_level="LOW",
    ),
    "gemma-4": ModelOption(
        "gemma-4", PROVIDER_GEMINI, DEFAULT_GEMMA_4,
        "Gemma 4 31B", "Google Gemma 4 31B dense open-weights flagship",
        default_thinking_level="HIGH",
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
    "gemini": "gemini-flash",
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
    "gemini-pro-latest": "gemini-pro",
    "gemini-3.1-pro-preview": "gemini-pro",
    "gemini-3-pro-preview": "gemini-pro",
    "gemma": "gemma-4",
    "gemma-4-31b": "gemma-4",
    "gemma-4-31b-it": "gemma-4",
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
    "deepseek": "deepseek-v4-flash-0731",
    "deepseek-flash": "deepseek-v4-flash-0731",
    "deepseek-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
}


TEXT_ECONOMY_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gpt-5.6-luna", "gemini-flash", "ministral-14b",
]
TEXT_OPEN_MODELS: List[str] = [
    "qwen3.5-moe", "qwen3.5-moe-small", "qwen3.5-dense",
    "deepseek-v4-flash-0731", "deepseek-v4-pro",
]
TEXT_EXTENDED_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gpt-5.6-luna", "gemini-flash", "gemma-4",
    "mistral-large", "ministral-14b", "mistral-small", "qwen3.5-moe",
]
TEXT_FULL_MODELS: List[str] = [
    DEFAULT_TEXT_MODEL_KEY, "gemini-flash", "gemini-pro", "gpt-5.6-luna",
    "gpt-5.6-sol", "mistral-large", "ministral-14b", "mistral-small",
    "qwen3.5-moe", "qwen3.5-dense", "deepseek-v4-pro",
]
GEMINI_DOCUMENT_MODELS: List[str] = ["gemini-flash", "gemini-pro", "gemma-4"]
LEGACY_CLI_MODEL_KEYS: List[str] = ["gpt-5-mini", "gpt-5.1", "gpt-5", "gpt-5-nano"]


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
