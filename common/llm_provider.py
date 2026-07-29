"""Shared helpers for selecting and calling Large Language Models
(OpenAI / Gemini / Mistral / OpenRouter).

This module centralizes provider/model selection so individual pipelines only need to
focus on their prompts. Adding new models or tweaking API settings now only requires
changing this file.
"""
from __future__ import annotations

import json
import os
import logging
import re
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv

# Type variable for Pydantic models
T = TypeVar('T')

# Optional imports (scripts should still run if a provider is not installed)
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - import guard
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover - import guard
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from mistralai.client import Mistral  # type: ignore
except Exception:  # pragma: no cover - import guard
    Mistral = None  # type: ignore

# Optional Pydantic import for structured outputs
try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover - import guard
    BaseModel = None  # type: ignore

load_dotenv()

LOGGER = logging.getLogger(__name__)

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_MISTRAL = "mistral"
PROVIDER_OPENROUTER = "openrouter"

# OpenAI GPT-5.6 family (released 2026-07-09): three durable capability tiers
# replacing the old numbered lineup. There is no mini/nano variant in this
# generation. The bare "gpt-5.6" id routes to Sol.
OPENAI_SOL_MODEL = "gpt-5.6-sol"      # flagship   — $5 / $30 per 1M tokens
OPENAI_TERRA_MODEL = "gpt-5.6-terra"  # balanced   — $2.50 / $15 per 1M tokens
OPENAI_LUNA_MODEL = "gpt-5.6-luna"    # high-volume — $1 / $6 per 1M tokens
DEFAULT_OPENAI_MODEL = OPENAI_LUNA_MODEL
OPENAI_FULL_MODEL = OPENAI_SOL_MODEL
DEFAULT_GEMINI_FLASH = "gemini-flash-latest"  # rolling alias -> newest stable Flash
DEFAULT_GEMINI_FLASH_LITE = "gemini-flash-lite-latest"  # rolling alias -> newest stable Flash-Lite
DEFAULT_GEMINI_PRO = "gemini-pro-latest"  # rolling alias -> newest stable Pro
DEFAULT_GEMMA_4 = "gemma-4-31b-it"
DEFAULT_MISTRAL_LARGE = "mistral-large-2512"
DEFAULT_MINISTRAL_14B = "ministral-14b-2512"

# ---------------------------------------------------------------------------
# OpenRouter
#
# OpenRouter is a router, not a lab: one API key and one OpenAI-compatible
# endpoint in front of the open-weights models (Qwen, DeepSeek, ...) that the
# three first-party SDKs above do not serve. It exists here so a francophone
# corpus can be run against open models at a fraction of GPT/Gemini prices,
# and so a fresh clone needs one key instead of three.
#
# Model ids are OpenRouter slugs, unversioned on purpose: like the Gemini
# ``-latest`` aliases, they follow the vendor's current pointer.
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_QWEN_FLASH_MODEL = "qwen/qwen3.7-flash"
OPENROUTER_DEEPSEEK_FLASH_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_DEEPSEEK_PRO_MODEL = "deepseek/deepseek-v4-pro"

#: Routing policy applied to *every* OpenRouter request.
#:
#: ``data_collection: "deny"`` is the important one. OpenRouter dispatches to
#: third-party inference backends and defaults to "allow", i.e. backends that
#: may retain or train on the payload. These pipelines send whole archival
#: documents — the same reason ``ModelOption.default_store`` is False for
#: OpenAI — so restrict routing to backends that do not collect user data.
#:
#: ``require_parameters: True`` keeps structured output honest: json_schema
#: support varies per backend, and without this a request can be routed to one
#: that silently ignores ``response_format`` and returns prose.
OPENROUTER_PROVIDER_PREFS: Dict[str, Any] = {
    "data_collection": "deny",
    "require_parameters": True,
}

#: Optional attribution headers; OpenRouter shows them on the account's
#: activity page, which makes a runaway pipeline easy to spot.
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
    default_temperature: float = 0.2
    # OpenAI-specific defaults
    # GPT-5.6: none/low/medium/high/xhigh/max. None means "send no reasoning
    # parameter at all" — used by the OpenRouter entries, where the accepted
    # effort values differ per model and an unsupported one narrows routing.
    default_reasoning_effort: Optional[str] = "low"
    # OpenRouter-only: the effort values this model accepts. Empty means the
    # model takes no reasoning parameter, so none is sent. A requested effort
    # outside this set falls back to ``default_reasoning_effort`` rather than
    # being forwarded, because ``require_parameters`` would otherwise strand
    # the request with no eligible backend. Ignored by the first-party clients,
    # whose SDKs validate the value themselves.
    supported_reasoning_efforts: tuple = ()
    default_text_verbosity: str = "low"    # "low", "medium", "high"
    # Do not retain request/response bodies server-side by default: these
    # pipelines send full archival documents, and there is no need for a
    # 30-day copy of the collection on the provider's side.
    default_store: bool = False
    # Gemini-specific defaults
    default_thinking_level: Optional[str] = None  # For Gemini 3: Flash="minimal"/"low"/"medium"/"high", Pro="low"/"high"

@dataclass
class LLMConfig:
    """Configuration for LLM generation requests.
    
    Scripts can create instances of this class to customize behavior per use case.
    
    OpenAI parameters:
        reasoning_effort: controls reasoning depth. GPT-5.6 accepts "none", "low",
                          "medium", "high", "xhigh" or "max" (API default "medium";
                          this project defaults to "low" for cost). "none" makes the
                          model behave like a non-reasoning one — the cheapest option
                          for mechanical work (OCR correction, summarization). When
                          migrating, OpenAI advises testing your current level and one
                          lower — GPT-5.6 often holds quality with fewer reasoning tokens.
        text_verbosity: "low", "medium", or "high" - controls response length
        store: whether OpenAI retains the request/response server-side (default
               False; these pipelines send full archival documents)

    Gemini parameters:
        temperature: 0.0-1.0 - controls randomness (OpenAI ignores this)
        
        For Gemini 3 (Flash and Pro):
            thinking_level: Controls reasoning depth (cannot be disabled)
                           Flash: "MINIMAL", "LOW", "MEDIUM", or "HIGH"
                           Pro: "LOW" or "HIGH" only

        For Gemma 4:
            thinking_level: "MINIMAL" or "HIGH" only (LOW/MEDIUM are remapped)
    
    Example:
        # High-quality reasoning for complex NER
        config = LLMConfig(reasoning_effort="high", text_verbosity="medium")

        # Fast OCR with Gemini Pro (low thinking)
        config = LLMConfig(thinking_level="low", temperature=0.1)

        # Fast OCR correction with Gemini Flash (minimal thinking)
        config = LLMConfig(thinking_level="minimal", temperature=0.1)
    """
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None
    store: Optional[bool] = None
    thinking_level: Optional[str] = None  # Gemini 3: Flash="minimal"/"low"/"medium"/"high", Pro="low"/"high"

    def merged_over(self, base: "LLMConfig") -> "LLMConfig":
        """Return a copy where unset (None) fields fall back to ``base``."""
        return LLMConfig(**{
            f.name: getattr(self, f.name) if getattr(self, f.name) is not None else getattr(base, f.name)
            for f in fields(self)
        })

MODEL_REGISTRY: Dict[str, ModelOption] = {
    "gpt-5.6-luna": ModelOption(
        key="gpt-5.6-luna",
        provider=PROVIDER_OPENAI,
        model=OPENAI_LUNA_MODEL,
        label="ChatGPT (GPT-5.6 Luna)",
        description="OpenAI Responses API — cost-optimized tier ($1/$6 per 1M tokens)"
    ),
    "gpt-5.6-terra": ModelOption(
        key="gpt-5.6-terra",
        provider=PROVIDER_OPENAI,
        model=OPENAI_TERRA_MODEL,
        label="ChatGPT (GPT-5.6 Terra)",
        description="OpenAI Responses API — balanced tier ($2.50/$15 per 1M tokens)"
    ),
    "gpt-5.6-sol": ModelOption(
        key="gpt-5.6-sol",
        provider=PROVIDER_OPENAI,
        model=OPENAI_SOL_MODEL,
        label="ChatGPT (GPT-5.6 Sol)",
        description="OpenAI Responses API — flagship tier ($5/$30 per 1M tokens)"
    ),
    "gemini-flash": ModelOption(
        key="gemini-flash",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_FLASH,
        label="Gemini Flash",
        description="Google Gemini Flash — latest stable (rolling alias gemini-flash-latest), fast & cost-effective",
        default_thinking_level="MINIMAL"  # Flash supports minimal/low/medium/high
    ),
    "gemini-flash-lite": ModelOption(
        key="gemini-flash-lite",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_FLASH_LITE,
        label="Gemini Flash-Lite",
        description="Google Gemini Flash-Lite — latest stable (rolling alias gemini-flash-lite-latest), cheapest/lowest latency",
        default_thinking_level="MINIMAL"  # Flash-Lite uses thinking_level; minimal keeps it cheap
    ),
    "gemini-pro": ModelOption(
        key="gemini-pro",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_PRO,
        label="Gemini Pro",
        description="Google Gemini Pro — latest stable (rolling alias gemini-pro-latest), highest quality",
        default_thinking_level="LOW"  # Pro supports LOW or HIGH
    ),
    "gemma-4": ModelOption(
        key="gemma-4",
        provider=PROVIDER_GEMINI,  # Served via the Gemini API (same google-genai client)
        model=DEFAULT_GEMMA_4,
        label="Gemma 4 31B",
        description="Google Gemma 4 31B dense — open-weights flagship, via Gemini API",
        default_thinking_level="HIGH"  # Gemma 4 supports only MINIMAL or HIGH
    ),
    "mistral-large": ModelOption(
        key="mistral-large",
        provider=PROVIDER_MISTRAL,
        model=DEFAULT_MISTRAL_LARGE,
        label="Mistral Large 3",
        description="Mistral AI Large 3 — 41B active params, multimodal MoE",
        default_temperature=0.2
    ),
    "ministral-14b": ModelOption(
        key="ministral-14b",
        provider=PROVIDER_MISTRAL,
        model=DEFAULT_MINISTRAL_14B,
        label="Ministral 3 14B",
        description="Mistral Ministral 3 14B — fast, cost-effective ($0.2/M tokens)",
        default_temperature=0.2
    ),
    # OpenRouter-served open-weights models. The two Flash tiers cost roughly a
    # tenth of gpt-5.6-luna, which is what makes a full-corpus NER or sentiment
    # pass affordable; Pro is the quality tier for the harder pipelines.
    "qwen3.7-flash": ModelOption(
        key="qwen3.7-flash",
        provider=PROVIDER_OPENROUTER,
        model=OPENROUTER_QWEN_FLASH_MODEL,
        label="Qwen3.7 Flash (OpenRouter)",
        description="Alibaba Qwen3.7 Flash — 1M context, strong multilingual ($0.03/$0.13 per 1M tokens)",
        default_temperature=0.2,
        # No reasoning parameter is sent: the accepted values are not published
        # per-backend, and non-thinking is the cheapest mode for mechanical work.
        default_reasoning_effort=None,
    ),
    "deepseek-v4-flash": ModelOption(
        key="deepseek-v4-flash",
        provider=PROVIDER_OPENROUTER,
        model=OPENROUTER_DEEPSEEK_FLASH_MODEL,
        label="DeepSeek V4 Flash (OpenRouter)",
        description="DeepSeek V4 Flash — 284B/13B active MoE, 1M context ($0.09/$0.18 per 1M tokens)",
        default_temperature=0.2,
        # V4 Flash is a hybrid thinking/non-thinking model: default to
        # non-thinking (cheapest for mechanical work), but honour an explicit
        # high/xhigh from a caller that wants the reasoning path.
        default_reasoning_effort=None,
        supported_reasoning_efforts=("high", "xhigh"),
    ),
    "deepseek-v4-pro": ModelOption(
        key="deepseek-v4-pro",
        provider=PROVIDER_OPENROUTER,
        model=OPENROUTER_DEEPSEEK_PRO_MODEL,
        label="DeepSeek V4 Pro (OpenRouter)",
        description="DeepSeek V4 Pro — 1.6T/49B active MoE flagship, 1M context ($0.435/$0.87 per 1M tokens)",
        default_temperature=0.2,
        # Quality tier: reasoning on by default. xhigh maps to max reasoning.
        default_reasoning_effort="high",
        supported_reasoning_efforts=("high", "xhigh"),
    ),
}

MODEL_ALIASES = {
    "gemini": "gemini-flash",
    # Gemini Flash-Lite aliases
    "flash-lite": "gemini-flash-lite",
    "gemini-flash-lite-latest": "gemini-flash-lite",
    "gemini-flash-lite-3.1": "gemini-flash-lite",
    "gemini-3.1-flash-lite": "gemini-flash-lite",  # pinned (explicit version)
    "gemini-3.1-flash-lite-preview": "gemini-flash-lite",  # legacy preview id
    # OpenAI GPT-5.6 tier aliases
    "openai": "gpt-5.6-luna",  # Generic name -> cost-optimized tier
    "gpt-5.6": "gpt-5.6-sol",  # Bare id routes to Sol, per OpenAI
    "sol": "gpt-5.6-sol",
    "terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
    "openai:gpt-5.6": "gpt-5.6-sol",
    "openai:gpt-5.6-sol": "gpt-5.6-sol",
    "openai:gpt-5.6-terra": "gpt-5.6-terra",
    "openai:gpt-5.6-luna": "gpt-5.6-luna",
    # Legacy OpenAI keys. The GPT-5/5.1 snapshots shut down 2026-10-23; these
    # keep older invocations, docs and muscle memory working.
    "gpt-5-mini": "gpt-5.6-luna",
    "openai:gpt-5-mini": "gpt-5.6-luna",
    "openai-mini": "gpt-5.6-luna",
    "gpt-5-nano": "gpt-5.6-luna",
    "gpt-5.1-mini": "gpt-5.6-luna",  # Legacy incorrect naming
    "openai:gpt-5.1-mini": "gpt-5.6-luna",
    "gpt-5.1": "gpt-5.6-sol",  # Previous flagship
    "openai:gpt-5.1": "gpt-5.6-sol",
    "gpt-5": "gpt-5.6-sol",
    "openai:gpt-5": "gpt-5.6-sol",
    "openai-5": "gpt-5.6-sol",  # Legacy key name
    "openai-5.1": "gpt-5.6-sol",  # Legacy key name
    "gemini-flash-latest": "gemini-flash",
    "gemini-3.5-flash": "gemini-flash",  # pinned (explicit version)
    "gemini-3-flash-preview": "gemini-flash",  # legacy (Gemini 3 Flash)
    "gemini-pro-latest": "gemini-pro",
    "gemini-3.1-pro-preview": "gemini-pro",  # pinned (explicit version)
    "gemini-3-pro-preview": "gemini-pro",  # legacy
    # Gemma 4 aliases
    "gemma": "gemma-4",
    "gemma-4-31b": "gemma-4",
    "gemma-4-31b-it": "gemma-4",
    # Mistral aliases
    "mistral": "mistral-large",
    "mistral-large-latest": "mistral-large",
    "mistral-large-2512": "mistral-large",
    # Ministral aliases
    "ministral": "ministral-14b",
    "ministral-3": "ministral-14b",
    "ministral-14b-2512": "ministral-14b",
    # OpenRouter aliases. The full slugs are accepted so a model id copied
    # straight off openrouter.ai resolves without translation.
    "qwen": "qwen3.7-flash",
    "qwen-flash": "qwen3.7-flash",
    "qwen3.7": "qwen3.7-flash",
    "qwen/qwen3.7-flash": "qwen3.7-flash",
    "deepseek": "deepseek-v4-flash",
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
}

# ---------------------------------------------------------------------------
# Model tiers
#
# Pipelines used to each carry their own ``ALLOWED_MODELS`` literal, so adding
# or retiring a model meant grepping for the old key across every pipeline,
# README and .env.example. Pick a tier here instead; the lists below reproduce
# what each pipeline previously declared, so interactive menu ordering is
# unchanged.
# ---------------------------------------------------------------------------

#: Cost-optimized tiers. Enough for mechanical work: summarization, correction.
TEXT_ECONOMY_MODELS: List[str] = ["gpt-5.6-luna", "gemini-flash", "ministral-14b"]

#: Open-weights models served through OpenRouter (one OPENROUTER_API_KEY).
TEXT_OPEN_MODELS: List[str] = ["qwen3.7-flash", "deepseek-v4-flash", "deepseek-v4-pro"]

#: Economy tiers plus the open-weights and flagship Mistral options (NER).
TEXT_EXTENDED_MODELS: List[str] = [
    "gpt-5.6-luna", "gemini-flash", "gemma-4", "mistral-large", "ministral-14b",
    "qwen3.7-flash", "deepseek-v4-flash",
]

#: Every text model, including the quality tiers, for output-quality-critical work.
TEXT_FULL_MODELS: List[str] = [
    "gemini-flash", "gemini-pro", "gpt-5.6-luna", "gpt-5.6-sol",
    "mistral-large", "ministral-14b",
    "qwen3.7-flash", "deepseek-v4-flash", "deepseek-v4-pro",
]

#: Models served via the Gemini API that accept native PDF/vision input.
GEMINI_DOCUMENT_MODELS: List[str] = ["gemini-flash", "gemini-pro", "gemma-4"]

#: Retired keys still accepted on the CLI; ``normalize_model_key`` maps them forward.
LEGACY_CLI_MODEL_KEYS: List[str] = ["gpt-5-mini", "gpt-5.1", "gpt-5", "gpt-5-nano"]


class BaseLLMClient:
    """Minimal interface implemented by provider-specific clients."""

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        self.option = option
        # Fill in defaults from ModelOption when not explicitly set
        model_defaults = LLMConfig(
            temperature=option.default_temperature,
            reasoning_effort=option.default_reasoning_effort,
            text_verbosity=option.default_text_verbosity,
            store=option.default_store,
            thinking_level=option.default_thinking_level,
        )
        self.config = (config or LLMConfig()).merged_over(model_defaults)

    def _get_effective_config(self, config: Optional[LLMConfig]) -> LLMConfig:
        """Merge a per-request config with client defaults."""
        if not config:
            return self.config
        return config.merged_over(self.config)

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        """Generate content with optional per-request config override.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User's input/question
            config: Optional config to override client defaults for this request only
        
        Returns:
            Generated text response
        """
        raise NotImplementedError

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output conforming to a Pydantic schema.
        
        This method uses native structured output support from both OpenAI and Gemini APIs
        to guarantee valid JSON matching your schema. No manual JSON parsing needed.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User's input/question
            response_schema: A Pydantic BaseModel class defining the expected output structure
            config: Optional config to override client defaults for this request only
        
        Returns:
            Instance of response_schema populated with model's response
        
        Example:
            from pydantic import BaseModel
            from typing import List
            
            class NERResult(BaseModel):
                persons: List[str]
                organizations: List[str]
                locations: List[str]
                subjects: List[str]
            
            result = client.generate_structured(
                system_prompt="Extract named entities...",
                user_prompt=text_content,
                response_schema=NERResult
            )
            print(result.persons)  # Typed access to extracted data
        """
        raise NotImplementedError

class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        super().__init__(option, config)
        self._client = OpenAI()

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        
        # Use configured values or model defaults
        reasoning_effort = effective_config.reasoning_effort
        text_verbosity = effective_config.text_verbosity
        
        LOGGER.debug(
            f"OpenAI request with reasoning_effort={reasoning_effort}, text_verbosity={text_verbosity}"
        )
        
        response = self._client.responses.create(
            model=self.option.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {"type": "text"},
                "verbosity": text_verbosity,
            },
            reasoning={"effort": reasoning_effort},
            tools=[],
            store=bool(effective_config.store),
        )
        raw_output = getattr(response, "output_text", None)
        if raw_output:
            return raw_output.strip()
        segments: List[str] = []
        for seg in getattr(response, "output", []) or []:
            if isinstance(seg, dict):
                content = seg.get("content")
                if isinstance(content, str):
                    segments.append(content)
        return "\n".join(filter(None, segments)).strip()

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output using OpenAI's native JSON schema support.

        Delegates to ``responses.parse(text_format=...)`` rather than building the
        JSON schema by hand. That matters: OpenAI's ``strict`` mode requires
        ``additionalProperties: false`` on every object and *every* property listed
        in ``required``, and ``model_json_schema()`` emits neither — it omits any
        field with a default from ``required``. Sending that raw schema with
        ``strict: true`` is rejected by the API, which the callers' retry loops then
        swallow as a generic failure. ``parse()`` runs the SDK's own
        ``to_strict_json_schema()`` transform, so the schema is always valid.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")

        effective_config = self._get_effective_config(config)
        reasoning_effort = effective_config.reasoning_effort
        text_verbosity = effective_config.text_verbosity

        LOGGER.debug(
            f"OpenAI structured request with schema={response_schema.__name__}, "
            f"reasoning_effort={reasoning_effort}"
        )

        response = self._client.responses.parse(
            model=self.option.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_schema,
            text={"verbosity": text_verbosity},
            reasoning={"effort": reasoning_effort},
            store=bool(effective_config.store),
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return parsed

        # A structured request can come back refused rather than parsed; surface
        # the reason instead of a bare "no output".
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                refusal = getattr(content, "refusal", None)
                if refusal:
                    raise ValueError(f"OpenAI refused the structured request: {refusal}")

        raise ValueError("No output received from OpenAI structured response")

class GeminiGenerateContentClient(BaseLLMClient):
    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if genai is None:
            raise RuntimeError("google-genai package is not installed")
        api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")):
            try:
                self._client = genai.Client()
                LOGGER.info("Gemini client initialized via ADC.")
            except Exception as exc:  # pragma: no cover - ADC fallback
                LOGGER.warning("ADC init failed: %s; falling back to API key", exc)
        if self._client is None:
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        super().__init__(option, config)

    def _build_generation_config(self, effective_config: LLMConfig) -> Any:
        """Build Gemini generation config with thinking support.
        
        All Gemini 3 models (Flash and Pro) use thinking_level ("MINIMAL", "LOW", or "HIGH").
        Thinking cannot be disabled for Gemini 3 models.
        """
        temp = effective_config.temperature
        gen_config_kwargs: Dict[str, Any] = {"temperature": temp}
        
        if genai_types is None:
            return gen_config_kwargs
        
        try:
            # All Gemini 3 models use thinking_level (cannot be disabled).
            # Gemma 4 also supports thinking_level via ThinkingConfig, but only
            # accepts "MINIMAL" or "HIGH" (no LOW/MEDIUM).
            thinking_level = effective_config.thinking_level
            if thinking_level is None:
                # Fallback based on model type
                model_lower = self.option.model.lower()
                is_pro_model = "pro" in model_lower
                is_gemma_model = "gemma" in model_lower
                if is_gemma_model:
                    thinking_level = "HIGH"  # Gemma 4 only supports MINIMAL or HIGH
                else:
                    thinking_level = "LOW" if is_pro_model else "MINIMAL"

            # Clamp unsupported values for Gemma 4 (only MINIMAL / HIGH accepted)
            if "gemma" in self.option.model.lower():
                requested = str(thinking_level).upper()
                if requested not in ("MINIMAL", "HIGH"):
                    # Map LOW/MEDIUM to the nearest supported tier
                    thinking_level = "HIGH" if requested in ("MEDIUM", "HIGH") else "MINIMAL"
                    LOGGER.debug(
                        "Gemma only supports MINIMAL/HIGH thinking_level; "
                        "mapped %s -> %s", requested, thinking_level
                    )
            
            # Normalize to uppercase for SDK compatibility (scripts can pass any case)
            thinking_level = thinking_level.upper()
            thinking_config = genai_types.ThinkingConfig(thinking_level=thinking_level)
            gen_config_kwargs["thinking_config"] = thinking_config
            LOGGER.debug(f"Gemini 3 request with thinking_level={thinking_level}, temperature={temp}")
        except Exception as exc:  # pragma: no cover - optional field
            LOGGER.warning("Failed to configure thinking mode: %s", exc)
        
        return gen_config_kwargs

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        gen_config_kwargs = self._build_generation_config(effective_config)
        
        # Use system_instruction parameter for system prompts (modern API)
        gen_config_kwargs["system_instruction"] = system_prompt
        
        if genai_types is None:
            raise RuntimeError("google-genai package is required for Gemini generation")
        try:
            gen_config = genai_types.GenerateContentConfig(**gen_config_kwargs)
        except Exception as exc:
            # Never fall back to config=None: that would silently drop the
            # system prompt and temperature and produce plausible-but-wrong output.
            raise RuntimeError(f"Failed to build Gemini generation config: {exc}") from exc

        response = self._client.models.generate_content(
            model=self.option.model,
            contents=user_prompt,
            config=gen_config,
        )
        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output using Gemini's native JSON schema support.
        
        Uses response_mime_type='application/json' and response_schema with the Pydantic
        class directly to guarantee valid JSON matching the provided schema.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")
        if genai_types is None:
            raise RuntimeError("google-genai package is required for structured outputs")
        
        effective_config = self._get_effective_config(config)
        gen_config_kwargs = self._build_generation_config(effective_config)
        
        # Use system_instruction and pass Pydantic model directly to response_schema
        gen_config_kwargs["system_instruction"] = system_prompt
        gen_config_kwargs["response_mime_type"] = "application/json"
        gen_config_kwargs["response_schema"] = response_schema  # Pass Pydantic class directly
        
        LOGGER.debug(
            f"Gemini structured request with schema={response_schema.__name__}, "
            f"temperature={gen_config_kwargs.get('temperature')}"
        )
        
        try:
            gen_config = genai_types.GenerateContentConfig(**gen_config_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure Gemini structured output: {exc}") from exc
        
        response = self._client.models.generate_content(
            model=self.option.model,
            contents=user_prompt,
            config=gen_config,
        )
        
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("No output received from Gemini structured response")
        
        # Parse and validate with Pydantic
        return response_schema.model_validate_json(text.strip())


class MistralClient(BaseLLMClient):
    """Mistral AI client using the mistralai SDK."""

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if Mistral is None:
            raise RuntimeError("mistralai package is not installed")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        super().__init__(option, config)
        self._client = Mistral(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature
        
        LOGGER.debug(f"Mistral request with temperature={temp}")
        
        response = self._client.chat.complete(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.strip() if content else ""
        return ""

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output using Mistral's native JSON schema support.
        
        Uses client.chat.parse() with Pydantic model to guarantee valid JSON
        matching the provided schema.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")
        
        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature
        
        LOGGER.debug(
            f"Mistral structured request with schema={response_schema.__name__}, "
            f"temperature={temp}"
        )
        
        response = self._client.chat.parse(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_schema,
            temperature=temp,
        )
        
        # The parse method returns a parsed object directly
        if response.choices and len(response.choices) > 0:
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        
        raise ValueError("No output received from Mistral structured response")


class OpenRouterClient(BaseLLMClient):
    """OpenRouter client, driven through the OpenAI SDK's chat-completions API.

    OpenRouter speaks the OpenAI wire format, so no extra dependency is needed —
    only a different ``base_url`` and key. Two differences from
    ``OpenAIResponsesClient`` are worth knowing:

    * It is chat-completions, not the Responses API, so there is no
      ``verbosity`` and no ``store`` flag. Retention is governed instead by the
      ``data_collection: "deny"`` routing preference in
      ``OPENROUTER_PROVIDER_PREFS``.
    * OpenRouter-specific parameters (``provider``, ``reasoning``) are not in
      the OpenAI SDK's typed signature and travel in ``extra_body``.
    """

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        super().__init__(option, config)
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _resolve_reasoning_effort(self, effective_config: LLMConfig) -> Optional[str]:
        """Pick the reasoning effort to send, or None to send none at all.

        ``LLMConfig`` is shared across providers, so a pipeline tuned for
        OpenAI (NER asks for "medium") reaches these models too. Forwarding an
        effort the model does not accept is worse than dropping it: with
        ``require_parameters`` on it can leave the request with no eligible
        backend. So a requested value is honoured only when the model declares
        it, and otherwise degrades to the model's own default.
        """
        requested = effective_config.reasoning_effort
        supported = self.option.supported_reasoning_efforts
        if requested and requested in supported:
            return requested
        if requested and requested != self.option.default_reasoning_effort:
            LOGGER.debug(
                "%s does not accept reasoning effort %r (accepts %s); using %r",
                self.option.model, requested, ", ".join(supported) or "none",
                self.option.default_reasoning_effort,
            )
        return self.option.default_reasoning_effort

    def _extra_body(self, effective_config: LLMConfig) -> Dict[str, Any]:
        """Build the OpenRouter-only part of the request body."""
        body: Dict[str, Any] = {"provider": dict(OPENROUTER_PROVIDER_PREFS)}
        effort = self._resolve_reasoning_effort(effective_config)
        if effort:
            body["reasoning"] = {"effort": effort}
        return body

    def _parse_endpoint(self) -> Callable[..., Any]:
        """Resolve ``chat.completions.parse`` across supported SDK versions.

        The helper moved out of ``client.beta`` during the openai 1.x line, and
        pyproject allows anything from 1.60 up, so both homes must be tried.
        """
        parse = getattr(self._client.chat.completions, "parse", None)
        if parse is not None:
            return parse
        beta_chat = getattr(getattr(self._client, "beta", None), "chat", None)
        parse = getattr(getattr(beta_chat, "completions", None), "parse", None)
        if parse is None:  # pragma: no cover - very old SDK
            raise RuntimeError(
                "Installed openai SDK exposes no chat.completions.parse; "
                "upgrade to openai>=1.60 for structured outputs"
            )
        return parse

    @staticmethod
    def _message_text(message: Any) -> str:
        """Return a message's answer text, ignoring any reasoning trace.

        Reasoning models on OpenRouter put the chain of thought in
        ``reasoning``/``reasoning_details`` and the answer in ``content``; only
        the latter is the result.
        """
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        # Some backends return content as a list of typed parts.
        if isinstance(content, list):
            parts = [
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            ]
            return "".join(filter(None, parts))
        return ""

    def _first_message(self, response: Any) -> Any:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError(f"No output received from OpenRouter ({self.option.model})")
        return choices[0].message

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature

        LOGGER.debug(
            "OpenRouter request model=%s temperature=%s reasoning_effort=%s",
            self.option.model, temp, effective_config.reasoning_effort,
        )

        response = self._client.chat.completions.create(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            extra_body=self._extra_body(effective_config),
            extra_headers=OPENROUTER_HEADERS,
        )
        return self._message_text(self._first_message(response)).strip()

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output via OpenRouter's ``json_schema`` support.

        Goes through the SDK's ``parse()`` helper for the same reason the OpenAI
        client does: it runs ``to_strict_json_schema()``, and a hand-built
        ``model_json_schema()`` is rejected under ``strict``.

        Unlike first-party OpenAI, the parsed object cannot be relied on. Open
        models reached through the router routinely return schema-valid JSON as
        a plain string — sometimes inside a ``` fence — which leaves
        ``message.parsed`` as None. Validating the raw content is the fallback,
        so a well-formed answer is not thrown away over its packaging.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")

        effective_config = self._get_effective_config(config)

        LOGGER.debug(
            "OpenRouter structured request model=%s schema=%s temperature=%s",
            self.option.model, response_schema.__name__, effective_config.temperature,
        )

        response = self._parse_endpoint()(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_schema,
            temperature=effective_config.temperature,
            extra_body=self._extra_body(effective_config),
            extra_headers=OPENROUTER_HEADERS,
        )

        message = self._first_message(response)

        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            return parsed

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"OpenRouter model refused the structured request: {refusal}")

        text = self._message_text(message).strip()
        if not text:
            raise ValueError(
                f"No output received from OpenRouter structured response ({self.option.model})"
            )
        return response_schema.model_validate_json(_extract_json_payload(text))


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_payload(text: str) -> str:
    """Pull the JSON document out of a model response.

    Only needed for the OpenRouter path: several open models wrap their answer
    in a Markdown fence or prepend a sentence even when a JSON schema was
    requested. Returns ``text`` unchanged when it already parses, so a
    well-behaved response is never rewritten.
    """
    candidate = text.strip()
    try:
        json.loads(candidate)
        return candidate
    except ValueError:
        pass

    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            json.loads(inner)
            return inner
        except ValueError:
            candidate = inner

    # Last resort: the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            span = candidate[start:end + 1]
            try:
                json.loads(span)
                return span
            except ValueError:
                continue

    # Nothing parsed; hand the original back so Pydantic raises the real error.
    return text


def normalize_model_key(model_key: Optional[str]) -> Optional[str]:
    if not model_key:
        return None
    key = model_key.strip().lower()
    return MODEL_ALIASES.get(key, key)

def get_model_option(model_key: Optional[str], allowed_keys: Optional[List[str]] = None) -> ModelOption:
    """Get model option by key or prompt user for selection.
    
    Args:
        model_key: Model key string (e.g., 'openai', 'gemini-flash')
        allowed_keys: Optional list of allowed model keys to restrict choices
    
    Returns:
        Selected ModelOption
    """
    normalized = normalize_model_key(model_key)
    normalized_allowed = [normalize_model_key(key) for key in allowed_keys] if allowed_keys else None
    if normalized and normalized in MODEL_REGISTRY:
        if normalized_allowed and normalized not in normalized_allowed:
            raise ValueError(f"Model '{model_key}' not allowed. Choose from: {', '.join(allowed_keys)}")
        return MODEL_REGISTRY[normalized]
    if normalized:
        raise ValueError(f"Unsupported model key: {model_key}")
    return prompt_for_model_choice(allowed_keys=normalized_allowed)

def prompt_for_model_choice(allowed_keys: Optional[List[str]] = None) -> ModelOption:
    """Prompt user to select a model, optionally filtered by allowed keys.
    
    Args:
        allowed_keys: Optional list of model keys to show. If None, shows all.
    """
    if allowed_keys:
        options = [MODEL_REGISTRY[key] for key in allowed_keys if key in MODEL_REGISTRY]
    else:
        options = list(MODEL_REGISTRY.values())
    
    print("Select AI model:")
    for idx, option in enumerate(options, start=1):
        print(f"  {idx}) {option.label} - {option.description}")
    while True:
        choice = input("Enter choice number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice. Please select a valid option.")

def build_llm_client(option: ModelOption, *, config: Optional[LLMConfig] = None, temperature: Optional[float] = None) -> BaseLLMClient:
    """Build an LLM client with optional configuration.
    
    Args:
        option: Model selection from MODEL_REGISTRY
        config: Optional LLMConfig for customizing behavior
        temperature: Deprecated - use config.temperature instead (kept for backward compatibility)
    
    Returns:
        Configured LLM client ready for generate() calls
    
    Example:
        # Simple usage with defaults
        client = build_llm_client(option)
        
        # High-quality reasoning for complex tasks
        config = LLMConfig(reasoning_effort="high", text_verbosity="medium")
        client = build_llm_client(option, config=config)
        
        # Fast processing with minimal thinking
        config = LLMConfig(thinking_level="minimal", temperature=0.1)
        client = build_llm_client(option, config=config)
    """
    # Backward compatibility: convert temperature to config
    if temperature is not None:
        config = (config or LLMConfig()).merged_over(LLMConfig(temperature=temperature))

    if option.provider == PROVIDER_OPENAI:
        return OpenAIResponsesClient(option, config)
    if option.provider == PROVIDER_GEMINI:
        return GeminiGenerateContentClient(option, config)
    if option.provider == PROVIDER_MISTRAL:
        return MistralClient(option, config)
    if option.provider == PROVIDER_OPENROUTER:
        return OpenRouterClient(option, config)
    raise ValueError(f"Unsupported provider: {option.provider}")

def summary_from_option(option: ModelOption) -> str:
    return f"{option.label} ({option.model})"
