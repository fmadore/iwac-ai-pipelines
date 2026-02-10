"""Shared helpers for selecting and calling Large Language Models (OpenAI / Gemini / Mistral).

This module centralizes provider/model selection so individual pipelines only need to
focus on their prompts. Adding new models or tweaking API settings now only requires
changing this file.
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

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
    from mistralai import Mistral  # type: ignore
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

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_FULL_MODEL = "gpt-5.1"
DEFAULT_GEMINI_FLASH = "gemini-3-flash-preview"
DEFAULT_GEMINI_PRO = "gemini-3-pro-preview"
DEFAULT_MISTRAL_LARGE = "mistral-large-2512"
DEFAULT_MINISTRAL_14B = "ministral-14b-2512"

@dataclass(frozen=True)
class ModelOption:
    key: str
    provider: str
    model: str
    label: str
    description: str
    default_temperature: float = 0.2
    # OpenAI-specific defaults
    default_reasoning_effort: str = "low"  # "low", "medium", "high"
    default_text_verbosity: str = "low"    # "low", "medium", "high"
    # Gemini-specific defaults
    default_thinking_level: Optional[str] = None  # For Gemini 3: Flash="minimal"/"low"/"medium"/"high", Pro="low"/"high"

@dataclass
class LLMConfig:
    """Configuration for LLM generation requests.
    
    Scripts can create instances of this class to customize behavior per use case.
    
    OpenAI parameters:
        reasoning_effort: "low", "medium", or "high" - controls reasoning depth
        text_verbosity: "low", "medium", or "high" - controls response length
    
    Gemini parameters:
        temperature: 0.0-1.0 - controls randomness (OpenAI ignores this)
        
        For Gemini 3 (Flash and Pro):
            thinking_level: Controls reasoning depth (cannot be disabled)
                           Flash: "MINIMAL", "LOW", "MEDIUM", or "HIGH"
                           Pro: "LOW" or "HIGH" only
    
    Example:
        # High-quality reasoning for complex NER
        config = LLMConfig(reasoning_effort="high", text_verbosity="medium")

        # Fast OCR with Gemini 3 Pro (low thinking)
        config = LLMConfig(thinking_level="low", temperature=0.1)

        # Fast OCR correction with Gemini 3 Flash (minimal thinking)
        config = LLMConfig(thinking_level="minimal", temperature=0.1)
    """
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None
    thinking_level: Optional[str] = None  # Gemini 3: Flash="minimal"/"low"/"medium"/"high", Pro="low"/"high"

MODEL_REGISTRY: Dict[str, ModelOption] = {
    "gpt-5-mini": ModelOption(
        key="gpt-5-mini",
        provider=PROVIDER_OPENAI,
        model=DEFAULT_OPENAI_MODEL,
        label="ChatGPT (GPT-5 mini)",
        description="OpenAI Responses API — cost-optimized gpt-5-mini"
    ),
    "gpt-5.1": ModelOption(
        key="gpt-5.1",
        provider=PROVIDER_OPENAI,
        model=OPENAI_FULL_MODEL,
        label="ChatGPT (GPT-5.1 full)",
        description="OpenAI Responses API — flagship gpt-5.1"
    ),
    "gemini-flash": ModelOption(
        key="gemini-flash",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_FLASH,
        label="Gemini 3 Flash",
        description="Google Gemini 3 Flash — fast, cost-effective",
        default_thinking_level="MINIMAL"  # Gemini 3 Flash uses thinking_level, cannot be disabled
    ),
    "gemini-pro": ModelOption(
        key="gemini-pro",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_PRO,
        label="Gemini 3.0 Pro",
        description="Google Gemini 3.0 Pro — highest quality",
        default_thinking_level="low"  # Gemini 3 Pro uses thinking_level, not thinking_budget
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
}

MODEL_ALIASES = {
    "gemini": "gemini-flash",
    "openai": "gpt-5-mini",  # Legacy generic name
    "openai:gpt-5-mini": "gpt-5-mini",
    "openai:gpt-5.1": "gpt-5.1",
    # Legacy aliases for decommissioned models/flags
    "openai-mini": "gpt-5-mini",
    "gpt-5.1-mini": "gpt-5-mini",  # Legacy incorrect naming
    "openai:gpt-5.1-mini": "gpt-5-mini",
    "gpt-5": "gpt-5.1",  # gpt-5 is previous flagship, maps to 5.1
    "openai:gpt-5": "gpt-5.1",
    "openai-5": "gpt-5.1",  # Legacy key name
    "openai-5.1": "gpt-5.1",  # Legacy key name
    "gemini-3-flash-preview": "gemini-flash",
    "gemini-3-pro-preview": "gemini-pro",
    # Mistral aliases
    "mistral": "mistral-large",
    "mistral-large-latest": "mistral-large",
    "mistral-large-2512": "mistral-large",
    # Ministral aliases
    "ministral": "ministral-14b",
    "ministral-3": "ministral-14b",
    "ministral-14b-2512": "ministral-14b",
}

class BaseLLMClient:
    """Minimal interface implemented by provider-specific clients."""

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        self.option = option
        self.config = config or LLMConfig()
        # Fill in defaults from ModelOption when not explicitly set
        if self.config.temperature is None:
            self.config = LLMConfig(
                temperature=option.default_temperature,
                reasoning_effort=self.config.reasoning_effort,
                text_verbosity=self.config.text_verbosity,
                thinking_level=self.config.thinking_level,
            )

    def _get_effective_config(self, config: Optional[LLMConfig]) -> LLMConfig:
        """Merge a per-request config with client defaults."""
        if not config:
            return self.config
        return LLMConfig(
            temperature=config.temperature if config.temperature is not None else self.config.temperature,
            reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else self.config.reasoning_effort,
            text_verbosity=config.text_verbosity if config.text_verbosity is not None else self.config.text_verbosity,
            thinking_level=config.thinking_level if config.thinking_level is not None else self.config.thinking_level,
        )

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
        reasoning_effort = effective_config.reasoning_effort or self.option.default_reasoning_effort
        text_verbosity = effective_config.text_verbosity or self.option.default_text_verbosity
        
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
            store=True,
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
        
        Uses the Responses API with structured output format to guarantee valid JSON
        matching the provided Pydantic schema.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")
        
        effective_config = self._get_effective_config(config)
        reasoning_effort = effective_config.reasoning_effort or self.option.default_reasoning_effort
        text_verbosity = effective_config.text_verbosity or self.option.default_text_verbosity
        
        # Get JSON schema from Pydantic model
        json_schema = response_schema.model_json_schema()
        
        LOGGER.debug(
            f"OpenAI structured request with schema={response_schema.__name__}, "
            f"reasoning_effort={reasoning_effort}"
        )
        
        response = self._client.responses.create(
            model=self.option.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "schema": json_schema,
                    "strict": True
                },
                "verbosity": text_verbosity,
            },
            reasoning={"effort": reasoning_effort},
            tools=[],
            store=True,
        )
        
        raw_output = getattr(response, "output_text", None)
        if not raw_output:
            # Try to extract from output segments
            segments: List[str] = []
            for seg in getattr(response, "output", []) or []:
                if isinstance(seg, dict):
                    content = seg.get("content")
                    if isinstance(content, str):
                        segments.append(content)
            raw_output = "\n".join(filter(None, segments)).strip()
        
        if not raw_output:
            raise ValueError("No output received from OpenAI structured response")
        
        # Parse and validate with Pydantic
        return response_schema.model_validate_json(raw_output)

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
        temp = effective_config.temperature or self.option.default_temperature
        gen_config_kwargs: Dict[str, Any] = {"temperature": temp}
        
        if genai_types is None:
            return gen_config_kwargs
        
        try:
            # All Gemini 3 models use thinking_level (cannot be disabled)
            # Default to "MINIMAL" for Flash, "LOW" for Pro if not specified
            thinking_level = effective_config.thinking_level or self.option.default_thinking_level
            if thinking_level is None:
                # Fallback based on model type
                is_pro_model = "pro" in self.option.model.lower()
                thinking_level = "LOW" if is_pro_model else "MINIMAL"
            
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
        
        gen_config = None
        if genai_types is not None:
            try:
                gen_config = genai_types.GenerateContentConfig(**gen_config_kwargs)
            except Exception:  # pragma: no cover - optional field
                gen_config = None
        
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
            raise RuntimeError(f"Failed to configure Gemini structured output: {exc}")
        
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
        temp = effective_config.temperature or self.option.default_temperature
        
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
        temp = effective_config.temperature or self.option.default_temperature
        
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
    if normalized and normalized in MODEL_REGISTRY:
        if allowed_keys and normalized not in allowed_keys:
            raise ValueError(f"Model '{model_key}' not allowed. Choose from: {', '.join(allowed_keys)}")
        return MODEL_REGISTRY[normalized]
    if normalized:
        raise ValueError(f"Unsupported model key: {model_key}")
    return prompt_for_model_choice(allowed_keys=allowed_keys)

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
    if temperature is not None and config is None:
        config = LLMConfig(temperature=temperature)
    elif temperature is not None and config is not None and config.temperature is None:
        config = LLMConfig(
            temperature=temperature,
            reasoning_effort=config.reasoning_effort,
            text_verbosity=config.text_verbosity,
            thinking_level=config.thinking_level,
        )
    
    if option.provider == PROVIDER_OPENAI:
        return OpenAIResponsesClient(option, config)
    if option.provider == PROVIDER_GEMINI:
        return GeminiGenerateContentClient(option, config)
    if option.provider == PROVIDER_MISTRAL:
        return MistralClient(option, config)
    raise ValueError(f"Unsupported provider: {option.provider}")

def summary_from_option(option: ModelOption) -> str:
    return f"{option.label} ({option.model})"
