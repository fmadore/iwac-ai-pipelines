"""Shared helpers for selecting and calling Large Language Models (OpenAI / Gemini).

This module centralizes provider/model selection so individual pipelines only need to
focus on their prompts. Adding new models or tweaking API settings now only requires
changing this file.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

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

load_dotenv()

LOGGER = logging.getLogger(__name__)

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

DEFAULT_OPENAI_MODEL = "gpt-5.1-mini"
OPENAI_FULL_MODEL = "gpt-5.1"
DEFAULT_GEMINI_FLASH = "gemini-2.5-flash"
DEFAULT_GEMINI_PRO = "gemini-2.5-pro"

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
    default_thinking_mode: bool = False    # Enable thinking for flash (Pro always has thinking)
    default_thinking_budget: Optional[int] = None  # None = model default, 0 = disabled (flash only), >0 = custom

@dataclass
class LLMConfig:
    """Configuration for LLM generation requests.
    
    Scripts can create instances of this class to customize behavior per use case.
    
    OpenAI parameters:
        reasoning_effort: "low", "medium", or "high" - controls reasoning depth
        text_verbosity: "low", "medium", or "high" - controls response length
    
    Gemini parameters:
        temperature: 0.0-1.0 - controls randomness (OpenAI ignores this)
        thinking_mode: bool - enable extended thinking (Pro: always on, Flash: optional)
        thinking_budget: int - 0 to disable (Flash only), None for default, >0 for custom budget
                              Note: Pro model cannot disable thinking (has minimum budget)
    
    Example:
        # High-quality reasoning for complex NER
        config = LLMConfig(reasoning_effort="high", text_verbosity="medium")
        
        # Fast OCR correction with minimal thinking
        config = LLMConfig(thinking_budget=0, temperature=0.1)
    """
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None
    thinking_mode: Optional[bool] = None
    thinking_budget: Optional[int] = None

MODEL_REGISTRY: Dict[str, ModelOption] = {
    "openai": ModelOption(
        key="openai",
        provider=PROVIDER_OPENAI,
        model=DEFAULT_OPENAI_MODEL,
        label="ChatGPT (GPT-5.1 mini)",
        description="OpenAI Responses API — default gpt-5.1-mini"
    ),
    "openai-5.1": ModelOption(
        key="openai-5.1",
        provider=PROVIDER_OPENAI,
        model=OPENAI_FULL_MODEL,
        label="ChatGPT (GPT-5.1 full)",
        description="OpenAI Responses API — flagship gpt-5.1"
    ),
    "gemini-flash": ModelOption(
        key="gemini-flash",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_FLASH,
        label="Gemini 2.5 Flash",
        description="Google Gemini 2.5 Flash — fast, cost-effective"
    ),
    "gemini-pro": ModelOption(
        key="gemini-pro",
        provider=PROVIDER_GEMINI,
        model=DEFAULT_GEMINI_PRO,
        label="Gemini 2.5 Pro",
        description="Google Gemini 2.5 Pro — highest quality",
        default_thinking_mode=True  # Pro requires thinking, cannot be disabled
    ),
}

MODEL_ALIASES = {
    "gemini": "gemini-flash",
    "gpt-5.1-mini": "openai",
    "openai:gpt-5.1-mini": "openai",
    "gpt-5.1": "openai-5.1",
    "openai:gpt-5.1": "openai-5.1",
    # Legacy aliases for decommissioned models/flags
    "openai-mini": "openai",
    "gpt-5-mini": "openai",
    "openai:gpt-5-mini": "openai",
    "gemini-2.5-flash": "gemini-flash",
    "gemini-2.5-pro": "gemini-pro",
}

class BaseLLMClient:
    """Minimal interface implemented by provider-specific clients."""

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        self.option = option
        self.config = config or LLMConfig()
        # Backward compatibility: support direct temperature parameter
        if self.config.temperature is None:
            self.config = LLMConfig(
                temperature=option.default_temperature,
                reasoning_effort=self.config.reasoning_effort,
                text_verbosity=self.config.text_verbosity,
                thinking_mode=self.config.thinking_mode,
                thinking_budget=self.config.thinking_budget
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

class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        super().__init__(option, config)
        self._client = OpenAI()

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        # Merge configs: request-specific overrides client defaults
        effective_config = self.config
        if config:
            effective_config = LLMConfig(
                temperature=config.temperature if config.temperature is not None else self.config.temperature,
                reasoning_effort=config.reasoning_effort or self.config.reasoning_effort,
                text_verbosity=config.text_verbosity or self.config.text_verbosity,
                thinking_mode=config.thinking_mode if config.thinking_mode is not None else self.config.thinking_mode,
                thinking_budget=config.thinking_budget if config.thinking_budget is not None else self.config.thinking_budget
            )
        
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

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        # Merge configs: request-specific overrides client defaults
        effective_config = self.config
        if config:
            effective_config = LLMConfig(
                temperature=config.temperature if config.temperature is not None else self.config.temperature,
                reasoning_effort=config.reasoning_effort or self.config.reasoning_effort,
                text_verbosity=config.text_verbosity or self.config.text_verbosity,
                thinking_mode=config.thinking_mode if config.thinking_mode is not None else self.config.thinking_mode,
                thinking_budget=config.thinking_budget if config.thinking_budget is not None else self.config.thinking_budget
            )
        
        temp = effective_config.temperature or self.option.default_temperature
        thinking_mode = effective_config.thinking_mode
        if thinking_mode is None:
            thinking_mode = self.option.default_thinking_mode
        thinking_budget = effective_config.thinking_budget
        if thinking_budget is None:
            thinking_budget = self.option.default_thinking_budget
        
        # Gemini Pro cannot disable thinking (has minimum budget requirement)
        is_pro_model = "pro" in self.option.model.lower()
        if is_pro_model and thinking_budget == 0:
            LOGGER.warning(
                f"Cannot disable thinking for {self.option.model} - Pro requires minimum thinking budget. "
                f"Ignoring thinking_budget=0."
            )
            thinking_budget = None  # Let model use its default minimum
        
        gen_config_kwargs = {"temperature": temp}
        
        # Add thinking config if enabled or budget specified
        if genai_types is not None and (thinking_mode or thinking_budget is not None):
            try:
                # Only add thinking_config if we have a budget to set
                if thinking_budget is not None:
                    thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
                    gen_config_kwargs["thinking_config"] = thinking_config
                    LOGGER.debug(f"Gemini request with thinking_budget={thinking_budget}, temperature={temp}")
                else:
                    LOGGER.debug(f"Gemini request with temperature={temp}")
            except Exception as exc:  # pragma: no cover - optional field
                LOGGER.warning("Failed to configure thinking mode: %s", exc)
        
        gen_config = None
        if genai_types is not None:
            try:
                gen_config = genai_types.GenerateContentConfig(**gen_config_kwargs)
            except Exception:  # pragma: no cover - optional field
                gen_config = None
        
        prompt = f"{system_prompt}\n\n{user_prompt}\nReturn ONLY the answer requested.".strip()
        response = self._client.models.generate_content(
            model=self.option.model,
            contents=prompt,
            config=gen_config,
        )
        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""

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
        config = LLMConfig(thinking_budget=0, temperature=0.1)
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
            thinking_mode=config.thinking_mode,
            thinking_budget=config.thinking_budget
        )
    
    if option.provider == PROVIDER_OPENAI:
        return OpenAIResponsesClient(option, config)
    if option.provider == PROVIDER_GEMINI:
        return GeminiGenerateContentClient(option, config)
    raise ValueError(f"Unsupported provider: {option.provider}")

def summary_from_option(option: ModelOption) -> str:
    return f"{option.label} ({option.model})"
