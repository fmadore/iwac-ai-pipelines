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
        description="Google Gemini 2.5 Pro — highest quality"
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

    def __init__(self, option: ModelOption, temperature: Optional[float] = None) -> None:
        self.option = option
        self.temperature = temperature or option.default_temperature

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: Optional[float] = None) -> str:
        raise NotImplementedError

class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, option: ModelOption, temperature: Optional[float] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        super().__init__(option, temperature)
        self._client = OpenAI()

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: Optional[float] = None) -> str:
        del temperature  # OpenAI path uses fixed settings block per spec
        response = self._client.responses.create(
            model=self.option.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {"type": "text"},
                "verbosity": "low",
            },
            reasoning={"effort": "low"},
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
    def __init__(self, option: ModelOption, temperature: Optional[float] = None) -> None:
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
        super().__init__(option, temperature)

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        gen_config = None
        if genai_types is not None:
            try:
                gen_config = genai_types.GenerateContentConfig(temperature=temp)
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

def get_model_option(model_key: Optional[str]) -> ModelOption:
    normalized = normalize_model_key(model_key)
    if normalized and normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]
    if normalized:
        raise ValueError(f"Unsupported model key: {model_key}")
    return prompt_for_model_choice()

def prompt_for_model_choice() -> ModelOption:
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

def build_llm_client(option: ModelOption, *, temperature: Optional[float] = None) -> BaseLLMClient:
    if option.provider == PROVIDER_OPENAI:
        return OpenAIResponsesClient(option, temperature)
    if option.provider == PROVIDER_GEMINI:
        return GeminiGenerateContentClient(option, temperature)
    raise ValueError(f"Unsupported provider: {option.provider}")

def summary_from_option(option: ModelOption) -> str:
    return f"{option.label} ({option.model})"
