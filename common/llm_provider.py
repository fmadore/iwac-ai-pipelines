"""Provider adapters for OpenAI, Gemini, Mistral, and OpenRouter.

Model metadata and aliases live in :mod:`common.llm_registry`; this module
re-exports that public catalog for compatibility and owns only SDK-backed calls.
"""
from __future__ import annotations

import json
import os
import logging
import re
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

from common.llm_registry import (  # noqa: F401  (compatibility re-exports)
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TEXT_MODEL_KEY,
    GEMINI_DOCUMENT_MODELS,
    LEGACY_CLI_MODEL_KEYS,
    LLMConfig,
    MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelOption,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_PROVIDER_PREFS,
    PROVIDER_GEMINI,
    PROVIDER_MISTRAL,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    TEXT_ECONOMY_MODELS,
    TEXT_EXTENDED_MODELS,
    TEXT_FULL_MODELS,
    TEXT_OPEN_MODELS,
    get_model_option,
    normalize_model_key,
    prompt_for_model_choice,
    summary_from_option,
)

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
            request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
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
        client_kwargs: Dict[str, Any] = {
            "timeout": self.config.request_timeout_seconds,
        }
        if self.config.sdk_max_retries is not None:
            client_kwargs["max_retries"] = self.config.sdk_max_retries
        self._client = OpenAI(**client_kwargs)

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
        super().__init__(option, config)
        api_key = os.getenv("GEMINI_API_KEY")
        http_options = (
            genai_types.HttpOptions(
                timeout=max(1, int(self.config.request_timeout_seconds * 1000))
            )
            if genai_types is not None and self.config.request_timeout_seconds is not None
            else None
        )
        self._client = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")):
            try:
                self._client = genai.Client(http_options=http_options)
                LOGGER.info("Gemini client initialized via ADC.")
            except Exception as exc:  # pragma: no cover - ADC fallback
                LOGGER.warning("ADC init failed: %s; falling back to API key", exc)
        if self._client is None:
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key, http_options=http_options)

    def _build_generation_config(self, effective_config: LLMConfig) -> Any:
        """Build Gemini generation config with thinking support.
        
        All Gemini 3 models (Flash and Pro) use thinking_level ("MINIMAL", "LOW", or "HIGH").
        Thinking cannot be disabled for Gemini 3 models.
        """
        temp = effective_config.temperature
        # Omit temperature entirely when unset. Google recommends sending no
        # temperature for Gemini 3 (see MODEL_REGISTRY), and there is a real
        # difference between not sending the parameter and sending its nominal
        # default, so the key has to be absent rather than set to 1.0.
        gen_config_kwargs: Dict[str, Any] = {}
        if temp is not None:
            gen_config_kwargs["temperature"] = temp

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

    def _resolve_reasoning_effort(self, effective_config: LLMConfig) -> Optional[str]:
        """Pick a ``reasoning_effort`` to send, or None to send none.

        Mistral Small 4 is a hybrid instruct/reasoning model but accepts only
        ``none`` or ``high`` — ``low`` and ``medium`` are hard 400 errors
        (verified against the live API, 2026-07-29). Since ``LLMConfig`` is
        shared across providers, a panel standardised on "medium" reaches here
        too, and forwarding it would fail the request outright. Round a
        mid-or-higher request up to ``high`` so the model still reasons, and
        report the substitution: this is the one point in the panel where
        effort is genuinely not comparable.
        """
        requested = effective_config.reasoning_effort
        supported = self.option.supported_reasoning_efforts
        if not requested or not supported:
            return None
        if requested in supported:
            return requested
        substitute = "high" if requested in ("medium", "xhigh", "max") else "none"
        if substitute in supported:
            LOGGER.debug(
                "%s accepts only %s; requested effort %r sent as %r",
                self.option.model, "/".join(supported), requested, substitute,
            )
            return substitute
        return None

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if Mistral is None:
            raise RuntimeError("mistralai package is not installed")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        super().__init__(option, config)
        timeout_ms = (
            max(1, int(self.config.request_timeout_seconds * 1000))
            if self.config.request_timeout_seconds is not None else None
        )
        self._client = Mistral(api_key=api_key, timeout_ms=timeout_ms)

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature
        
        LOGGER.debug(f"Mistral request with temperature={temp}")

        effort = self._resolve_reasoning_effort(effective_config)
        response = self._client.chat.complete(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **({} if temp is None else {"temperature": temp}),
            **({} if effort is None else {"reasoning_effort": effort}),
        )

        if response.choices and len(response.choices) > 0:
            # Reasoning mode returns a chunk list, not a string; keep only the
            # answer text and drop the thinking chunk.
            return self._content_text(response.choices[0].message.content).strip()
        return ""

    @staticmethod
    def _content_text(content: Any) -> str:
        """Flatten a Mistral message content into plain text.

        In reasoning mode the API stops returning a string and returns a list
        of chunks instead — ``{"type": "thinking", ...}`` followed by
        ``{"type": "text", ...}``. Only the text chunk is the answer; the
        thinking chunk is the model's scratchpad and must not be parsed as the
        payload. Chunks arrive as SDK objects or plain dicts depending on the
        call path, so both are handled.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    chunk_type, chunk_text = chunk.get("type"), chunk.get("text")
                else:
                    chunk_type, chunk_text = getattr(chunk, "type", None), getattr(chunk, "text", None)
                if chunk_type == "text" and chunk_text:
                    parts.append(chunk_text)
            return "".join(parts)
        return str(content)

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output using Mistral's native JSON schema support.

        Two paths, because ``chat.parse()`` cannot read a reasoning response:
        with reasoning enabled the SDK raises ``TypeError: Unexpected type for
        message.content: <class 'list'>`` on the thinking/text chunk list. So
        reasoning requests go through ``chat.complete()`` and are validated
        here against the same schema.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")

        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature
        effort = self._resolve_reasoning_effort(effective_config)
        reasoning_on = effort is not None and effort != "none"

        LOGGER.debug(
            f"Mistral structured request with schema={response_schema.__name__}, "
            f"temperature={temp}, reasoning_effort={effort}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        common = {
            **({} if temp is None else {"temperature": temp}),
            **({} if effort is None else {"reasoning_effort": effort}),
        }

        if not reasoning_on:
            response = self._client.chat.parse(
                model=self.option.model,
                messages=messages,
                response_format=response_schema,
                **common,
            )
            if response.choices:
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    return parsed
            raise ValueError("No output received from Mistral structured response")

        response = self._client.chat.complete(
            model=self.option.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            },
            **common,
        )
        if not response.choices:
            raise ValueError("No output received from Mistral structured response")
        text = self._content_text(response.choices[0].message.content).strip()
        if not text:
            raise ValueError("Mistral returned reasoning but no answer text")
        return response_schema.model_validate_json(text)


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
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": OPENROUTER_BASE_URL,
            "timeout": self.config.request_timeout_seconds,
        }
        if self.config.sdk_max_retries is not None:
            client_kwargs["max_retries"] = self.config.sdk_max_retries
        self._client = OpenAI(**client_kwargs)

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
            **({} if temp is None else {"temperature": temp}),
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
            **(
                {}
                if effective_config.temperature is None
                else {"temperature": effective_config.temperature}
            ),
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


def build_llm_client(option: ModelOption, *, config: Optional[LLMConfig] = None, temperature: Optional[float] = None) -> BaseLLMClient:
    """Build an LLM client with optional configuration.
    
    Args:
        option: Model selection from MODEL_REGISTRY
        config: Optional LLMConfig for customizing behavior
        temperature: Deprecated - overrides the model's vendor-recommended default,
                     which is rarely what you want (see LLMConfig). Kept only for
                     backward compatibility.
    
    Returns:
        Configured LLM client ready for generate() calls
    
    Example:
        # Simple usage with defaults
        client = build_llm_client(option)
        
        # High-quality reasoning for complex tasks
        config = LLMConfig(reasoning_effort="high", text_verbosity="medium")
        client = build_llm_client(option, config=config)
        
        # Fast processing with minimal thinking
        config = LLMConfig(thinking_level="minimal")
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
