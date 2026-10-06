"""Provider adapters for OpenAI, Gemini, Mistral, OpenRouter, and self-hosted
OpenAI-compatible endpoints.

Model metadata and aliases live in :mod:`common.llm_registry`; this module
re-exports that public catalog for compatibility and owns only SDK-backed calls.
"""
from __future__ import annotations

import json
import os
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from dotenv import load_dotenv

# Type variable for Pydantic models
T = TypeVar('T')

# Optional imports (scripts should still run if a provider is not installed)
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - import guard
    OpenAI = None  # type: ignore

# The SDK's Pydantic-to-json_schema converter, used by ``chat.completions.parse``
# on the way out. The open-model clients below need the same conversion — its
# strict-mode rewriting (``additionalProperties: false``, every field required)
# is what backends validate against — but must NOT hand it the incoming response
# to parse; see ``OpenRouterClient.generate_structured``. A private path, so it
# is imported defensively and falls back to a plain schema.
try:  # pragma: no cover - optional dependency
    from openai.lib._parsing import type_to_response_format_param  # type: ignore
except Exception:  # pragma: no cover - import guard
    type_to_response_format_param = None  # type: ignore

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

# The converter ``chat.parse()`` applies on the way out, so the reasoning path
# below sends the same schema the non-reasoning path does.
try:  # pragma: no cover - optional dependency
    from mistralai.extra import response_format_from_pydantic_model  # type: ignore
except Exception:  # pragma: no cover - import guard
    response_format_from_pydantic_model = None  # type: ignore

# Optional Pydantic import for structured outputs
try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover - import guard
    BaseModel = None  # type: ignore

load_dotenv()

LOGGER = logging.getLogger(__name__)

from common.retry import PermanentError
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
    OPENROUTER_ZDR_ENV,
    PROVIDER_GEMINI,
    PROVIDER_MISTRAL,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_SELFHOSTED,
    SELFHOSTED_QWEN38_MODEL,
    TEXT_ECONOMY_MODELS,
    TEXT_EXTENDED_MODELS,
    TEXT_FULL_MODELS,
    TEXT_OPEN_MODELS,
    THINKING_LEVELS,
    clamp_thinking_level,
    get_model_option,
    model_defaults,
    normalize_model_key,
    prompt_for_model_choice,
    resolve_reasoning_effort,
    summary_from_option,
)

@dataclass
class UsageTotals:
    """Tokens and cost accumulated over a client's lifetime.

    Every adapter records what its provider reports after each call, so a
    pipeline can print what a run cost beside what it produced. ``cost`` is
    only known where the provider states it (OpenRouter); elsewhere it stays
    ``None`` rather than being guessed from a rate card.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None
    # Pipelines share one client across worker threads, and ``+=`` on an
    # attribute is a read-modify-write that loses counts under contention.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def add(
        self,
        *,
        input_tokens: Any = None,
        output_tokens: Any = None,
        cached_input_tokens: Any = None,
        reasoning_tokens: Any = None,
        cost_usd: Any = None,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += _as_int(input_tokens)
            self.output_tokens += _as_int(output_tokens)
            self.cached_input_tokens += _as_int(cached_input_tokens)
            self.reasoning_tokens += _as_int(reasoning_tokens)
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool):
                self.cost_usd = (self.cost_usd or 0.0) + float(cost_usd)

    def summary(self) -> str:
        """One line for a run summary, e.g. ``12 calls · 40,120 in / 3,800 out tokens · $0.0210``."""
        parts = [f"{self.requests:,} calls", f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens"]
        if self.cached_input_tokens:
            parts.append(f"{self.cached_input_tokens:,} cached")
        if self.reasoning_tokens:
            parts.append(f"{self.reasoning_tokens:,} reasoning")
        if self.cost_usd is not None:
            parts.append(f"${self.cost_usd:,.4f}")
        return " · ".join(parts)


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _attr(obj: Any, *names: str) -> Any:
    """Walk ``obj.a.b`` / ``obj["a"]["b"]``, returning None when any step is missing."""
    current = obj
    for name in names:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


class TruncatedOutputError(PermanentError, RuntimeError):
    """The model stopped before finishing its answer, so the text is cut off.

    Raised instead of returning the partial text, which reads as a complete
    answer: OCR correction would save it and step 03 would upload it as the
    item's full text. The usual cause is the output limit, and the same request
    would be cut at the same place, so it is never retried. Send less input per
    request instead.
    """

    def __init__(self, route: str, model: str, reason: str) -> None:
        super().__init__(
            f"{route} ({model}) stopped before finishing its answer ({reason}); "
            "the output is incomplete. Send less text per request."
        )
        self.reason = reason


#: Finish reasons that mean the answer was cut off, across the four wire
#: formats: OpenAI Responses ``incomplete_details.reason``, Gemini
#: ``finish_reason``, and the chat-completions ``finish_reason`` that Mistral,
#: OpenRouter and vLLM report.
_TRUNCATED_FINISH_REASONS = frozenset(
    {"max_output_tokens", "max_tokens", "length", "model_length", "content_filter"}
)


def _finish_reason(value: Any) -> str:
    """A finish reason as a bare lowercase name, whatever the SDK wrapped it in."""
    if value is None:
        return ""
    name = getattr(value, "name", None) or getattr(value, "value", None) or value
    return str(name).rsplit(".", 1)[-1].lower()


def _raise_if_truncated(route: str, model: str, reason: Any) -> None:
    name = _finish_reason(reason)
    if name in _TRUNCATED_FINISH_REASONS:
        raise TruncatedOutputError(route, model, name)


class BaseLLMClient:
    """Minimal interface implemented by provider-specific clients."""

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        self.option = option
        #: What this client has consumed so far — see :class:`UsageTotals`.
        self.usage = UsageTotals()
        defaults = LLMConfig(
            request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        ).merged_over(model_defaults(option))
        self.config = (config or LLMConfig()).merged_over(defaults)

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

    @staticmethod
    def _tier_kwargs(effective_config: LLMConfig) -> Dict[str, Any]:
        """``service_tier`` only when asked for: absent means the project default."""
        tier = effective_config.service_tier
        return {"service_tier": tier} if tier else {}

    def _record_usage(self, response: Any) -> None:
        self.usage.add(
            input_tokens=_attr(response, "usage", "input_tokens"),
            output_tokens=_attr(response, "usage", "output_tokens"),
            cached_input_tokens=_attr(response, "usage", "input_tokens_details", "cached_tokens"),
            reasoning_tokens=_attr(response, "usage", "output_tokens_details", "reasoning_tokens"),
        )

    def _check_complete(self, response: Any) -> None:
        if getattr(response, "status", None) == "incomplete":
            reason = _attr(response, "incomplete_details", "reason") or "incomplete"
            _raise_if_truncated("OpenAI", self.option.model, reason)

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
            text={"verbosity": text_verbosity},
            reasoning={"effort": reasoning_effort},
            store=bool(effective_config.store),
            **self._tier_kwargs(effective_config),
        )
        self._record_usage(response)
        self._check_complete(response)
        # ``output_text`` is the SDK's join of every text part of the response;
        # reasoning items carry none, so nothing else needs walking.
        return (getattr(response, "output_text", None) or "").strip()

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
            **self._tier_kwargs(effective_config),
        )
        self._record_usage(response)
        self._check_complete(response)

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

        All Gemini 3 models (Flash and Pro) use thinking_level; thinking cannot
        be disabled. Which rungs of the ladder exist differs per model, so the
        requested level is snapped to a supported one by
        ``llm_registry.clamp_thinking_level`` rather than sent as-is.
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

        thinking_level = effective_config.thinking_level
        if thinking_level is None:
            # Every Gemini entry in MODEL_REGISTRY declares a default level, so
            # this is a model nobody has configured: send nothing and let it
            # use its own default rather than guess a rung from its name.
            return gen_config_kwargs

        try:
            # Snap to a rung this model actually has. Gemma 4 offers only
            # MINIMAL/HIGH; Gemini 3.7 Flash and every Pro dropped MINIMAL.
            requested = thinking_level
            thinking_level = clamp_thinking_level(self.option.model, thinking_level)
            if str(requested).lower() != thinking_level:
                LOGGER.debug(
                    "%s does not accept thinking_level %s; mapped to %s",
                    self.option.model, requested, thinking_level,
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
        self._record_usage(response)
        self._check_complete(response)
        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""

    def _check_complete(self, response: Any) -> None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            _raise_if_truncated(
                "Gemini", self.option.model, getattr(candidates[0], "finish_reason", None)
            )

    def _record_usage(self, response: Any) -> None:
        self.usage.add(
            input_tokens=_attr(response, "usage_metadata", "prompt_token_count"),
            output_tokens=_attr(response, "usage_metadata", "candidates_token_count"),
            cached_input_tokens=_attr(response, "usage_metadata", "cached_content_token_count"),
            reasoning_tokens=_attr(response, "usage_metadata", "thoughts_token_count"),
        )

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
        self._record_usage(response)
        self._check_complete(response)

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
        effort = resolve_reasoning_effort(self.option, requested)
        if effort is not None and effort != requested:
            LOGGER.debug(
                "%s accepts only %s; requested effort %r sent as %r",
                self.option.model, "/".join(self.option.supported_reasoning_efforts),
                requested, effort,
            )
        return effort

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

        self._record_usage(response)
        self._check_complete(response)
        if response.choices and len(response.choices) > 0:
            # Reasoning mode returns a chunk list, not a string; keep only the
            # answer text and drop the thinking chunk.
            return self._content_text(response.choices[0].message.content).strip()
        return ""

    def _record_usage(self, response: Any) -> None:
        self.usage.add(
            input_tokens=_attr(response, "usage", "prompt_tokens"),
            output_tokens=_attr(response, "usage", "completion_tokens"),
        )

    def _check_complete(self, response: Any) -> None:
        choices = getattr(response, "choices", None) or []
        if choices:
            _raise_if_truncated(
                "Mistral", self.option.model, getattr(choices[0], "finish_reason", None)
            )

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
            self._record_usage(response)
            self._check_complete(response)
            if response.choices:
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    return parsed
            raise ValueError("No output received from Mistral structured response")

        response = self._client.chat.complete(
            model=self.option.model,
            messages=messages,
            response_format=_mistral_response_format(response_schema),
            **common,
        )
        self._record_usage(response)
        self._check_complete(response)
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

    #: Names the route in errors and debug logs. ``SelfHostedClient`` inherits
    #: everything below and would otherwise blame OpenRouter for a failure on a
    #: machine down the hall.
    _route_label = "OpenRouter"

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
        effort = resolve_reasoning_effort(self.option, requested)
        if requested and requested != effort:
            LOGGER.debug(
                "%s does not accept reasoning effort %r (accepts %s); using %r",
                self.option.model, requested,
                ", ".join(self.option.supported_reasoning_efforts) or "none", effort,
            )
        return effort

    def _extra_body(self, effective_config: LLMConfig) -> Dict[str, Any]:
        """Build the OpenRouter-only part of the request body."""
        provider = dict(OPENROUTER_PROVIDER_PREFS)
        if os.getenv(OPENROUTER_ZDR_ENV, "").strip().lower() in ("1", "true", "yes"):
            provider["zdr"] = True  # no storage at all, see llm_registry
        body: Dict[str, Any] = {"provider": provider}
        effort = self._resolve_reasoning_effort(effective_config)
        if effort:
            body["reasoning"] = {"effort": effort}
        return body

    def _record_usage(self, response: Any) -> None:
        self.usage.add(
            input_tokens=_attr(response, "usage", "prompt_tokens"),
            output_tokens=_attr(response, "usage", "completion_tokens"),
            cached_input_tokens=_attr(response, "usage", "prompt_tokens_details", "cached_tokens"),
            reasoning_tokens=_attr(response, "usage", "completion_tokens_details", "reasoning_tokens"),
            cost_usd=_attr(response, "usage", "cost"),
        )

    def _request_headers(self) -> Dict[str, str]:
        """Extra headers for every request. Subclasses serving a different
        endpoint override this — OpenRouter's attribution pair means nothing to
        anyone else, and a strict server may reject what it does not know."""
        return dict(OPENROUTER_HEADERS)

    def structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        effective_config: LLMConfig,
    ) -> Any:
        """Issue a structured request and return the RAW chat completion.

        Separated from :meth:`generate_structured` so a diagnostic can read
        ``usage`` and ``reasoning_content`` off the same request production
        sends, instead of rebuilding it and measuring something else
        (``serving/probe_reasoning.py``).
        """
        temp = effective_config.temperature
        return self._client.chat.completions.create(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_response_format_param(response_schema),
            **({} if temp is None else {"temperature": temp}),
            # Sent explicitly because ``parse()`` sent it: this method replaced
            # that call, and a backend keying on the field's presence should see
            # no difference between the two.
            stream=False,
            extra_body=self._extra_body(effective_config),
            extra_headers=self._request_headers(),
        )

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
        """The first choice's message, once the answer is known to be whole."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError(
                f"No output received from {self._route_label} ({self.option.model})"
            )
        _raise_if_truncated(
            self._route_label, self.option.model, getattr(choices[0], "finish_reason", None)
        )
        return choices[0].message

    def generate(self, system_prompt: str, user_prompt: str, *, config: Optional[LLMConfig] = None) -> str:
        effective_config = self._get_effective_config(config)
        temp = effective_config.temperature

        LOGGER.debug(
            "%s request model=%s temperature=%s reasoning_effort=%s",
            self._route_label, self.option.model, temp,
            effective_config.reasoning_effort,
        )

        response = self._client.chat.completions.create(
            model=self.option.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **({} if temp is None else {"temperature": temp}),
            extra_body=self._extra_body(effective_config),
            extra_headers=self._request_headers(),
        )
        self._record_usage(response)
        return self._message_text(self._first_message(response)).strip()

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        *,
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate structured output via the endpoint's ``json_schema`` support.

        The schema goes out exactly as the SDK's ``parse()`` helper would send
        it — ``to_strict_json_schema()`` conversion included, because a plain
        ``model_json_schema()`` is rejected under ``strict``. What is deliberately
        NOT delegated is parsing the *response*.

        ``parse()`` validates ``message.content`` itself and raises before
        returning, so a model that wraps schema-valid JSON in a ``` fence — which
        open models do routinely, whether reached through a router or served
        locally — produced a ``ValidationError`` no caller could recover from,
        and the tolerant path below was unreachable. (It looked covered: mocking
        ``parse()`` returns fenced content the real SDK would never hand back.)
        Issuing the same request through ``create()`` keeps the recovery in this
        method, where the raw text is still available.
        """
        if BaseModel is None:
            raise RuntimeError("pydantic package is required for structured outputs")

        effective_config = self._get_effective_config(config)

        LOGGER.debug(
            "%s structured request model=%s schema=%s temperature=%s",
            self._route_label, self.option.model, response_schema.__name__,
            effective_config.temperature,
        )

        response = self.structured_response(
            system_prompt, user_prompt, response_schema, effective_config
        )
        self._record_usage(response)

        message = self._first_message(response)

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(
                f"Model refused the structured request via {self._route_label}: {refusal}"
            )

        text = self._message_text(message).strip()
        if not text:
            raise ValueError(
                "No output received in the structured response from "
                f"{self._route_label} ({self.option.model})"
            )
        return response_schema.model_validate_json(_extract_json_payload(text))


class SelfHostedClient(OpenRouterClient):
    """An OpenAI-compatible endpoint you run yourself — typically vLLM.

    Inherits the OpenRouter transport wholesale, because the two face the same
    problem: an open model behind ``response_format`` frequently answers with
    schema-valid JSON as a plain string, sometimes fenced. The recovery path
    (``_extract_json_payload``, the refusal check, the reasoning-trace-aware
    ``_message_text``) is what makes that survivable, and it is worth strictly
    more here — a self-hosted server has no router in front of it filtering for
    backends that honour structured output.

    Three things differ:

    * **The endpoint comes from the environment, not the catalog.** A registry
      entry describes a model; where it happens to be served today is deployment
      state, and on a Slurm cluster it changes with every job. So
      ``SELFHOSTED_LLM_BASE_URL`` is read here, the same way every other client
      reads its key, and a pipeline never passes a URL through.
    * **No routing preferences and no attribution headers.** There is no router
      to instruct: ``data_collection: "deny"`` is a contract with a third party,
      and here there is no third party — the text never leaves the tunnel. A
      strict server may also reject body fields it does not recognise.
    * **Reasoning depth rides in ``chat_template_kwargs``**, which is how vLLM
      passes arguments into a model's chat template. Qwen3.8 reads
      ``reasoning_effort`` there (low/medium/xhigh; xhigh is its own default).
      Sending nothing leaves the server default in place.

    Requires ``SELFHOSTED_LLM_BASE_URL``; ``SELFHOSTED_LLM_API_KEY`` matches the
    server's ``--api-key`` and falls back to vLLM's ``EMPTY`` convention for a
    server started without one. Construction fails loudly when the URL is unset,
    which is what lets the sentiment pilot list this model as *skipped* on a
    laptop with no tunnel open rather than abort the whole run.
    """

    _route_label = "the self-hosted endpoint"

    def __init__(self, option: ModelOption, config: Optional[LLMConfig] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        base_url = os.getenv("SELFHOSTED_LLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "SELFHOSTED_LLM_BASE_URL not set — start the server and open the "
                "tunnel first (see serving/README.md)"
            )
        # Skips OpenRouterClient.__init__ deliberately: everything it does is
        # wanted except its demand for an OPENROUTER_API_KEY, which this route
        # has no use for.
        BaseLLMClient.__init__(self, option, config)
        client_kwargs: Dict[str, Any] = {
            # vLLM requires a key only when started with --api-key, but the
            # OpenAI SDK will not construct without one; "EMPTY" is the
            # convention vLLM's own docs use for the unauthenticated case.
            "api_key": os.getenv("SELFHOSTED_LLM_API_KEY") or "EMPTY",
            "base_url": base_url,
            "timeout": self.config.request_timeout_seconds,
        }
        if self.config.sdk_max_retries is not None:
            client_kwargs["max_retries"] = self.config.sdk_max_retries
        self._client = OpenAI(**client_kwargs)

    def _extra_body(self, effective_config: LLMConfig) -> Dict[str, Any]:
        """Reasoning depth only, and only when the model declares the level."""
        effort = self._resolve_reasoning_effort(effective_config)
        if not effort:
            return {}
        return {"chat_template_kwargs": {"reasoning_effort": effort}}

    def _request_headers(self) -> Dict[str, str]:
        return {}


def _mistral_response_format(response_schema: Type[T]) -> Any:
    """Render a Pydantic model as the ``response_format`` ``chat.parse()`` sends.

    Strict mode wants ``additionalProperties: false`` on every object, which a
    bare ``model_json_schema()`` does not emit; the SDK's own converter does.
    The hand-built fallback exists only for an SDK that has moved the helper.
    """
    if response_format_from_pydantic_model is not None:
        return response_format_from_pydantic_model(response_schema)  # type: ignore[misc]
    return {  # pragma: no cover - only on an SDK without the helper
        "type": "json_schema",
        "json_schema": {
            "name": response_schema.__name__,
            "strict": True,
            "schema": response_schema.model_json_schema(),  # type: ignore[attr-defined]
        },
    }


def _response_format_param(response_schema: Type[T]) -> Dict[str, Any]:
    """Render a Pydantic model as an OpenAI ``response_format`` payload.

    Uses the SDK's own converter so the request is byte-identical to what
    ``chat.completions.parse`` would have sent: strict mode rewrites the schema
    (``additionalProperties: false``, every field listed in ``required``), and
    backends validate against that form. The hand-built fallback exists only for
    an SDK that has moved the private helper; it omits the strict rewriting, so
    a backend enforcing it may reject the request rather than silently answer
    differently.
    """
    if type_to_response_format_param is not None:
        return type_to_response_format_param(response_schema)  # type: ignore[misc]
    return {  # pragma: no cover - only on an SDK without the helper
        "type": "json_schema",
        "json_schema": {
            "name": response_schema.__name__,
            "strict": True,
            "schema": response_schema.model_json_schema(),  # type: ignore[attr-defined]
        },
    }


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_payload(text: str) -> str:
    """Pull the JSON document out of a model response.

    Only needed where open models are served: several of them wrap their answer
    in a Markdown fence or prepend a sentence even when a JSON schema was
    requested. That is true through OpenRouter and equally true of a vLLM
    endpoint you run yourself. Returns ``text`` unchanged when it already
    parses, so a well-behaved response is never rewritten.
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


def build_llm_client(option: ModelOption, *, config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """Build the client for ``option``'s provider.

    ``config`` overrides the model's registry defaults field by field; leave
    ``temperature`` out of it (see CLAUDE.md). For example
    ``build_llm_client(option, config=LLMConfig(thinking_level="minimal"))``.
    """
    if option.provider == PROVIDER_OPENAI:
        return OpenAIResponsesClient(option, config)
    if option.provider == PROVIDER_GEMINI:
        return GeminiGenerateContentClient(option, config)
    if option.provider == PROVIDER_MISTRAL:
        return MistralClient(option, config)
    if option.provider == PROVIDER_OPENROUTER:
        return OpenRouterClient(option, config)
    if option.provider == PROVIDER_SELFHOSTED:
        return SelfHostedClient(option, config)
    raise ValueError(f"Unsupported provider: {option.provider}")
