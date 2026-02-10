"""
Shared Gemini API utilities for multimodal pipelines.

Provides consistent configuration for all scripts that call the Gemini API
directly (OCR extraction, HTR, audio transcription, video processing,
magazine article extraction).

Text-only pipelines should use ``common/llm_provider.py`` instead.

Usage:
    from common.gemini_utils import (
        SAFETY_SETTINGS_NONE,
        build_generation_config,
        extract_text_from_response,
        get_thinking_level,
    )
"""

import logging
from typing import Optional

from google.genai import types

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reusable safety settings — disable all content filters for document
# processing (archival/research use cases).
# ---------------------------------------------------------------------------

SAFETY_SETTINGS_NONE = [
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE",
    ),
]


def get_thinking_level(model_name: str, override: Optional[str] = None) -> str:
    """Return the appropriate thinking level for a Gemini 3 model.

    Args:
        model_name: The full model ID (e.g. ``gemini-3-pro-preview``).
        override: An explicit level to use instead of the default.
            Returned as-is when provided.

    Returns:
        A lowercase thinking-level string accepted by the Gemini 3 API.
    """
    if override:
        return override
    is_pro = "pro" in model_name.lower()
    return "low" if is_pro else "minimal"


def build_generation_config(
    model_name: str,
    *,
    thinking_level: Optional[str] = None,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 65_535,
    response_mime_type: str = "text/plain",
    response_schema=None,
) -> types.GenerateContentConfig:
    """Build a ``GenerateContentConfig`` with consistent defaults.

    Handles thinking-level resolution, safety settings, and optional
    system instructions so that each pipeline does not have to repeat
    this boilerplate.
    """
    level = get_thinking_level(model_name, thinking_level)
    LOGGER.debug("Gemini config: model=%s thinking_level=%s", model_name, level)

    kwargs: dict = {
        "max_output_tokens": max_output_tokens,
        "response_mime_type": response_mime_type,
        "thinking_config": types.ThinkingConfig(thinking_level=level),
        "safety_settings": SAFETY_SETTINGS_NONE,
    }
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
        kwargs["response_schema"] = response_schema
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def extract_text_from_response(response) -> str:
    """Safely extract text from a Gemini response, skipping thinking traces.

    Works correctly with models that return ``thought=True`` parts in
    their responses.  Returns an empty string when the response is empty
    or invalid.
    """
    if not response.candidates:
        return ""
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return ""
    text_parts = []
    for part in candidate.content.parts:
        # Skip thinking trace parts
        if getattr(part, "thought", False):
            continue
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
    return "".join(text_parts).replace("\xa0", " ").strip()
