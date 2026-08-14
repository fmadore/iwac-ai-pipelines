"""
Shared Gemini API utilities for multimodal pipelines.

Provides consistent configuration for all scripts that call the Gemini API
directly (OCR extraction, HTR, audio transcription, video processing,
magazine article extraction).

Text-only pipelines should use ``common/llm_provider.py`` instead.

Usage:
    from common.gemini_utils import (
        INLINE_REQUEST_LIMIT_BYTES,
        SAFETY_SETTINGS_NONE,
        build_generation_config,
        delete_uploaded_file,
        extract_text_from_response,
        get_thinking_level,
        upload_and_wait_active,
    )
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional, Union

from google import genai
from google.genai import types

from common.llm_registry import clamp_thinking_level

LOGGER = logging.getLogger(__name__)

# The Gemini API caps a request at 20 MB; prompt bytes count toward it, so
# inline media payloads should stay below this margin before falling back to
# the Files API.
INLINE_REQUEST_LIMIT_BYTES = 18 * 1024 * 1024
DEFAULT_MULTIMODAL_REQUEST_TIMEOUT_SECONDS = 600.0


def build_gemini_client(
    api_key: str,
    *,
    timeout_seconds: float = DEFAULT_MULTIMODAL_REQUEST_TIMEOUT_SECONDS,
) -> genai.Client:
    """Create a direct multimodal client with a finite HTTP deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=max(1, int(timeout_seconds * 1000))),
    )

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

    The caller states the depth it wants; this snaps it to a rung the model
    actually has. That clamp is not cosmetic — Gemini 3.7 Flash dropped MINIMAL
    and ``gemini-flash-latest`` rolled onto it, so the old unconditional
    ``"minimal"`` became a 400 on every Flash call this module serves.

    Args:
        model_name: The full model ID (e.g. ``gemini-pro-latest``).
        override: An explicit level to use instead of the default. Still
            clamped — an override names a depth, not a promise the model has it.

    Returns:
        A lowercase thinking-level string accepted by *model_name*.
    """
    requested = override or ("low" if "pro" in model_name.lower() else "minimal")
    return clamp_thinking_level(model_name, requested)


def build_generation_config(
    model_name: str,
    *,
    thinking_level: Optional[str] = None,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 65_535,
    response_mime_type: str = "text/plain",
    response_schema=None,
    temperature: Optional[float] = None,
    media_resolution: Optional[str] = None,
) -> types.GenerateContentConfig:
    """Build a ``GenerateContentConfig`` with consistent defaults.

    Handles thinking-level resolution, safety settings, and optional
    system instructions so that each pipeline does not have to repeat
    this boilerplate.

    Args:
        temperature: Optional sampling temperature (0.0 is honored). Leave unset:
            Google recommends sending no temperature for Gemini 3, because a
            value below the 1.0 default "may lead to unexpected behavior, such
            as looping or degraded performance" — for these pipelines that means
            a transcript repeating a paragraph or OCR stalling on one line.
            Constrain output through the system instruction instead.
        media_resolution: Optional media resolution name, e.g. ``"HIGH"``
            for handwriting/archival scans where fine detail matters.
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
    if temperature is not None:
        kwargs["temperature"] = temperature
    if media_resolution is not None:
        kwargs["media_resolution"] = getattr(
            types.MediaResolution, f"MEDIA_RESOLUTION_{media_resolution.upper()}"
        )
    return types.GenerateContentConfig(**kwargs)


def upload_and_wait_active(
    client,
    source: Union[str, Path, bytes, io.IOBase],
    *,
    mime_type: Optional[str] = None,
    max_wait: int = 600,
    poll_interval: float = 5.0,
):
    """Upload a file to the Gemini Files API and wait until it is ACTIVE.

    Centralizes the upload/poll loop that audio, video, HTR, and OCR
    pipelines each used to implement (with different bugs: no timeout,
    no FAILED handling, leaked uploads).

    Args:
        client: A ``genai.Client`` instance.
        source: A file path, raw bytes, or an open binary stream.
        mime_type: Required when *source* is bytes or a stream.
        max_wait: Maximum seconds to wait for processing.
        poll_interval: Seconds between state polls.

    Returns:
        The ACTIVE file object.

    Raises:
        RuntimeError: If the upload fails server-side (FAILED state) or
            enters an unexpected state. The upload is deleted first.
        TimeoutError: If the file is not ACTIVE within *max_wait* seconds.
            The upload is deleted first.
    """
    if max_wait < 0:
        raise ValueError("max_wait must not be negative")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    uploaded = _upload_file(client, source, mime_type)

    if not uploaded or not uploaded.name:
        raise RuntimeError("Gemini Files API upload returned no file handle")

    waited = 0.0
    current = uploaded
    while True:
        state = current.state.name if current.state else None
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            delete_uploaded_file(client, current)
            raise RuntimeError("Gemini file processing failed (state=FAILED)")
        if state not in (None, "PROCESSING"):
            delete_uploaded_file(client, current)
            raise RuntimeError(f"Unexpected Gemini file state: {state}")
        if waited >= max_wait:
            delete_uploaded_file(client, current)
            raise TimeoutError(f"Gemini file not ACTIVE after {max_wait}s (state={state})")
        time.sleep(poll_interval)
        waited += poll_interval
        current = client.files.get(name=uploaded.name)


def _upload_file(
    client,
    source: Union[str, Path, bytes, io.IOBase],
    mime_type: Optional[str],
):
    """Normalize supported sources into one Gemini Files API upload call."""
    if isinstance(source, bytes):
        if not mime_type:
            raise ValueError("mime_type is required when uploading bytes")
        source = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        return client.files.upload(
            file=str(source),
            config={"mime_type": mime_type} if mime_type else None,
        )
    elif not mime_type:
        raise ValueError("mime_type is required when uploading a stream")
    return client.files.upload(file=source, config={"mime_type": mime_type})


def delete_uploaded_file(client, uploaded_file) -> None:
    """Best-effort deletion of a Files API upload.

    Never raises: cleanup failure is non-critical (uploads expire after
    48h), but leaked multi-GB uploads waste quota in the meantime.
    """
    name = getattr(uploaded_file, "name", None)
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception as exc:
        LOGGER.debug("Could not delete uploaded file %s: %s", name, exc)


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
