"""
Page-by-page PDF processing with Claude, on the Gemini loop.

OCR and HTR offer Claude Sonnet 5.5 beside Gemini. Everything but the request
itself is the same machine — split the PDF, one page per request, page markers,
failures recorded in the result rather than in the text, ``partial/`` for an
incomplete document, the ``.artifact.json`` sidecar, quota exhaustion stopping
the batch — so :class:`ClaudePageProcessor` subclasses
:class:`~common.gemini_page_processor.GeminiPageProcessor` and replaces only
:meth:`process_page`.

What differs from the Gemini path:

- **The page goes in as a PDF ``document`` block**, the same single-page bytes
  Gemini receives. Claude honours the page's crop box, so a two-up scan split by
  cropping reads as one page. There is no per-part resolution setting and no
  Files API fallback: a page above ``CLAUDE_INLINE_LIMIT_BYTES`` fails.
- **Depth is ``output_config.effort``**; thinking cannot be disabled on Sonnet
  5.5 and no ``thinking`` field is sent. Nor is any sampling parameter.
- **A refusal skips the page.** Anthropic's server-side ``fallbacks`` would
  re-run a declined page on another model inside the same call, and step 03
  would then stamp Sonnet as the author of text it did not write. The page is
  recorded as failed instead, like a Gemini RECITATION with no recovery.
- **Retries are the SDK's.** It retries 408/409/429/5xx and connection errors
  with backoff; what is left after that fails the page. An exhausted credit
  balance is a 400 on Anthropic, not a 429 or 402, and is matched by its
  message so it stops the batch instead of failing every remaining page.

Usage:
    client = build_claude_client(api_key)
    config = claude_request_config(option, system_prompt, effort="minimal")
    processor = ClaudePageProcessor(client, option.model, config, PagePolicy(user_prompt=...))
"""

import base64
import logging
from typing import Any, Dict, Optional

import anthropic

from common.gemini_page_processor import GeminiPageProcessor
from common.gemini_utils import DEFAULT_MULTIMODAL_REQUEST_TIMEOUT_SECONDS, TRUNCATION_MARKER
from common.llm_provider import ANTHROPIC_MAX_TOKENS
from common.llm_registry import ModelOption, resolve_reasoning_effort
from common.rate_limiter import QuotaExhaustedError, is_quota_exhausted

LOGGER = logging.getLogger(__name__)

#: Raw page bytes accepted per request. The API caps a request at 32 MB and
#: base64 adds a third, so this leaves room for the prompt and the encoding.
CLAUDE_INLINE_LIMIT_BYTES = 22 * 1024 * 1024


def build_claude_client(
    api_key: str,
    *,
    timeout_seconds: float = DEFAULT_MULTIMODAL_REQUEST_TIMEOUT_SECONDS,
) -> "anthropic.Anthropic":
    """Create an Anthropic client with a finite HTTP deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)


def claude_request_config(
    option: ModelOption, system_prompt: str, *, effort: Optional[str] = None
) -> Dict[str, Any]:
    """The request settings shared by every page, and recorded in the sidecar.

    *effort* is a depth request like ``thinking_level`` on the Gemini path:
    ``minimal`` becomes the lowest level the model has, ``None`` its default.
    """
    config: Dict[str, Any] = {"max_tokens": ANTHROPIC_MAX_TOKENS, "system": system_prompt}
    resolved = resolve_reasoning_effort(option, effort)
    if resolved:
        config["output_config"] = {"effort": resolved}
    return config


def is_credit_exhausted(error: Exception) -> bool:
    """True when the account can make no more requests until it is topped up."""
    if is_quota_exhausted(error):
        return True
    return (getattr(error, "status_code", None) == 400
            and "credit balance" in str(getattr(error, "message", error)).lower())


class ClaudePageProcessor(GeminiPageProcessor):
    """Runs a PDF through Claude one page at a time."""

    def _create(self, page_bytes: bytes):
        self.rate_limiter.wait()
        document = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(page_bytes).decode("ascii"),
            },
        }
        return self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": [  # document first, then prompt
                document, {"type": "text", "text": self.policy.user_prompt},
            ]}],
            **self.generation_config,
        )

    def _extract_claude(self, response, page_num: int) -> Optional[str]:
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        reason = response.stop_reason

        if reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            self.console.print(
                f"  [yellow]⚠ Page {page_num} skipped[/] - Claude declined the request"
                + (f" ({category})" if category else "")
            )
            self.logger.warning("Page %d: refusal (%s)", page_num, category)
            return None

        if reason == "max_tokens":
            if text:
                self.console.print(
                    f"  [yellow]⚠ Page {page_num} truncated[/] - output exceeded max tokens (partial text saved)"
                )
                self.logger.warning(
                    "Page %d: max_tokens — truncated but %d chars recovered", page_num, len(text)
                )
                return text + TRUNCATION_MARKER
            self.console.print(f"  [red]✗ Page {page_num}[/] - max_tokens with no recoverable text")
            self.logger.error("Page %d: max_tokens but no text could be extracted", page_num)
            return None

        if not text:
            self.console.print(f"  [red]✗ Page {page_num}[/] - empty response (stop reason: {reason})")
            self.logger.error("Page %d: empty Claude response, stop_reason=%s", page_num, reason)
            return None
        return text

    def process_page(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """Send one page. Returns text or None."""
        if len(page_bytes) > CLAUDE_INLINE_LIMIT_BYTES:
            self.console.print(
                f"  [red]✗ Page {page_num}[/] - {len(page_bytes) / 1024 / 1024:.1f} MB exceeds the "
                f"{CLAUDE_INLINE_LIMIT_BYTES // 1024 // 1024} MB page limit for Claude"
            )
            self.logger.error("Page %d: %d bytes, above CLAUDE_INLINE_LIMIT_BYTES", page_num, len(page_bytes))
            return None
        if self.verbose:
            self.console.print(f"  └─ [cyan]📄[/] Processing page {page_num}...")
        try:
            return self._extract_claude(self._create(page_bytes), page_num)
        except QuotaExhaustedError:
            raise
        except anthropic.APIStatusError as exc:
            if is_credit_exhausted(exc):
                raise QuotaExhaustedError(str(exc)) from exc
            self._log_error(exc, "Claude page processing", page_num)
            return None
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            self._log_error(exc, "Claude page processing", page_num)
            return None

    def _log_error(self, error: Exception, context: str, page_num: Optional[int]) -> None:
        if not isinstance(error, anthropic.APIStatusError):
            super()._log_error(error, context, page_num)
            return
        page_info = f" (page {page_num})" if page_num else ""
        self.console.print(
            f"  [red]✗ API error{page_info}[/] - code {error.status_code}: {str(error.message)[:80]}"
        )
        self.logger.error(
            "Anthropic API error during %s%s\n  HTTP code: %s\n  Message: %s\n  Request id: %s",
            context, page_info, error.status_code, error.message, getattr(error, "request_id", None),
        )
