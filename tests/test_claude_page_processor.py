"""Tests for the Claude path of the page-by-page PDF loop.

The loop itself is the Gemini one and is covered in
test_gemini_page_processor.py; these pin what the Claude request adds.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx2
import pytest
from rich.console import Console

from common.claude_page_processor import (
    CLAUDE_INLINE_LIMIT_BYTES,
    ClaudePageProcessor,
    build_claude_client,
    claude_request_config,
)
from common.gemini_page_processor import TRUNCATION_MARKER, PagePolicy
from common.iwac_config import AI_MODEL_ITEMS
from common.llm_registry import DOCUMENT_MODELS, MODEL_REGISTRY
from common.rate_limiter import QuotaExhaustedError

SONNET = MODEL_REGISTRY["claude-sonnet-5-5"]


def make_response(text="", stop_reason="end_turn", category=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
    )


def make_processor(response=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.messages.create.side_effect = side_effect
    else:
        client.messages.create.return_value = response
    processor = ClaudePageProcessor(
        client,
        SONNET.model,
        claude_request_config(SONNET, "System prompt.", effort="minimal"),
        PagePolicy(user_prompt="Transcribe."),
        console=Console(file=io.StringIO(), force_terminal=False),
        model_key=SONNET.key,
    )
    return processor, client


def api_error(cls, status, message):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx2.Response(status, request=request), body=None)


def test_minimal_effort_is_the_lowest_level_and_no_sampling_is_sent():
    config = claude_request_config(SONNET, "System prompt.", effort="minimal")
    assert config["output_config"] == {"effort": "low"}
    assert config["system"] == "System prompt."
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(config)


def test_page_is_sent_as_a_pdf_document_before_the_prompt():
    processor, client = make_processor(make_response("نص الصفحة"))
    assert processor.process_page(b"%PDF-page", 1) == "نص الصفحة"

    kwargs = client.messages.create.call_args.kwargs
    document, prompt = kwargs["messages"][0]["content"]
    assert document["type"] == "document"
    assert document["source"]["media_type"] == "application/pdf"
    assert prompt == {"type": "text", "text": "Transcribe."}
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["output_config"] == {"effort": "low"}


def test_max_tokens_salvages_partial_text():
    processor, _ = make_processor(make_response("partial", "max_tokens"))
    assert processor.process_page(b"%PDF", 1) == "partial" + TRUNCATION_MARKER


def test_refusal_skips_the_page():
    """No fallback model: step 03 would stamp Sonnet on another model's text."""
    processor, _ = make_processor(make_response("", "refusal", category="cyber"))
    assert processor.process_page(b"%PDF", 1) is None


def test_empty_answer_fails_the_page():
    processor, _ = make_processor(make_response(""))
    assert processor.process_page(b"%PDF", 1) is None


def test_exhausted_credit_stops_the_batch():
    """Anthropic reports an empty balance as a 400, not a 402 or 429."""
    processor, _ = make_processor(side_effect=api_error(
        anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API."
    ))
    with pytest.raises(QuotaExhaustedError):
        processor.process_page(b"%PDF", 1)


def test_other_api_errors_fail_only_the_page():
    processor, _ = make_processor(side_effect=api_error(
        anthropic.InternalServerError, 500, "Overloaded"
    ))
    assert processor.process_page(b"%PDF", 1) is None


def test_oversized_page_is_not_sent():
    processor, client = make_processor(make_response("text"))
    assert processor.process_page(b"x" * (CLAUDE_INLINE_LIMIT_BYTES + 1), 1) is None
    client.messages.create.assert_not_called()


def test_sidecar_records_the_request_settings(tmp_path):
    processor, _ = make_processor(make_response("page"))
    source = MagicMock()
    source.__len__.return_value = 1
    source.page_bytes.return_value = b"%PDF"
    (tmp_path / "doc.pdf").write_bytes(b"synthetic source")

    with patch("common.gemini_page_processor.PdfPageSource", return_value=source):
        result = processor.process_pdf(tmp_path / "doc.pdf", tmp_path / "doc.txt")

    assert result.ok
    sidecar = json.loads((tmp_path / "doc.txt.artifact.json").read_text(encoding="utf-8"))
    assert '"effort": "low"' in json.dumps(sidecar)
    assert "claude-sonnet-5-5" in json.dumps(sidecar)


def test_client_has_finite_timeout():
    with pytest.raises(ValueError):
        build_claude_client("key", timeout_seconds=0)
    assert build_claude_client("key").timeout == 600.0


def test_sonnet_is_offered_for_documents_and_can_be_stamped():
    assert "claude-sonnet-5-5" in DOCUMENT_MODELS
    assert AI_MODEL_ITEMS["claude-sonnet-5-5"]["item_id"] == 188959
