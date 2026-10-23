"""Tests for provider-independent magazine consolidation and batch setup."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

from rich.panel import Panel

import magazine_extraction
from common.llm_provider import get_model_option


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "AI_summary_issue"
    / "02_Mistral_generate_summaries_issue.py"
)
SPEC = importlib.util.spec_from_file_location("mistral_magazine_pipeline", SCRIPT)
assert SPEC and SPEC.loader
mistral_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mistral_pipeline
SPEC.loader.exec_module(mistral_pipeline)


def test_text_consolidator_uses_shared_structured_client(monkeypatch):
    expected = magazine_extraction.MagazineIndex(articles=[])
    client = Mock()
    client.generate_structured.return_value = expected
    build_client = Mock(return_value=client)
    monkeypatch.setattr(magazine_extraction, "build_llm_client", build_client)

    option = get_model_option("deepseek-v4-flash-0731")
    consolidate = magazine_extraction.build_text_consolidator(option)
    result = consolidate("system", '[{"page": 1}]')

    assert result is expected
    build_client.assert_called_once()
    assert build_client.call_args.args == (option,)
    assert build_client.call_args.kwargs["config"].reasoning_effort == "low"
    client.generate_structured.assert_called_once_with(
        "system",
        'Consolidez les articles extraits ci-dessous:\n\n[{"page": 1}]',
        magazine_extraction.MagazineIndex,
    )


def test_magazine_batch_requires_every_provider_key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    process_magazine = Mock()

    exit_code = magazine_extraction.run_magazine_batch(
        process_magazine,
        script_dir=tmp_path,
        intro_panel=Panel("test"),
        api_key_env=("GEMINI_API_KEY", "OPENROUTER_API_KEY"),
    )

    assert exit_code == 1
    process_magazine.assert_not_called()


def test_mistral_ocr_client_has_finite_timeout(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(mistral_pipeline, "Mistral", constructor)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mistral_pipeline._mistral_client_from_env()

    constructor.assert_called_once_with(
        api_key="test-key",
        timeout_ms=mistral_pipeline.MISTRAL_OCR_TIMEOUT_MS,
    )


def test_mistral_upload_cleanup_is_best_effort():
    client = Mock()
    client.files.delete.side_effect = RuntimeError("network failure")

    mistral_pipeline._delete_mistral_upload(client, "file-123")

    client.files.delete.assert_called_once_with(file_id="file-123")


# --- Claude Sonnet 5.5 as the page reader (profile "sonnet") -----------------

GEMINI_SCRIPT = SCRIPT.with_name("02_AI_generate_summaries_issue.py")
_GSPEC = importlib.util.spec_from_file_location("gemini_magazine_pipeline", GEMINI_SCRIPT)
gemini_pipeline = importlib.util.module_from_spec(_GSPEC)
sys.modules[_GSPEC.name] = gemini_pipeline
_GSPEC.loader.exec_module(gemini_pipeline)


def _claude_response(text="", stop_reason="end_turn"):
    from types import SimpleNamespace
    blocks = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def _claude_call(client):
    from common.rate_limiter import RateLimiter
    return gemini_pipeline.generate_with_claude(
        client, "claude-sonnet-5-5", b"%PDF", 4, {"max_tokens": 16000}, RateLimiter(None),
    )


def test_each_profile_stays_with_one_provider():
    """One profile, one provider, one API key; the consolidator is stamped by 03."""
    from common.iwac_config import AI_MODEL_ITEMS

    for profile, (reader, consolidator) in gemini_pipeline.PROFILES.items():
        step1, step2 = gemini_pipeline.get_model_pair(profile)
        assert (step1.key, step2.key) == (reader, consolidator)
        assert step1.provider == step2.provider, profile
        assert step2.key in AI_MODEL_ITEMS, profile
    assert gemini_pipeline.PROFILES["light"] == ("gemini-3.8-flash", "gemini-3.8-flash")
    assert gemini_pipeline.PROFILES["sonnet"] == ("claude-sonnet-5-5", "claude-sonnet-5-5")
    assert gemini_pipeline.PROFILE_API_KEYS["sonnet"] == "ANTHROPIC_API_KEY"


def test_claude_page_extraction_parses_the_schema():
    client = Mock()
    client.messages.create.return_value = _claude_response(
        '{"page_number": 1, "articles": [{"titre": "Le Hajj", "resume": "Récit."}]}'
    )
    extraction = _claude_call(client)
    assert extraction.page_number == 4          # the loop's number wins
    assert extraction.articles[0].titre == "Le Hajj"
    document = client.messages.create.call_args.kwargs["messages"][0]["content"][0]
    assert document["source"]["media_type"] == "application/pdf"


def test_claude_refusal_is_not_retried():
    import pytest
    from common.retry import PermanentError

    client = Mock()
    client.messages.create.return_value = _claude_response(stop_reason="refusal")
    with pytest.raises(PermanentError):
        _claude_call(client)
    assert client.messages.create.call_count == 1


def test_claude_empty_credit_stops_the_batch():
    import anthropic
    import httpx2
    import pytest
    from common.rate_limiter import QuotaExhaustedError

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = Mock()
    client.messages.create.side_effect = anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx2.Response(400, request=request), body=None,
    )
    with pytest.raises(QuotaExhaustedError):
        _claude_call(client)
    assert client.messages.create.call_count == 1
