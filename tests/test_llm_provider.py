"""Tests for common.llm_provider model selection and config merging."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from common.llm_provider import (
    LLMConfig,
    MODEL_REGISTRY,
    OpenAIResponsesClient,
    get_model_option,
    normalize_model_key,
)


def test_alias_normalization():
    assert normalize_model_key("gemini") == "gemini-flash"
    assert normalize_model_key("openai") == "gpt-5.6-luna"
    assert normalize_model_key("mistral") == "mistral-large"
    assert normalize_model_key("GEMINI") == "gemini-flash"
    assert normalize_model_key(None) is None


def test_gpt_56_tier_aliases():
    assert normalize_model_key("luna") == "gpt-5.6-luna"
    assert normalize_model_key("terra") == "gpt-5.6-terra"
    assert normalize_model_key("sol") == "gpt-5.6-sol"
    # The bare id routes to Sol, matching OpenAI's own routing.
    assert normalize_model_key("gpt-5.6") == "gpt-5.6-sol"


def test_retired_openai_keys_map_forward():
    # GPT-5/5.1 snapshots shut down 2026-10-23; old keys must keep resolving.
    assert normalize_model_key("gpt-5-mini") == "gpt-5.6-luna"
    assert normalize_model_key("gpt-5.1") == "gpt-5.6-sol"
    assert normalize_model_key("gpt-5") == "gpt-5.6-sol"
    assert get_model_option("gpt-5-mini").model == "gpt-5.6-luna"


def test_get_model_option_by_key():
    option = get_model_option("gemini-flash")
    assert option is MODEL_REGISTRY["gemini-flash"]


def test_get_model_option_via_alias():
    option = get_model_option("gemini")
    assert option.key == "gemini-flash"


def test_allowed_keys_accept_aliases():
    # allowed_keys entries are normalized too: 'gemini' used to be rejected
    # even when the resolved key was allowed.
    option = get_model_option("gemini", allowed_keys=["gemini-flash", "gpt-5.6-luna"])
    assert option.key == "gemini-flash"


def test_disallowed_model_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        get_model_option("mistral-large", allowed_keys=["gemini-flash"])


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        get_model_option("gpt-99")


def test_merged_over_prefers_explicit_values():
    base = LLMConfig(temperature=0.2, reasoning_effort="low")
    override = LLMConfig(temperature=0.7)
    merged = override.merged_over(base)
    assert merged.temperature == 0.7
    assert merged.reasoning_effort == "low"


def test_merged_over_honors_zero_temperature():
    # 0.0 is falsy but explicitly set; it must NOT fall back to the default.
    base = LLMConfig(temperature=0.2)
    merged = LLMConfig(temperature=0.0).merged_over(base)
    assert merged.temperature == 0.0


def test_merged_over_honors_store_false():
    # store=False is falsy but explicitly set; it must NOT fall back to the default.
    merged = LLMConfig(store=False).merged_over(LLMConfig(store=True))
    assert merged.store is False


def test_registry_and_aliases_are_consistent():
    from common.llm_provider import MODEL_ALIASES

    for alias, target in MODEL_ALIASES.items():
        assert target in MODEL_REGISTRY, f"alias {alias!r} points to unknown key {target!r}"
    for key, option in MODEL_REGISTRY.items():
        assert option.key == key


# ---------------------------------------------------------------------------
# OpenAI structured output
# ---------------------------------------------------------------------------

class _Sample(BaseModel):
    """Schema with a defaulted field — the case that broke the hand-rolled path."""

    required_field: str
    optional_field: Optional[int] = Field(default=None)


def _openai_client_with_stub(monkeypatch, parsed=None, output=None):
    """Build an OpenAIResponsesClient whose SDK client is a stub."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    client = OpenAIResponsesClient(MODEL_REGISTRY["gpt-5.6-luna"])
    stub = MagicMock()
    stub.responses.parse.return_value = MagicMock(output_parsed=parsed, output=output or [])
    client._client = stub
    return client, stub


def test_structured_output_uses_parse_with_pydantic_model(monkeypatch):
    """The schema must go through responses.parse, not a hand-built json_schema.

    model_json_schema() emits no additionalProperties:false and drops defaulted
    fields from `required`, both of which OpenAI's strict mode rejects.
    """
    expected = _Sample(required_field="ok")
    client, stub = _openai_client_with_stub(monkeypatch, parsed=expected)

    result = client.generate_structured("system", "user", _Sample)

    assert result is expected
    stub.responses.create.assert_not_called()
    kwargs = stub.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is _Sample
    # No hand-rolled schema smuggled in via text=
    assert "format" not in kwargs.get("text", {})


def test_structured_output_does_not_store_by_default(monkeypatch):
    """Full archival documents should not be retained server-side."""
    client, stub = _openai_client_with_stub(monkeypatch, parsed=_Sample(required_field="ok"))

    client.generate_structured("system", "user", _Sample)

    assert stub.responses.parse.call_args.kwargs["store"] is False


def test_structured_output_surfaces_refusal(monkeypatch):
    """A refusal must not be reported as an empty response."""
    refusal_item = MagicMock(content=[MagicMock(refusal="cannot comply")])
    client, _ = _openai_client_with_stub(monkeypatch, parsed=None, output=[refusal_item])

    with pytest.raises(ValueError, match="refused"):
        client.generate_structured("system", "user", _Sample)
