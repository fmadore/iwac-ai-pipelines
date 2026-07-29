"""Tests for common.llm_provider model selection and config merging."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from common.llm_provider import (
    LLMConfig,
    MODEL_REGISTRY,
    OpenAIResponsesClient,
    OpenRouterClient,
    TEXT_EXTENDED_MODELS,
    TEXT_FULL_MODELS,
    TEXT_OPEN_MODELS,
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


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

def _openrouter_client_with_stub(monkeypatch, key="qwen3.7-flash", message=None):
    """Build an OpenRouterClient whose SDK client is a stub."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    client = OpenRouterClient(MODEL_REGISTRY[key])
    stub = MagicMock()
    response = MagicMock(choices=[MagicMock(message=message)] if message is not None else [])
    stub.chat.completions.create.return_value = response
    stub.chat.completions.parse.return_value = response
    client._client = stub
    return client, stub


def _message(content=None, parsed=None, refusal=None):
    return MagicMock(content=content, parsed=parsed, refusal=refusal)


def test_openrouter_aliases_resolve():
    assert normalize_model_key("qwen") == "qwen3.7-flash"
    assert normalize_model_key("deepseek") == "deepseek-v4-flash"
    assert normalize_model_key("deepseek-pro") == "deepseek-v4-pro"
    # A slug pasted straight off openrouter.ai must resolve too.
    assert normalize_model_key("qwen/qwen3.7-flash") == "qwen3.7-flash"


def test_openrouter_models_are_offered_by_the_right_tiers():
    # NER runs on the extended tier; the two Flash models must be selectable there.
    assert "qwen3.7-flash" in TEXT_EXTENDED_MODELS
    assert "deepseek-v4-flash" in TEXT_EXTENDED_MODELS
    # Pro is a quality tier: full only, not extended.
    assert "deepseek-v4-pro" in TEXT_FULL_MODELS
    assert "deepseek-v4-pro" not in TEXT_EXTENDED_MODELS
    assert set(TEXT_OPEN_MODELS) <= set(MODEL_REGISTRY)


def test_openrouter_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(MODEL_REGISTRY["qwen3.7-flash"])


def test_openrouter_denies_provider_data_collection(monkeypatch):
    """Whole archival documents must not reach a backend that may retain them."""
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user")

    provider = stub.chat.completions.create.call_args.kwargs["extra_body"]["provider"]
    assert provider["data_collection"] == "deny"
    # Without require_parameters a request can land on a backend that ignores
    # response_format and answers in prose.
    assert provider["require_parameters"] is True


def test_openrouter_omits_unsupported_reasoning_effort(monkeypatch):
    """NER sets reasoning_effort='medium' for OpenAI; Qwen must not receive it."""
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user", config=LLMConfig(reasoning_effort="medium"))

    assert "reasoning" not in stub.chat.completions.create.call_args.kwargs["extra_body"]


def test_openrouter_sends_supported_reasoning_effort(monkeypatch):
    client, stub = _openrouter_client_with_stub(
        monkeypatch, key="deepseek-v4-flash", message=_message(content="ok")
    )

    client.generate("system", "user", config=LLMConfig(reasoning_effort="xhigh"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["reasoning"] == {"effort": "xhigh"}


def test_openrouter_pro_reasons_by_default_and_clamps(monkeypatch):
    client, stub = _openrouter_client_with_stub(
        monkeypatch, key="deepseek-v4-pro", message=_message(content="ok")
    )

    # 'medium' is not accepted by V4 Pro: fall back to the model's own default
    # rather than forwarding a value that would strand the request.
    client.generate("system", "user", config=LLMConfig(reasoning_effort="medium"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["reasoning"] == {"effort": "high"}


def test_openrouter_structured_prefers_parsed(monkeypatch):
    expected = _Sample(required_field="ok")
    client, stub = _openrouter_client_with_stub(
        monkeypatch, message=_message(parsed=expected)
    )

    result = client.generate_structured("system", "user", _Sample)

    assert result is expected
    assert stub.chat.completions.parse.call_args.kwargs["response_format"] is _Sample


def test_openrouter_structured_falls_back_to_raw_content(monkeypatch):
    """Open models often return schema-valid JSON as a plain fenced string."""
    fenced = '```json\n{"required_field": "ok", "optional_field": 3}\n```'
    client, _ = _openrouter_client_with_stub(monkeypatch, message=_message(content=fenced))

    result = client.generate_structured("system", "user", _Sample)

    assert result.required_field == "ok"
    assert result.optional_field == 3


def test_openrouter_structured_tolerates_prose_around_json(monkeypatch):
    noisy = 'Here is the result:\n{"required_field": "ok"}\nHope that helps.'
    client, _ = _openrouter_client_with_stub(monkeypatch, message=_message(content=noisy))

    assert client.generate_structured("system", "user", _Sample).required_field == "ok"


def test_openrouter_structured_surfaces_refusal(monkeypatch):
    client, _ = _openrouter_client_with_stub(
        monkeypatch, message=_message(refusal="cannot comply")
    )

    with pytest.raises(ValueError, match="refused"):
        client.generate_structured("system", "user", _Sample)


def test_openrouter_structured_falls_back_to_beta_parse(monkeypatch):
    """pyproject allows openai>=1.60, where parse() still lives under .beta."""
    expected = _Sample(required_field="ok")
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(parsed=expected))
    del stub.chat.completions.parse  # older SDK layout
    stub.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=_message(parsed=expected))]
    )

    assert client.generate_structured("system", "user", _Sample) is expected
    stub.beta.chat.completions.parse.assert_called_once()


def test_openrouter_ignores_reasoning_trace_in_content(monkeypatch):
    """Only `content` is the answer; a reasoning trace must not leak into it."""
    client, _ = _openrouter_client_with_stub(monkeypatch, message=_message(content="  answer  "))

    assert client.generate("system", "user") == "answer"
