"""Tests for common.llm_provider model selection and config merging."""

import pytest

from common.llm_provider import (
    LLMConfig,
    MODEL_REGISTRY,
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


def test_registry_and_aliases_are_consistent():
    from common.llm_provider import MODEL_ALIASES

    for alias, target in MODEL_ALIASES.items():
        assert target in MODEL_REGISTRY, f"alias {alias!r} points to unknown key {target!r}"
    for key, option in MODEL_REGISTRY.items():
        assert option.key == key
