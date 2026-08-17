"""Tests for common.llm_provider model selection and config merging."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from common.llm_provider import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TEXT_MODEL_KEY,
    GeminiGenerateContentClient,
    LLMConfig,
    MODEL_REGISTRY,
    OpenAIResponsesClient,
    OpenRouterClient,
    PROVIDER_SELFHOSTED,
    SELFHOSTED_QWEN38_MODEL,
    SelfHostedClient,
    TEXT_ECONOMY_MODELS,
    TEXT_EXTENDED_MODELS,
    TEXT_FULL_MODELS,
    TEXT_OPEN_MODELS,
    clamp_thinking_level,
    get_model_option,
    normalize_model_key,
)


def test_alias_normalization():
    # The bare vendor name resolves to the *pinned* current Flash, not to the
    # rolling ``gemini-flash``: whatever a bare alias resolves to can end up
    # stamped in an iwac:*Model annotation, and a rolling id cannot be cited.
    assert normalize_model_key("gemini") == "gemini-3.7-flash"
    assert normalize_model_key("openai") == "gpt-5.6-luna"
    assert normalize_model_key("mistral") == "mistral-large"
    assert normalize_model_key("GEMINI") == "gemini-3.7-flash"
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
    assert option.key == "gemini-3.7-flash"


def test_allowed_keys_accept_aliases():
    # allowed_keys entries are normalized too: 'gemini' used to be rejected
    # even when the resolved key was allowed.
    option = get_model_option("gemini", allowed_keys=["gemini-3.7-flash", "gpt-5.6-luna"])
    assert option.key == "gemini-3.7-flash"


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


def test_default_transport_timeout_is_finite(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    constructor = MagicMock()
    monkeypatch.setattr("common.llm_provider.OpenAI", constructor)

    OpenAIResponsesClient(MODEL_REGISTRY["gpt-5.6-luna"])

    assert constructor.call_args.kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert 0 < DEFAULT_REQUEST_TIMEOUT_SECONDS < 600


def test_pipeline_can_own_sdk_retry_budget(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    constructor = MagicMock()
    monkeypatch.setattr("common.llm_provider.OpenAI", constructor)

    OpenRouterClient(
        MODEL_REGISTRY["deepseek-v4-flash-0731"],
        LLMConfig(request_timeout_seconds=30.0, sdk_max_retries=0),
    )

    assert constructor.call_args.kwargs["timeout"] == 30.0
    assert constructor.call_args.kwargs["max_retries"] == 0


def test_registry_and_aliases_are_consistent():
    from common.llm_provider import MODEL_ALIASES

    for alias, target in MODEL_ALIASES.items():
        assert target in MODEL_REGISTRY, f"alias {alias!r} points to unknown key {target!r}"
    for key, option in MODEL_REGISTRY.items():
        assert option.key == key


def test_provider_facade_reexports_the_dependency_free_registry():
    from common import llm_provider, llm_registry

    assert llm_provider.MODEL_REGISTRY is llm_registry.MODEL_REGISTRY
    assert llm_provider.LLMConfig is llm_registry.LLMConfig
    assert llm_provider.get_model_option is llm_registry.get_model_option


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

def _openrouter_client_with_stub(monkeypatch, key="qwen3.5-moe", message=None):
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
    assert normalize_model_key("qwen") == "qwen3.5-moe"
    assert normalize_model_key("deepseek") == "deepseek-v4-flash-0731"
    assert normalize_model_key("deepseek-pro") == "deepseek-v4-pro"
    # A slug pasted straight off openrouter.ai must resolve too — and to the
    # entry for that exact model. "qwen3.5-moe" moved from 35B-A3B to
    # 122B-A10B on 2026-07-31, so the old slug must land on the small entry
    # rather than silently following the bare "qwen" alias to a bigger model.
    assert normalize_model_key("qwen/qwen3.5-122b-a10b") == "qwen3.5-moe"
    assert normalize_model_key("qwen/qwen3.5-35b-a3b") == "qwen3.5-moe-small"
    assert MODEL_REGISTRY["qwen3.5-moe"].model == "qwen/qwen3.5-122b-a10b"
    assert MODEL_REGISTRY["qwen3.5-moe-small"].model == "qwen/qwen3.5-35b-a3b"
    assert MODEL_REGISTRY["deepseek-v4-flash-0731"].model \
        == "deepseek/deepseek-v4-flash-0731"
    assert MODEL_REGISTRY["deepseek-v4-flash"].model \
        == "deepseek/deepseek-v4-flash"
    assert DEFAULT_TEXT_MODEL_KEY == "deepseek-v4-flash-0731"


def test_openrouter_models_are_offered_by_the_right_tiers():
    # NER runs on the extended tier; the two Flash models must be selectable there.
    assert "qwen3.5-moe" in TEXT_EXTENDED_MODELS
    assert "deepseek-v4-flash-0731" in TEXT_EXTENDED_MODELS
    # Pro is a quality tier: full only, not extended.
    assert "deepseek-v4-pro" in TEXT_FULL_MODELS
    assert "deepseek-v4-pro" not in TEXT_EXTENDED_MODELS
    assert set(TEXT_OPEN_MODELS) <= set(MODEL_REGISTRY)


def test_the_deepseek_preview_is_archive_only():
    """Every DeepSeek Flash run goes to the dated 0731 release.

    The April preview keeps its ``MODEL_REGISTRY`` entry so its slug still
    resolves where it survives in old artifacts, but a tier is what a pipeline
    passes to ``choices=``, so membership in one is what decides whether a run
    can still land on it. Being merely *superseded* in a description had not
    stopped it being selectable.
    """
    for tier in (TEXT_ECONOMY_MODELS, TEXT_OPEN_MODELS,
                 TEXT_EXTENDED_MODELS, TEXT_FULL_MODELS):
        assert "deepseek-v4-flash" not in tier
    assert "deepseek-v4-flash" in MODEL_REGISTRY
    assert "deepseek-v4-flash-0731" in TEXT_OPEN_MODELS


def test_openrouter_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(MODEL_REGISTRY["qwen3.5-moe"])


def test_openrouter_denies_provider_data_collection(monkeypatch):
    """Whole archival documents must not reach a backend that may retain them."""
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user")

    provider = stub.chat.completions.create.call_args.kwargs["extra_body"]["provider"]
    assert provider["data_collection"] == "deny"
    # Without require_parameters a request can land on a backend that ignores
    # response_format and answers in prose.
    assert provider["require_parameters"] is True


def test_openrouter_forwards_medium_effort(monkeypatch):
    """medium is accepted by the OpenRouter models and must reach the request.

    Verified live 2026-07-29: both qwen3.5-* and deepseek-v4-* route and return
    reasoning_details at effort=medium. An earlier registry restricted them to
    high/xhigh, which silently downgraded a medium request to no reasoning at
    all — the sentiment panel is standardised on medium, so that mattered.
    """
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user", config=LLMConfig(reasoning_effort="medium"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["reasoning"] == {"effort": "medium"}


def test_openrouter_omits_unsupported_reasoning_effort(monkeypatch):
    """An effort the model does not declare is dropped, not forwarded.

    Forwarding it is worse than dropping it: with require_parameters on it can
    leave the request with no eligible backend at all.
    """
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user", config=LLMConfig(reasoning_effort="max"))

    assert "reasoning" not in stub.chat.completions.create.call_args.kwargs["extra_body"]


def test_openrouter_sends_supported_reasoning_effort(monkeypatch):
    client, stub = _openrouter_client_with_stub(
        monkeypatch, key="deepseek-v4-flash", message=_message(content="ok")
    )

    client.generate("system", "user", config=LLMConfig(reasoning_effort="xhigh"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["reasoning"] == {"effort": "xhigh"}


def test_deepseek_0731_uses_exact_reasoning_levels(monkeypatch):
    client, stub = _openrouter_client_with_stub(
        monkeypatch, key="deepseek-v4-flash-0731", message=_message(content="ok")
    )

    client.generate("system", "user", config=LLMConfig(reasoning_effort="max"))
    assert stub.chat.completions.create.call_args.kwargs["extra_body"]["reasoning"] \
        == {"effort": "max"}

    # The official release has no medium effort. Bulk pipelines clamp to its
    # low default; the sentiment panel explicitly asks for high.
    client.generate("system", "user", config=LLMConfig(reasoning_effort="medium"))
    assert stub.chat.completions.create.call_args.kwargs["extra_body"]["reasoning"] \
        == {"effort": "low"}


def test_openrouter_pro_reasons_by_default_and_clamps(monkeypatch):
    client, stub = _openrouter_client_with_stub(
        monkeypatch, key="deepseek-v4-pro", message=_message(content="ok")
    )

    # 'max' is not accepted by V4 Pro: fall back to the model's own default
    # rather than forwarding a value that would strand the request.
    client.generate("system", "user", config=LLMConfig(reasoning_effort="max"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["reasoning"] == {"effort": "high"}


def test_openrouter_structured_sends_the_strict_schema(monkeypatch):
    """The schema must go out exactly as ``parse()`` would have sent it.

    Strict mode rewrites a Pydantic schema — ``additionalProperties: false`` and
    every field in ``required`` — and backends validate against that form, so
    building the payload by hand from ``model_json_schema()`` is not equivalent.
    """
    content = '{"required_field": "ok", "optional_field": 3}'
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content=content))

    result = client.generate_structured("system", "user", _Sample)

    assert result.required_field == "ok"
    sent = stub.chat.completions.create.call_args.kwargs["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["schema"]["additionalProperties"] is False
    assert set(sent["json_schema"]["schema"]["required"]) == {
        "required_field", "optional_field",
    }
    # parse() sent this explicitly; the switch to create() must not drop it.
    assert stub.chat.completions.create.call_args.kwargs["stream"] is False


def test_structured_output_never_delegates_parsing_to_the_sdk(monkeypatch):
    """A regression guard, and the reason the recovery path exists at all.

    ``chat.completions.parse`` validates ``message.content`` itself and raises
    before returning, so fenced JSON became a ``ValidationError`` no caller could
    recover from — while the fallback below looked well covered, because mocking
    ``parse()`` hands back content the real SDK would never have returned. Any
    reintroduction of ``parse()`` here silently kills that recovery again.
    """
    fenced = '```json\n{"required_field": "ok"}\n```'
    for client, stub in (
        _openrouter_client_with_stub(monkeypatch, message=_message(content=fenced)),
        _selfhosted_client_with_stub(monkeypatch, message=_message(content=fenced)),
    ):
        assert client.generate_structured("s", "u", _Sample).required_field == "ok"
        stub.chat.completions.parse.assert_not_called()
        stub.chat.completions.create.assert_called_once()


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


def test_structured_schema_conversion_survives_an_sdk_without_the_helper(monkeypatch):
    """``type_to_response_format_param`` is a private SDK path.

    pyproject allows openai>=1.60 through <3, so the helper may move. The
    fallback must still produce a well-formed json_schema payload rather than
    crash — it only loses the strict rewriting, which a backend may reject
    outright but cannot mistake for a different request.
    """
    monkeypatch.setattr("common.llm_provider.type_to_response_format_param", None)
    content = '{"required_field": "ok"}'
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content=content))

    assert client.generate_structured("s", "u", _Sample).required_field == "ok"

    sent = stub.chat.completions.create.call_args.kwargs["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "_Sample"
    assert "required_field" in sent["json_schema"]["schema"]["properties"]


def test_openrouter_ignores_reasoning_trace_in_content(monkeypatch):
    """Only `content` is the answer; a reasoning trace must not leak into it."""
    client, _ = _openrouter_client_with_stub(monkeypatch, message=_message(content="  answer  "))

    assert client.generate("system", "user") == "answer"


# ---------------------------------------------------------------------------
# Self-hosted endpoint
# ---------------------------------------------------------------------------
#
# An OpenAI-compatible server you run yourself. It reuses the whole OpenRouter
# transport, so the tests below cover only what differs: where the endpoint comes
# from, and what must NOT be sent to it.


def _selfhosted_client_with_stub(monkeypatch, key="qwen3.8-27b-selfhosted",
                                 message=None, api_key="sk-test"):
    monkeypatch.setenv("SELFHOSTED_LLM_BASE_URL", "http://localhost:8000/v1")
    if api_key is None:
        monkeypatch.delenv("SELFHOSTED_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SELFHOSTED_LLM_API_KEY", api_key)
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    client = SelfHostedClient(MODEL_REGISTRY[key])
    stub = MagicMock()
    response = MagicMock(choices=[MagicMock(message=message)] if message is not None else [])
    stub.chat.completions.create.return_value = response
    stub.chat.completions.parse.return_value = response
    client._client = stub
    return client, stub


def test_selfhosted_requires_a_base_url(monkeypatch):
    """No endpoint, no client — and the error must name the variable to set.

    Failing in the constructor is what lets the sentiment pilot report this
    model as *skipped* on a laptop with no tunnel open, instead of dying
    mid-corpus. CI has no endpoint either, which is the same case.
    """
    monkeypatch.delenv("SELFHOSTED_LLM_BASE_URL", raising=False)
    monkeypatch.setattr("common.llm_provider.OpenAI", MagicMock())

    with pytest.raises(RuntimeError, match="SELFHOSTED_LLM_BASE_URL"):
        SelfHostedClient(MODEL_REGISTRY["qwen3.8-27b-selfhosted"])


def test_selfhosted_reads_its_endpoint_from_the_environment(monkeypatch):
    """The URL is deployment state: on a cluster it changes every job."""
    monkeypatch.setenv("SELFHOSTED_LLM_BASE_URL", "http://localhost:8123/v1")
    monkeypatch.setenv("SELFHOSTED_LLM_API_KEY", "sk-secret")
    openai_cls = MagicMock()
    monkeypatch.setattr("common.llm_provider.OpenAI", openai_cls)

    SelfHostedClient(MODEL_REGISTRY["qwen3.8-27b-selfhosted"])

    kwargs = openai_cls.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:8123/v1"
    assert kwargs["api_key"] == "sk-secret"
    assert kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_selfhosted_tolerates_a_server_without_a_key(monkeypatch):
    """vLLM only demands a key when started with --api-key, but the SDK always
    wants one; "EMPTY" is the convention vLLM's own documentation uses."""
    monkeypatch.setenv("SELFHOSTED_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("SELFHOSTED_LLM_API_KEY", raising=False)
    openai_cls = MagicMock()
    monkeypatch.setattr("common.llm_provider.OpenAI", openai_cls)

    SelfHostedClient(MODEL_REGISTRY["qwen3.8-27b-selfhosted"])

    assert openai_cls.call_args.kwargs["api_key"] == "EMPTY"


def test_selfhosted_sends_no_routing_prefs_and_no_attribution_headers(monkeypatch):
    """The counterpart to ``test_openrouter_denies_provider_data_collection``.

    That test pins a *contract* with a third party: please do not retain this
    archival article. Here there is no third party to ask — the text reaches a
    machine the operator controls and goes no further, so the guarantee is
    physical rather than contractual. Sending routing preferences anyway would
    be meaningless at best, and a strict server may reject body fields it does
    not recognise.
    """
    client, stub = _selfhosted_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user")

    kwargs = stub.chat.completions.create.call_args.kwargs
    assert "provider" not in kwargs["extra_body"]
    assert kwargs["extra_headers"] == {}


def test_selfhosted_forwards_reasoning_via_chat_template_kwargs(monkeypatch):
    """vLLM passes template arguments, not an OpenRouter-style reasoning block.

    Qwen3.8 reads ``reasoning_effort`` out of its chat template, so the depth
    has to travel there. Its own default is ``xhigh``; the registry asks for
    ``low`` so an unconfigured bulk run does not reason as hard as it can on
    shared GPU hours.
    """
    client, stub = _selfhosted_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user", config=LLMConfig(reasoning_effort="medium"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_selfhosted_drops_an_undeclared_reasoning_level(monkeypatch):
    """Qwen3.8's ladder is low/medium/xhigh — there is no ``high`` rung.

    NER asks the whole pipeline for "medium" and other callers for "high"; an
    effort this model does not declare degrades to its default rather than being
    forwarded into a template that has no branch for it.
    """
    client, stub = _selfhosted_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user", config=LLMConfig(reasoning_effort="high"))

    body = stub.chat.completions.create.call_args.kwargs["extra_body"]
    assert body == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_selfhosted_structured_falls_back_to_raw_content(monkeypatch):
    """The recovery path matters more here, not less: there is no router
    filtering for backends that honour ``response_format``."""
    fenced = '```json\n{"required_field": "ok", "optional_field": 3}\n```'
    client, _ = _selfhosted_client_with_stub(monkeypatch, message=_message(content=fenced))

    result = client.generate_structured("system", "user", _Sample)

    assert result.required_field == "ok"
    assert result.optional_field == 3


def test_selfhosted_errors_name_the_right_route(monkeypatch):
    """A failure on a machine down the hall must not blame OpenRouter."""
    client, _ = _selfhosted_client_with_stub(monkeypatch, message=_message(content=""))

    with pytest.raises(ValueError, match="self-hosted"):
        client.generate_structured("system", "user", _Sample)


def test_qwen38_resolves_by_route():
    """Same weights, two routes, and the two names collide once lowercased.

    ``Qwen/Qwen3.8-27B`` (the Hugging Face id) normalizes to exactly the
    OpenRouter slug ``qwen/qwen3.8-27b``, so the vendor-prefixed form is given
    to the hosted route — as it is for ``google/gemma-4-31b-it`` — and the bare
    names mean the endpoint you run yourself.
    """
    assert normalize_model_key("qwen3.8") == "qwen3.8-27b-selfhosted"
    assert normalize_model_key("qwen3.8-27b") == "qwen3.8-27b-selfhosted"
    assert normalize_model_key("qwen/qwen3.8-27b") == "qwen3.8-27b-openrouter"
    assert normalize_model_key("Qwen/Qwen3.8-27B") == "qwen3.8-27b-openrouter"

    selfhosted = MODEL_REGISTRY["qwen3.8-27b-selfhosted"]
    assert selfhosted.provider == PROVIDER_SELFHOSTED
    # The served name vLLM reports from /v1/models is the repo id it launched
    # with; a different --served-model-name means changing this string too.
    assert selfhosted.model == SELFHOSTED_QWEN38_MODEL == "Qwen/Qwen3.8-27B"
    assert MODEL_REGISTRY["qwen3.8-27b-openrouter"].model == "qwen/qwen3.8-27b"


def test_the_qwen38_openrouter_twin_is_comparison_only():
    """It exists to measure one route against the other, not to be picked.

    At $0.45/$3.20 per 1M it is roughly twice the sentiment panel's output-cost
    band — the reason the experiment went to a GPU cluster at all. Tier
    membership is what makes a model reachable from a pipeline's ``--model``,
    so staying out of every tier is what keeps a full-corpus run off it.
    """
    for tier in (TEXT_ECONOMY_MODELS, TEXT_OPEN_MODELS,
                 TEXT_EXTENDED_MODELS, TEXT_FULL_MODELS):
        assert "qwen3.8-27b-openrouter" not in tier
    assert "qwen3.8-27b-openrouter" in MODEL_REGISTRY
    assert "qwen3.8-27b-selfhosted" in TEXT_OPEN_MODELS


# --- Sampling temperature -------------------------------------------------
#
# Temperature is a per-vendor decision recorded in MODEL_REGISTRY, not a knob
# pipelines turn. Google, Alibaba and DeepSeek all warn that lowering it degrades
# their models — Gemini 3 and Qwen name looping/endless repetition specifically.


def test_gemini_models_declare_no_temperature():
    """Google recommends sending no temperature at all for Gemini 3."""
    for key in ("gemini-3.7-flash", "gemini-flash", "gemini-flash-lite",
                "gemini-pro", "gemma-4"):
        assert MODEL_REGISTRY[key].default_temperature is None, key


def test_vendor_temperature_defaults_match_published_guidance():
    # DeepSeek V4 model card: "temperature = 1.0, top_p = 1.0" for every mode.
    assert MODEL_REGISTRY["deepseek-v4-flash"].default_temperature == 1.0
    assert MODEL_REGISTRY["deepseek-v4-flash-0731"].default_temperature == 1.0
    assert MODEL_REGISTRY["deepseek-v4-pro"].default_temperature == 1.0
    # Qwen's published non-thinking recipe.
    assert MODEL_REGISTRY["qwen3.5-moe"].default_temperature == 0.7
    # Qwen3.8 carries the *thinking-mode* recipe instead — 1.0 (with top_p 0.95,
    # top_k 20, both left to the server), which is what generation_config.json
    # ships. The 0.7 above is the same vendor's non-thinking figure, and these
    # models always run thinking-on: inheriting 0.7 from the neighbouring entry
    # would be a quiet misconfiguration, not a rounding.
    for key in ("qwen3.8-27b-selfhosted", "qwen3.8-27b-openrouter"):
        assert MODEL_REGISTRY[key].default_temperature == 1.0, key
    # Mistral is the one vendor recommending a low value (0.05-0.20).
    assert MODEL_REGISTRY["mistral-large"].default_temperature == 0.2
    assert MODEL_REGISTRY["ministral-14b"].default_temperature == 0.2


def _gemini_client(monkeypatch, key="gemini-flash"):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr("common.llm_provider.genai", MagicMock())
    return GeminiGenerateContentClient(MODEL_REGISTRY[key])


def test_gemini_generation_config_omits_temperature(monkeypatch):
    """The key must be absent, not set to 1.0.

    Sending the nominal default is not the same request as sending nothing, and
    the recommendation is to send nothing.
    """
    client = _gemini_client(monkeypatch)

    kwargs = client._build_generation_config(client.config)

    assert "temperature" not in kwargs


def test_gemini_still_honors_an_explicit_temperature(monkeypatch):
    """Overriding stays possible — it just is not the default any more."""
    client = _gemini_client(monkeypatch)

    kwargs = client._build_generation_config(LLMConfig(temperature=0.4).merged_over(client.config))

    assert kwargs["temperature"] == 0.4


# --- Thinking levels ------------------------------------------------------
#
# Google's thinking ladder is per-model and changes between releases: Gemini 3.7
# Flash dropped MINIMAL, and ``gemini-flash-latest`` rolled onto it the same day.
# Every pipeline here asked for MINIMAL, so all of them started 400ing at once.
# The registry now states which rungs exist and clamps to the nearest.


def test_gemini_37_flash_has_no_minimal_rung():
    """Verified against the live API on 2026-08-14: LOW is the floor."""
    assert MODEL_REGISTRY["gemini-3.7-flash"].supported_thinking_levels == (
        "low", "medium", "high",
    )
    assert MODEL_REGISTRY["gemini-3.7-flash"].default_thinking_level == "LOW"


def test_rolling_flash_alias_tracks_37s_ladder():
    """``gemini-flash-latest`` resolves to 3.7, so it lost MINIMAL too."""
    option = MODEL_REGISTRY["gemini-flash"]
    assert "minimal" not in option.supported_thinking_levels
    assert option.default_thinking_level == "LOW"


def test_clamp_rounds_minimal_up_to_the_shallowest_rung_that_exists():
    assert clamp_thinking_level("gemini-3.7-flash", "minimal") == "low"
    assert clamp_thinking_level("gemini-flash-latest", "MINIMAL") == "low"
    assert clamp_thinking_level("gemini-pro-latest", "minimal") == "low"


def test_clamp_leaves_supported_levels_alone():
    assert clamp_thinking_level("gemini-3.7-flash", "high") == "high"
    assert clamp_thinking_level("gemini-3.6-flash", "minimal") == "minimal"
    assert clamp_thinking_level("gemini-3.5-flash-lite", "minimal") == "minimal"
    assert clamp_thinking_level("gemini-3.7-flash", None) is None


def test_clamp_reproduces_the_gemma_mapping_it_replaced():
    """Gemma has only MINIMAL and HIGH; ties round up, so MEDIUM -> HIGH.

    This was hand-coded in the Gemini adapter before the ladder became data.
    The panel asks Gemma for "medium" and must land on high, not on a
    non-reasoning mode.
    """
    assert clamp_thinking_level("gemma-4-31b-it", "medium") == "high"
    assert clamp_thinking_level("gemma-4-31b-it", "low") == "minimal"
    assert clamp_thinking_level("gemma-4-31b-it", "high") == "high"


def test_clamp_passes_through_unknown_models_and_levels():
    """Guessing a restriction would silently downgrade an unprobed model."""
    assert clamp_thinking_level("some-future-model", "minimal") == "minimal"
    assert clamp_thinking_level("gemini-3.7-flash", "ludicrous") == "ludicrous"


def test_gemini_client_clamps_before_calling_the_sdk(monkeypatch):
    """The end-to-end guard: a MINIMAL request must not reach 3.7 Flash."""
    client = _gemini_client(monkeypatch, key="gemini-3.7-flash")

    kwargs = client._build_generation_config(
        LLMConfig(thinking_level="minimal").merged_over(client.config)
    )

    assert kwargs["thinking_config"].thinking_level == "LOW"


def test_openrouter_sends_the_vendor_default_temperature(monkeypatch):
    """A pipeline that sets no temperature gets Qwen's 0.7, not a low default."""
    client, stub = _openrouter_client_with_stub(monkeypatch, message=_message(content="ok"))

    client.generate("system", "user")

    assert stub.chat.completions.create.call_args.kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Mistral reasoning mode
#
# Mistral Small 4 is a hybrid instruct/reasoning model whose API accepts only
# reasoning_effort none|high, and which switches message.content from a string
# to a thinking/text chunk list once reasoning is on. Both facts broke real
# calls before they were handled; these lock in the fixes.
# ---------------------------------------------------------------------------

def test_mistral_content_text_plain_string():
    from common.llm_provider import MistralClient

    assert MistralClient._content_text("hello") == "hello"
    assert MistralClient._content_text(None) == ""


def test_mistral_content_text_drops_thinking_chunk():
    """The thinking chunk is a scratchpad, not the answer — it must not be parsed."""
    from common.llm_provider import MistralClient

    content = [
        {"type": "thinking", "thinking": ["let me reason about this"], "closed": True},
        {"type": "text", "text": '{"polarite": "Neutre"}'},
    ]
    assert MistralClient._content_text(content) == '{"polarite": "Neutre"}'


def test_mistral_content_text_handles_sdk_objects():
    """Chunks arrive as SDK objects on some call paths, plain dicts on others."""
    from common.llm_provider import MistralClient

    class _Chunk:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    content = [_Chunk("thinking"), _Chunk("text", '{"ok": true}')]
    assert MistralClient._content_text(content) == '{"ok": true}'


def test_mistral_rounds_medium_effort_up_to_high():
    """The panel asks for medium; Mistral has no medium, so it must not 400.

    Rounding up keeps Mistral in the reasoning regime like the rest of the
    panel; rounding down to 'none' would make it the only non-reasoning member.
    """
    from common.llm_provider import MODEL_REGISTRY, LLMConfig, MistralClient

    option = MODEL_REGISTRY["mistral-small"]
    assert option.supported_reasoning_efforts == ("none", "high")

    client = MistralClient.__new__(MistralClient)
    client.option = option
    resolved = client._resolve_reasoning_effort(LLMConfig(reasoning_effort="medium"))
    assert resolved == "high"


def test_mistral_forwards_supported_effort_unchanged():
    from common.llm_provider import MODEL_REGISTRY, LLMConfig, MistralClient

    client = MistralClient.__new__(MistralClient)
    client.option = MODEL_REGISTRY["mistral-small"]
    assert client._resolve_reasoning_effort(LLMConfig(reasoning_effort="high")) == "high"
    assert client._resolve_reasoning_effort(LLMConfig(reasoning_effort="none")) == "none"


def test_mistral_without_reasoning_support_sends_nothing():
    """Ministral 14B declares no efforts, so none is sent at all."""
    from common.llm_provider import MODEL_REGISTRY, LLMConfig, MistralClient

    client = MistralClient.__new__(MistralClient)
    client.option = MODEL_REGISTRY["ministral-14b"]
    assert client._resolve_reasoning_effort(LLMConfig(reasoning_effort="medium")) is None
