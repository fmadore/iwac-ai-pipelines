"""Serializable model settings for reproducible text-pipeline artifacts."""
from dataclasses import asdict

from common.llm_registry import (
    LLMConfig,
    ModelOption,
    PROVIDER_GEMINI,
    clamp_thinking_level,
    model_defaults,
    resolve_reasoning_effort,
)


def model_context(option: ModelOption, config: LLMConfig | None = None) -> dict:
    """The model and configuration a run sends, as recorded in its checkpoint.

    Levels are recorded after the same snapping the client applies, so the
    record names what the provider received, not what the pipeline asked for.
    Reasoning effort is resolved only for models that declare a ladder; for the
    rest the requested value is kept, so existing checkpoints still match.
    """
    effective = (config or LLMConfig()).merged_over(model_defaults(option))
    values = asdict(effective)
    if option.provider == PROVIDER_GEMINI and effective.thinking_level:
        values["thinking_level"] = clamp_thinking_level(option.model, effective.thinking_level)
    if option.supported_reasoning_efforts:
        values["reasoning_effort"] = resolve_reasoning_effort(option, effective.reasoning_effort)
    return {"model_key": option.key, "model_id": option.model, "provider": option.provider,
            "configuration": {key: value for key, value in values.items() if value is not None}}
