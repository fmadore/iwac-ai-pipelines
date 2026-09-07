"""Serializable model settings for reproducible text-pipeline artifacts."""
from dataclasses import asdict

from common.llm_registry import LLMConfig, ModelOption, PROVIDER_GEMINI, clamp_thinking_level


def model_context(option: ModelOption, config: LLMConfig | None = None) -> dict:
    effective = (config or LLMConfig()).merged_over(LLMConfig(
        temperature=option.default_temperature, reasoning_effort=option.default_reasoning_effort,
        text_verbosity=option.default_text_verbosity, store=option.default_store,
        thinking_level=option.default_thinking_level,
    ))
    values = asdict(effective)
    if option.provider == PROVIDER_GEMINI and effective.thinking_level:
        values["thinking_level"] = clamp_thinking_level(option.model, effective.thinking_level)
    if option.supported_reasoning_efforts and effective.reasoning_effort not in option.supported_reasoning_efforts:
        values["reasoning_effort"] = option.default_reasoning_effort
    return {"model_key": option.key, "model_id": option.model, "provider": option.provider,
            "configuration": {key: value for key, value in values.items() if value is not None}}
