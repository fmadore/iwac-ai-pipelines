# LLM Provider Configuration Guide

This guide explains how to use the enhanced `llm_provider.py` to configure AI model behavior for different pipeline use cases.

## Overview

The `LLMConfig` class allows individual scripts to customize AI behavior without modifying the shared provider code. You can now configure:

- **OpenAI**: `reasoning_effort` and `text_verbosity`
- **Gemini**: `temperature`, `thinking_mode`, and `thinking_budget`

## Quick Start

```python
from common.llm_provider import build_llm_client, get_model_option, LLMConfig

# Get model selection
model_option = get_model_option("openai")  # or use --model flag

# Configure for your use case
config = LLMConfig(
    reasoning_effort="high",
    text_verbosity="medium"
)

# Build client with config
llm_client = build_llm_client(model_option, config=config)

# Generate content
response = llm_client.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Extract named entities from this text."
)
```

## Configuration Parameters

### OpenAI Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `reasoning_effort` | `"low"`, `"medium"`, `"high"` | `"low"` | Controls reasoning depth and quality |
| `text_verbosity` | `"low"`, `"medium"`, `"high"` | `"low"` | Controls response length and detail |

**Note**: OpenAI's Responses API ignores `temperature` - use `reasoning_effort` and `text_verbosity` instead.

### Gemini Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` (0.0-1.0) | `0.2` | Controls randomness in responses |
| `thinking_mode` | `bool` | `False` (Flash)<br>`True` (Pro) | Enable extended thinking |
| `thinking_budget` | `int` or `None` | `None` | `0` = disabled (Flash only)<br>`None` = model default<br>`>0` = custom budget |

**Important**: 
- **Gemini Pro** cannot disable thinking (has minimum budget requirement)
- **Gemini Flash** can set `thinking_budget=0` for faster responses

## Recommended Configurations by Use Case

### Named Entity Recognition (NER)
Complex analysis requiring careful reasoning and detailed output.

```python
config = LLMConfig(
    reasoning_effort="high",      # OpenAI: careful analysis
    text_verbosity="medium",       # OpenAI: detailed explanations
    thinking_budget=1000,          # Gemini: extended thinking
    temperature=0.2                # Gemini: consistent results
)
```

### OCR Extraction/Correction
Fast processing with minimal thinking needed.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick processing
    text_verbosity="low",          # OpenAI: concise output
    thinking_budget=0,             # Gemini Flash: disable thinking for speed
    temperature=0.1                # Gemini: very consistent
)
```

### Document Summarization
Comprehensive analysis with moderate creativity.

```python
config = LLMConfig(
    reasoning_effort="medium",     # OpenAI: balanced reasoning
    text_verbosity="medium",       # OpenAI: detailed summaries
    thinking_mode=True,            # Gemini: enable thinking
    temperature=0.3                # Gemini: some creativity
)
```

### Text Classification
Simple categorization with deterministic output.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick classification
    text_verbosity="low",          # OpenAI: just the category
    thinking_budget=0,             # Gemini Flash: no thinking needed
    temperature=0.0                # Gemini: completely deterministic
)
```

### Translation
Moderate reasoning with low creativity.

```python
config = LLMConfig(
    reasoning_effort="medium",     # OpenAI: consider context
    text_verbosity="low",          # OpenAI: just the translation
    temperature=0.2                # Gemini: consistent translations
)
```

## Per-Request Configuration Override

You can override the client's default config for specific requests:

```python
# Client with default low settings
default_config = LLMConfig(reasoning_effort="low", text_verbosity="low")
llm_client = build_llm_client(model_option, config=default_config)

# Most requests use default
response = llm_client.generate(system_prompt, "Simple question")

# Override for complex requests
complex_config = LLMConfig(reasoning_effort="high", text_verbosity="high")
response = llm_client.generate(
    system_prompt, 
    "Complex analysis needed",
    config=complex_config  # Override just for this request
)
```

## Backward Compatibility

Old scripts using the `temperature` parameter still work:

```python
# Old way (still supported)
llm_client = build_llm_client(model_option, temperature=0.5)

# New way (recommended)
config = LLMConfig(temperature=0.5)
llm_client = build_llm_client(model_option, config=config)
```

## Implementation Example

Here's a complete example for an NER pipeline:

```python
import argparse
from common.llm_provider import (
    build_llm_client,
    get_model_option,
    LLMConfig,
    summary_from_option,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model key (e.g., openai, gemini-flash)")
    args = parser.parse_args()
    
    # Get model selection
    model_option = get_model_option(args.model)
    print(f"Using {summary_from_option(model_option)}")
    
    # Configure for NER use case
    config = LLMConfig(
        reasoning_effort="high",
        text_verbosity="medium",
        thinking_budget=1000,
        temperature=0.2
    )
    
    # Build client
    llm_client = build_llm_client(model_option, config=config)
    
    # Load prompts
    with open("ner_system_prompt.md") as f:
        system_prompt = f.read()
    
    # Process items
    for item in items:
        if not item.text.strip():
            continue
            
        response = llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=f"Extract entities from: {item.text}"
        )
        
        # Process response...

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Choose the right effort level**: Don't use `high` reasoning for simple tasks
2. **Disable thinking for speed**: Use `thinking_budget=0` (Flash only) for fast, simple tasks
3. **Use low temperature for consistency**: OCR, classification, extraction benefit from `temperature=0.0-0.2`
4. **Use higher temperature for creativity**: Summaries and generation can use `temperature=0.3-0.7`
5. **Respect model constraints**: Remember Pro cannot disable thinking
6. **Log your config**: Always log the configuration used for reproducibility

## Adding New Models

To add a new model to the registry:

1. Update `MODEL_REGISTRY` in `llm_provider.py`
2. Set appropriate defaults for `default_thinking_mode` and `default_thinking_budget`
3. Add aliases if needed in `MODEL_ALIASES`
4. Update this README with model-specific guidance

## Troubleshooting

**Q: Why is Gemini Pro ignoring `thinking_budget=0`?**  
A: Pro requires thinking and has a minimum budget. The provider will warn you and use the model's default.

**Q: Why isn't OpenAI using my `temperature` setting?**  
A: OpenAI's Responses API uses fixed configuration. Use `reasoning_effort` and `text_verbosity` instead.

**Q: How do I know which settings were actually used?**  
A: Enable debug logging: `logging.basicConfig(level=logging.DEBUG)` to see the actual parameters sent to each provider.
