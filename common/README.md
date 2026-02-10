# Shared Utilities

This directory contains shared modules used across all pipelines.

## Omeka S Client (`omeka_client.py`)

Authenticated client for the Omeka S REST API with automatic retry and pagination.

### Quick Start

```python
from common.omeka_client import OmekaClient

client = OmekaClient.from_env()  # Reads OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
items = client.get_items(item_set_id=123)
item = client.get_item(456)
client.update_item(456, item_data)
```

### API Reference

| Method | Description |
|--------|-------------|
| `OmekaClient.from_env()` | Create client from `.env` variables |
| `get_items(item_set_id)` | Fetch all items in a set (handles pagination) |
| `get_item(item_id)` | Fetch a single item by ID |
| `get_item_set(item_set_id)` | Fetch item set metadata |
| `update_item(item_id, data)` | PATCH an item (returns `True`/`False`) |
| `get_resource(url)` | GET any Omeka S resource URL |

The client includes:
- **Automatic retry** on transient errors (429, 500-504) with exponential backoff
- **Automatic pagination** — `get_items()` fetches all pages transparently
- **Environment-based configuration** via `python-dotenv`

### Environment Variables

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential
```

---

## FFmpeg Utilities (`ffmpeg_utils.py`)

Shared FFmpeg discovery, pydub configuration, video/audio format constants, conversion, splitting, and cleanup helpers for multimodal pipelines.

### Quick Start

```python
from common.ffmpeg_utils import (
    AUDIO_FORMATS, VIDEO_FORMATS,
    get_ffmpeg_paths, setup_pydub, is_video_file, get_mime_type,
    convert_video_to_audio, split_audio, cleanup_files,
)

# Discover ffmpeg/ffprobe (cached after first call)
paths = get_ffmpeg_paths()  # FFmpegPaths(ffmpeg, ffprobe) or None

# Configure pydub to use discovered paths
if setup_pydub():
    segments = split_audio(audio_path, output_dir, segment_minutes=10)

# Convert video to audio
audio = convert_video_to_audio(video_path, output_dir)

# Unified MIME type lookup (audio + video + mimetypes fallback)
mime = get_mime_type(Path("file.mp3"))  # "audio/mpeg"
```

### API Reference

| Function / Constant | Description |
|---------------------|-------------|
| `AUDIO_FORMATS` | `dict[str, str]` — 8 audio extension-to-MIME mappings |
| `VIDEO_FORMATS` | `dict[str, str]` — 12 video extension-to-MIME mappings |
| `AUDIO_EXPORT_FORMAT_MAP` | `dict[str, str]` — pydub export format names |
| `get_ffmpeg_paths()` | Discover ffmpeg/ffprobe; returns `FFmpegPaths` namedtuple or `None` |
| `setup_pydub()` | Import pydub and configure it with discovered paths; returns `bool` |
| `is_video_file(path)` | Check extension against `VIDEO_FORMATS` + mimetypes fallback |
| `get_mime_type(path)` | Lookup in `AUDIO_FORMATS` + `VIDEO_FORMATS` + mimetypes fallback |
| `convert_video_to_audio(video, out_dir)` | ffmpeg subprocess call; returns output `Path` or `None` |
| `split_audio(audio, out_dir, minutes)` | pydub-based splitting; returns `[audio]` on failure |
| `cleanup_files(paths, remove_parents)` | Delete files and optionally empty parent directories |

---

## Rate Limiter (`rate_limiter.py`)

Shared rate-limiting and quota-exhaustion utilities for Gemini API pipelines. Prevents wasting time retrying when daily quota is exhausted, and optionally throttles requests to stay under RPM limits.

### Quick Start

```python
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted

# Proactive throttling (e.g. free tier: 5 RPM)
limiter = RateLimiter(requests_per_minute=5)
limiter.wait()  # call before each API request

# Quota detection in error handlers
try:
    response = client.models.generate_content(...)
except APIError as e:
    if is_quota_exhausted(e):
        raise QuotaExhaustedError(str(e))  # stops pipeline immediately
```

### API Reference

| Component | Description |
|-----------|-------------|
| `QuotaExhaustedError` | Exception signaling daily/billing quota is hit — pipeline should stop |
| `is_quota_exhausted(error)` | Returns `True` for daily quota exhaustion (429 + quota indicators), `False` for transient rate limits |
| `RateLimiter(rpm, logger)` | Proactive throttler; `wait()` sleeps to space requests at `60/rpm` second intervals |

### Quota Detection

The `is_quota_exhausted()` function distinguishes between:
- **Transient rate limits** (per-minute) → worth retrying after a short delay
- **Quota exhaustion** (per-day, billing) → stop immediately, retrying is pointless

Detection is based on HTTP 429 status + message patterns like `"exceeded your current quota"`, `"requests_per_model_per_day"`, or status `RESOURCE_EXHAUSTED`.

---

## Retry Decorator (`retry.py`)

Exponential backoff decorator with jitter and quota-aware passthrough.

```python
from common.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=2.0)
def call_api():
    ...
```

Features:
- Exponential backoff with random jitter (prevents synchronized retries)
- `QuotaExhaustedError` is always re-raised immediately (never retried)

---

# LLM Provider Configuration Guide

This guide explains how to use `llm_provider.py` to configure AI model behavior for different pipeline use cases.

## Overview

The `LLMConfig` class allows individual scripts to customize AI behavior without modifying the shared provider code. You can now configure:

- **OpenAI**: `reasoning_effort` and `text_verbosity`
- **Gemini 3 Flash**: `temperature` and `thinking_level` ("minimal", "low", "medium", or "high")
- **Gemini 3 Pro**: `temperature` and `thinking_level` ("low" or "high")
- **Mistral**: `temperature`

## Available Models

The provider supports these models via the `MODEL_REGISTRY`:

| Key | Provider | Model ID | Label | Description |
|-----|----------|----------|-------|-------------|
| `gpt-5-mini` | OpenAI | `gpt-5-mini` | ChatGPT (GPT-5 mini) | Cost-optimized, fast |
| `gpt-5.1` | OpenAI | `gpt-5.1` | ChatGPT (GPT-5.1 full) | Flagship model |
| `gemini-flash` | Gemini | `gemini-3-flash-preview` | Gemini 3 Flash | Fast, cost-effective |
| `gemini-pro` | Gemini | `gemini-3-pro-preview` | Gemini 3.0 Pro | Highest quality |
| `mistral-large` | Mistral | `mistral-large-2512` | Mistral Large 3 | 41B active params MoE |
| `ministral-14b` | Mistral | `ministral-14b-2512` | Ministral 3 14B | Fast, cost-effective |

### Model Aliases

For convenience, these aliases are also supported:

| Alias | Resolves To |
|-------|-------------|
| `openai` | `gpt-5-mini` |
| `gemini` | `gemini-flash` |
| `mistral` | `mistral-large` |
| `ministral` | `ministral-14b` |

See `MODEL_ALIASES` in `llm_provider.py` for the full list of legacy aliases.

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

## Structured Outputs

The provider supports **native structured outputs** for OpenAI, Gemini, and Mistral APIs. This guarantees valid JSON responses matching your schema - no manual JSON parsing needed!

### Using Structured Outputs

```python
from pydantic import BaseModel, Field
from typing import List
from common.llm_provider import build_llm_client, get_model_option

# Define your output schema with Pydantic
class NERResult(BaseModel):
    persons: List[str] = Field(description="List of person names")
    organizations: List[str] = Field(description="List of organization names")
    locations: List[str] = Field(description="List of place names")
    subjects: List[str] = Field(description="List of topic keywords")

# Build client
model_option = get_model_option("openai")
llm_client = build_llm_client(model_option)

# Generate with guaranteed structure
result = llm_client.generate_structured(
    system_prompt="Extract named entities from the text.",
    user_prompt="Paris is the capital of France. Emmanuel Macron is the president.",
    response_schema=NERResult
)

# Access typed results directly - no parsing needed!
print(result.persons)       # ['Emmanuel Macron']
print(result.locations)     # ['Paris', 'France']
```

### Benefits of Structured Outputs

1. **Guaranteed valid JSON**: The API enforces your schema at generation time
2. **No parsing errors**: Eliminates regex extraction and `json.loads()` failures
3. **Type safety**: Pydantic validates and types your data automatically
4. **Better prompts**: Schema descriptions guide the model's output
5. **Cleaner code**: Remove boilerplate JSON extraction and error handling

### When to Use Structured vs. Text Output

| Use Case | Method | Why |
|----------|--------|-----|
| NER extraction | `generate_structured()` | Need consistent JSON structure |
| Data extraction | `generate_structured()` | Parsing specific fields |
| Classification | `generate_structured()` | Enum values, confidence scores |
| Summaries | `generate()` | Free-form text output |
| Translation | `generate()` | Just need the translated text |
| Creative writing | `generate()` | Open-ended generation |

## Configuration Parameters

### OpenAI Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `reasoning_effort` | `"low"`, `"medium"`, `"high"` | `"low"` | Controls reasoning depth and quality |
| `text_verbosity` | `"low"`, `"medium"`, `"high"` | `"low"` | Controls response length and detail |

**Note**: OpenAI's Responses API ignores `temperature` - use `reasoning_effort` and `text_verbosity` instead.

### Gemini Parameters

Both Gemini 3 models use `thinking_level` to control how much reasoning the model does before answering. Thinking cannot be disabled — these models always reason to some degree.

#### Gemini 3 Flash

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` (0.0-1.0) | `0.2` | Controls randomness (lower = more consistent) |
| `thinking_level` | `str` | `"minimal"` | `"minimal"` = fastest, least reasoning<br>`"low"` / `"medium"` = balanced<br>`"high"` = deepest reasoning |

#### Gemini 3 Pro

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` (0.0-1.0) | `0.2` | Controls randomness (lower = more consistent) |
| `thinking_level` | `str` | `"low"` | `"low"` = faster, less reasoning<br>`"high"` = deeper reasoning, slower |

#### Model Comparison

| Model | Thinking Levels | Default | Best For |
|-------|----------------|---------|----------|
| Gemini 3 Flash | `"minimal"`, `"low"`, `"medium"`, `"high"` | `"minimal"` | Fast processing, bulk tasks |
| Gemini 3 Pro | `"low"`, `"high"` | `"low"` | Complex analysis, higher accuracy |

### Mistral Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` (0.0-1.0) | `0.2` | Controls randomness in responses |

**Available Mistral Models**:
- **`mistral-large`**: Mistral Large 3 — flagship 41B active params MoE model
- **`ministral-14b`**: Ministral 3 14B — fast, cost-effective ($0.2/M tokens)

**Note**: Both Mistral models support native structured outputs via `client.chat.parse()`.

## Recommended Configurations by Use Case

### Named Entity Recognition (NER)
Complex analysis requiring careful reasoning and detailed output.

```python
config = LLMConfig(
    reasoning_effort="high",      # OpenAI: careful analysis
    text_verbosity="medium",       # OpenAI: detailed explanations
    thinking_level="high",         # Gemini: deep reasoning
    temperature=0.2                # Gemini: consistent results
)
```

### OCR Extraction/Correction
Fast processing with minimal reasoning needed.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick processing
    text_verbosity="low",          # OpenAI: concise output
    thinking_level="low",          # Gemini: minimal reasoning
    temperature=0.1                # Gemini: very consistent
)
```

### Document Summarization
Comprehensive analysis with moderate creativity.

```python
config = LLMConfig(
    reasoning_effort="medium",     # OpenAI: balanced reasoning
    text_verbosity="medium",       # OpenAI: detailed summaries
    thinking_level="medium",       # Gemini Flash: balanced thinking
    temperature=0.3                # Gemini: some creativity
)
```

### Text Classification
Simple categorization with deterministic output.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick classification
    text_verbosity="low",          # OpenAI: just the category
    thinking_level="minimal",      # Gemini Flash: least reasoning
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
        thinking_level="high",
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

1. **Choose the right effort level**: Don't use `"high"` reasoning for simple tasks — it's slower and more expensive
2. **Match thinking to model**:
   - Gemini 3 Flash: `"minimal"` for fast tasks, `"low"`/`"medium"` for balanced work, `"high"` for complex analysis
   - Gemini 3 Pro: `"low"` for fast tasks, `"high"` for complex analysis
3. **Use low temperature for consistency**: OCR, classification, and extraction benefit from `temperature=0.0-0.2`
4. **Use higher temperature for creativity**: Summaries and generation can use `temperature=0.3-0.7`
5. **Log your config**: Always log the configuration used for reproducibility
6. **Use structured outputs**: For NER, classification, and data extraction, prefer `generate_structured()` over parsing JSON manually

## Adding New Models

To add a new model to the registry:

1. Update `MODEL_REGISTRY` in `llm_provider.py`
2. Set appropriate defaults (e.g., `default_thinking_level` for Gemini 3 models)
3. Add aliases if needed in `MODEL_ALIASES`
4. Update this README with model-specific guidance

## Troubleshooting

**Q: Why is Gemini ignoring my `thinking_budget` setting?**
A: Gemini 3 models use `thinking_level` (e.g., `"low"`, `"high"`), not the older `thinking_budget` parameter. The provider handles this automatically.

**Q: Can I disable thinking for Gemini?**
A: No. Gemini 3 models always reason to some degree. Use `thinking_level="minimal"` (Flash) or `"low"` (Pro) for the fastest responses.

**Q: Why isn't OpenAI using my `temperature` setting?**  
A: OpenAI's Responses API uses fixed configuration. Use `reasoning_effort` and `text_verbosity` instead.

**Q: How do I know which settings were actually used?**  
A: Enable debug logging: `logging.basicConfig(level=logging.DEBUG)` to see the actual parameters sent to each provider.

**Q: What model keys can I use with `--model`?**  
A: Use registry keys like `gpt-5-mini`, `gpt-5.1`, `gemini-flash`, `gemini-pro`, `mistral-large`, `ministral-14b`. Common aliases like `openai`, `gemini`, `mistral` also work.

**Q: How do I restrict which models a pipeline can use?**  
A: Use `allowed_keys` in `get_model_option()`:
```python
model_option = get_model_option(args.model, allowed_keys=["gemini-flash", "gemini-pro"])
```
