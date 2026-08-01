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

## Gemini Page Processor (`gemini_page_processor.py`)

The page-by-page Gemini PDF loop, shared by `AI_ocr_extraction/02` and
`AI_htr_extraction`. Handles splitting the PDF, the inline→Files-API fallback,
retrying only transient failures, interpreting `finish_reason`, joining pages
with `--- Page N ---` markers, and aborting the batch on quota exhaustion.

```python
from common.gemini_page_processor import GeminiPageProcessor, PagePolicy, process_pdf_batch

processor = GeminiPageProcessor(
    client, model_name, generation_config,
    PagePolicy(
        user_prompt="Transcribe this page.",
        media_resolution="ULTRA_HIGH",   # per-Part only; the config caps at HIGH
        on_blocked=my_recitation_fallback,  # optional
    ),
    rate_limiter=RateLimiter(rpm),
    console=console,
)
batch = process_pdf_batch(processor, pdf_files, output_dir, progress=progress)
```

Behaviour worth knowing:

- Inline requests are gated on `INLINE_REQUEST_LIMIT_BYTES`, not a hand-picked
  megabyte figure. Larger pages go straight to the Files API.
- `MAX_TOKENS` salvages the partial transcription and appends a truncation marker.
- `RECITATION` calls `PagePolicy.on_blocked` if set, otherwise skips the page.
- Failures are recorded in `PdfResult`, never written into the output text — an
  `[ERROR: ...]` placeholder in an archival transcript would end up in Omeka.
- No output file is written unless at least one page succeeded, so `PdfResult.ok`
  means "this run produced something", not "a file of some size exists".

---

## Omeka Text Updater (`omeka_text_updater.py`)

The `03` write step shared by `AI_summary`, `AI_ocr_extraction`,
`AI_ocr_correction` and `AI_audio_summary`.

```python
from common.omeka_text_updater import PropertyTarget, run_text_updates, updates_from_directory

target = PropertyTarget(
    term="bibo:shortDescription",
    property_id=summary_property_id,
    property_label="shortDescription",
    annotation_term="iwac:summaryModel",
    annotation_value=model_value,
)
stats = run_text_updates(
    client, updates_from_directory(Path("Summaries_FR_TXT")), target,
    console=console, dry_run=args.dry_run, require_confirmation=not args.yes,
)
```

Every pipeline using it gets: the full item fetched and PATCHed back (never a
trimmed payload), `@annotation` re-attached after `upsert_property_value`,
unchanged items skipped rather than re-PATCHed, and a `--dry-run` plus
confirmation gate. `updates_from_directory` reads `<item_id>.txt` files;
build `TextUpdate` objects yourself when items are matched some other way (the
transcription updater resolves `dcterms:identifier` first).

---

## Console Utilities (`console_utils.py`)

One definition of the rich furniture every pipeline prints.

```python
from common.console_utils import count_table, key_value_table, print_file_table, standard_progress

with standard_progress(console) as progress:
    task = progress.add_task("[cyan]Working...", total=len(items))
    ...
    progress.update(task, advance=1)

console.print(key_value_table([("Model", "gemini-flash"), ("Items", "42")]))
```

Rows whose value is `None` are skipped, so optional settings can be expressed
inline rather than guarded with an `if` around each `add_row`.

---

## Streaming Downloader (`downloader.py`)

`stream_download(url, path, timeout=...)` — writes to a `.part` temp file and
renames on success, checking `Content-Length` where the server provides it. Used
by `pdf_downloader.py` and `AI_audio_summary/01`. The temp file is the point:
these pipelines re-run against the same output directory, and a transfer
interrupted halfway must not be mistaken for a finished file next time.

---

# LLM Provider Configuration Guide

This guide explains how to use `llm_provider.py` to configure AI model behavior for different pipeline use cases.

## Overview

The `LLMConfig` class allows individual scripts to customize AI behavior without modifying the shared provider code. You can now configure:

- **OpenAI**: `reasoning_effort` and `text_verbosity`
- **Gemini Flash**: `thinking_level` ("minimal", "low", "medium", or "high")
- **Gemini Pro**: `thinking_level` ("low" or "high")
- **Mistral**: no per-script parameters
- **OpenRouter**: `reasoning_effort` on the models that accept one

`temperature` is deliberately absent from that list. See
[Temperature](#temperature-dont-set-it) below.

## Available Models

The provider supports these models via the `MODEL_REGISTRY`:

| Key | Provider | Model ID | Label | Description |
|-----|----------|----------|-------|-------------|
| `gpt-5.6-luna` | OpenAI | `gpt-5.6-luna` | ChatGPT (GPT-5.6 Luna) | Cost-optimized tier, $1/$6 per 1M tokens |
| `gpt-5.6-terra` | OpenAI | `gpt-5.6-terra` | ChatGPT (GPT-5.6 Terra) | Balanced tier, $2.50/$15 per 1M tokens |
| `gpt-5.6-sol` | OpenAI | `gpt-5.6-sol` | ChatGPT (GPT-5.6 Sol) | Flagship tier, $5/$30 per 1M tokens |
| `gemini-flash` | Gemini | `gemini-flash-latest` | Gemini Flash | Fast, cost-effective |
| `gemini-flash-lite` | Gemini | `gemini-flash-lite-latest` | Gemini Flash-Lite | Most cost-effective, lowest latency |
| `gemini-pro` | Gemini | `gemini-pro-latest` | Gemini Pro | Highest quality |
| `mistral-large` | Mistral | `mistral-large-2512` | Mistral Large 3 | 41B active params MoE |
| `ministral-14b` | Mistral | `ministral-14b-2512` | Ministral 3 14B | Fast, cost-effective |
| `qwen3.5-moe` | OpenRouter | `qwen/qwen3.5-122b-a10b` | Qwen3.5 122B-A10B | Apache-2.0 open weights, MoE 10B active, $0.26/$2.08 per 1M tokens |
| `qwen3.5-moe-small` | OpenRouter | `qwen/qwen3.5-35b-a3b` | Qwen3.5 35B-A3B | Apache-2.0 open weights, MoE 3B active, $0.14/$1.00 per 1M tokens |
| `qwen3.5-dense` | OpenRouter | `qwen/qwen3.5-27b` | Qwen3.5 27B | Apache-2.0 open weights, dense, $0.195/$1.56 per 1M tokens |
| `deepseek-v4-flash-0731` | OpenRouter | `deepseek/deepseek-v4-flash-0731` | DeepSeek V4 Flash 0731 | **Default text model**; official release, 284B/13B active, 1M context, from $0.09/$0.18 per 1M tokens |
| `deepseek-v4-flash` | OpenRouter | `deepseek/deepseek-v4-flash` | DeepSeek V4 Flash Preview | Superseded April preview, retained for reproducibility |
| `deepseek-v4-pro` | OpenRouter | `deepseek/deepseek-v4-pro` | DeepSeek V4 Pro | 1.6T/49B active MoE flagship, $0.435/$0.87 per 1M tokens |

### Model Aliases

For convenience, these aliases are also supported:

| Alias | Resolves To |
|-------|-------------|
| `openai` | `gpt-5.6-luna` |
| `luna` | `gpt-5.6-luna` |
| `terra` | `gpt-5.6-terra` |
| `sol` | `gpt-5.6-sol` |
| `gpt-5.6` | `gpt-5.6-sol` |
| `gemini` | `gemini-flash` |
| `mistral` | `mistral-large` |
| `ministral` | `ministral-14b` |
| `qwen` | `qwen3.5-moe` |
| `deepseek` | `deepseek-v4-flash-0731` |
| `deepseek-pro` | `deepseek-v4-pro` |

OpenRouter slugs resolve as-is too, so a model id copied off openrouter.ai
(`qwen/qwen3.5-122b-a10b`) works without translation.

The retired OpenAI keys still resolve: `gpt-5-mini` → `gpt-5.6-luna`, and
`gpt-5.1` / `gpt-5` → `gpt-5.6-sol`. Their underlying snapshots shut down on
2026-10-23, so prefer the new tier keys in new code.

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

### Never hand-build the JSON schema

Each provider gets the Pydantic class itself, not `model_json_schema()`:

| Provider | Call |
|---|---|
| OpenAI | `responses.parse(text_format=Model)` |
| Gemini | `GenerateContentConfig(response_schema=Model)` |
| Mistral | `chat.parse(response_format=Model)` |
| OpenRouter | `chat.completions.parse(response_format=Model)` |

This matters for OpenAI in particular. Its `strict` mode requires
`additionalProperties: false` on every object and *every* property listed in
`required`, and `model_json_schema()` emits neither — it drops any field that has
a default from `required`. Passing that raw schema with `strict: true` is rejected
by the API, and a caller's retry loop will report it as a generic failure.
`responses.parse()` runs the SDK's own `to_strict_json_schema()` transform, so the
schema is always valid.

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
| `reasoning_effort` | `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` | `"low"` | Controls reasoning depth and quality. `"none"` makes the model behave like a non-reasoning one — the cheapest option for mechanical work |
| `text_verbosity` | `"low"`, `"medium"`, `"high"` | `"low"` | Controls response length and detail |
| `store` | `True`, `False` | `False` | Whether OpenAI retains the request/response server-side. Off by default: these pipelines send full archival documents |

**Note**: OpenAI's Responses API ignores `temperature` - use `reasoning_effort` and `text_verbosity` instead.

### Temperature: don't set it

Every model's temperature is decided by its vendor and recorded once in
`MODEL_REGISTRY`. Pipelines should not pass one, because a pipeline picks a model
*tier* and cannot know which vendor's model the run will land on — and the right
value differs sharply between them:

| Model | Sent | Why |
|---|---|---|
| Gemini 3.x, Gemma 4 | *nothing at all* | Google: "we strongly recommend keeping the temperature parameter at its default value of `1.0`"; below 1.0 "may lead to unexpected behavior, such as looping or degraded performance" |
| DeepSeek V4 family | `1.0` | DeepSeek's 0731 card recommends `temperature = 1.0`, with `top_p = 1.0` outside agentic scenarios |
| Qwen3.5 | `0.7` | Qwen's published non-thinking recipe; Qwen warns near-greedy decoding causes "performance degradation and endless repetitions" |
| Mistral Large 3, Ministral 3 | `0.2` | The one vendor here recommending a low value — 0.05-0.20 for non-creative instruct work |
| GPT-5.6 (all tiers) | n/a | The Responses API ignores it |

Looping is the failure that motivates this. In these pipelines it shows up as a
transcript repeating a paragraph for the rest of a 90-minute interview, or OCR
stalling on one line until `max_output_tokens` — expensive, and silent until
someone reads the output. Note that "send nothing" and "send `1.0`" are different
requests; for Gemini the key is omitted entirely.

`top_p` and `top_k` are never set anywhere in this repo, which matches Google's
advice to remove them too. When output needs to be constrained, do it with
explicit rules in the system prompt or with a structured-output schema — not with
sampling parameters.

`LLMConfig(temperature=...)` still works and still overrides the default. It is
an escape hatch for a one-off experiment, not something to leave in a script.

### Gemini Parameters

Both Gemini 3 models use `thinking_level` to control how much reasoning the model does before answering. Thinking cannot be disabled — these models always reason to some degree.

#### Gemini Flash

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinking_level` | `str` | `"minimal"` | `"minimal"` = fastest, least reasoning<br>`"low"` / `"medium"` = balanced<br>`"high"` = deepest reasoning |

#### Gemini Pro

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinking_level` | `str` | `"low"` | `"low"` = faster, less reasoning<br>`"high"` = deeper reasoning, slower |

#### Model Comparison

| Model | Thinking Levels | Default | Best For |
|-------|----------------|---------|----------|
| Gemini Flash | `"minimal"`, `"low"`, `"medium"`, `"high"` | `"minimal"` | Fast processing, bulk tasks |
| Gemini Pro | `"low"`, `"high"` | `"low"` | Complex analysis, higher accuracy |

### Mistral Parameters

Mistral takes no per-script parameters; its `temperature` of `0.2` comes from
`MODEL_REGISTRY`.

**Available Mistral Models**:
- **`mistral-large`**: Mistral Large 3 — flagship 41B active params MoE model
- **`ministral-14b`**: Ministral 3 14B — fast, cost-effective ($0.2/M tokens)

**Note**: Both Mistral models support native structured outputs via `client.chat.parse()`.

### OpenRouter Parameters

OpenRouter is a router in front of open-weights models, not a lab. One
`OPENROUTER_API_KEY` covers all of them, and because the endpoint is
OpenAI-compatible it reuses the already-installed `openai` SDK — no extra
dependency.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_effort` | `str` | per model | Only sent to models that accept one; see below |

**Available OpenRouter models**:
- **`qwen3.5-moe` / `qwen3.5-moe-small` / `qwen3.5-dense`**: Qwen3.5 122B-A10B, 35B-A3B and 27B, all Apache-2.0 open weights. `qwen3.5-moe` was re-pointed from 35B-A3B to 122B-A10B on 2026-07-31 so the sentiment panel's open-weights members sit at comparable active-parameter counts (10B vs DeepSeek V4 Flash's 13B; 35B-A3B activates only 3B)
  (deliberately not the Flash/Plus/Max hosted tiers, which publish no weights). Accept
  `reasoning_effort` minimal/low/medium/high/xhigh, normalised by OpenRouter.
- **`deepseek-v4-flash-0731`**: official DeepSeek V4 Flash release and the shared text default; accepts exactly `"low"`, `"high"`, or `"max"` reasoning (bulk pipelines use low; sentiment uses high).
- **`deepseek-v4-flash`**: superseded April preview, retained only to reproduce earlier runs.
- **`deepseek-v4-pro`**: DeepSeek V4 Pro — quality tier, reasons at `"high"` by default; accepts `"high"` / `"xhigh"`.

Three behaviours are specific to this provider and worth knowing:

**Data collection is denied.** OpenRouter dispatches to third-party inference
backends and defaults to allowing ones that may retain or train on the payload.
These pipelines send whole archival documents, so every request carries
`provider: {"data_collection": "deny"}` — the same intent as `store=False` on
the OpenAI path. It is applied in `OPENROUTER_PROVIDER_PREFS`, not per call, so
no pipeline can forget it.

**`require_parameters` is on.** `json_schema` support varies by backend. Without
this flag a structured request can be routed to one that ignores
`response_format` and answers in prose.

**Reasoning effort is clamped, not forwarded.** `LLMConfig` is shared across
providers, so a pipeline tuned for OpenAI (NER asks for `"medium"`) reaches
these models too. An effort a model does not declare in
`supported_reasoning_efforts` is dropped in favour of the model's own default,
because with `require_parameters` on, forwarding it could leave the request
with no eligible backend.

**Structured output has a fallback.** Unlike first-party OpenAI, `message.parsed`
cannot be relied on: open models routinely return schema-valid JSON as a plain
string, sometimes inside a ``` fence. `OpenRouterClient.generate_structured()`
validates the raw content when `parsed` is empty, so a well-formed answer is not
thrown away over its packaging.

## Recommended Configurations by Use Case

### Named Entity Recognition (NER)
Complex analysis requiring careful reasoning and detailed output.

```python
config = LLMConfig(
    reasoning_effort="high",      # OpenAI: careful analysis
    text_verbosity="medium",       # OpenAI: detailed explanations
    thinking_level="high",         # Gemini: deep reasoning
)
```

### OCR Extraction/Correction
Fast processing with minimal reasoning needed.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick processing
    text_verbosity="low",          # OpenAI: concise output
    thinking_level="low",          # Gemini: minimal reasoning
)
```

### Document Summarization
Comprehensive analysis with moderate creativity.

```python
config = LLMConfig(
    reasoning_effort="medium",     # OpenAI: balanced reasoning
    text_verbosity="medium",       # OpenAI: detailed summaries
    thinking_level="medium",       # Gemini Flash: balanced thinking
)
```

### Text Classification
Simple categorization. Constrain the output with a structured-output schema or an
explicit instruction, not with `temperature=0.0` — near-greedy decoding is what
Google and Alibaba both warn causes looping.

```python
config = LLMConfig(
    reasoning_effort="low",        # OpenAI: quick classification
    text_verbosity="low",          # OpenAI: just the category
    thinking_level="minimal",      # Gemini Flash: least reasoning
)
```

### Translation
Moderate reasoning with low creativity.

```python
config = LLMConfig(
    reasoning_effort="medium",     # OpenAI: consider context
    text_verbosity="low",          # OpenAI: just the translation
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

Old scripts passing `temperature` still work, but both forms override the model's
vendor-recommended default, which is rarely what you want — see
[Temperature](#temperature-dont-set-it):

```python
# Old way (still supported)
llm_client = build_llm_client(model_option, temperature=0.5)

# Equivalent
config = LLMConfig(temperature=0.5)
llm_client = build_llm_client(model_option, config=config)

# What pipelines should do: say nothing, and inherit the vendor's value
llm_client = build_llm_client(model_option)
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
   - Gemini Flash: `"minimal"` for fast tasks, `"low"`/`"medium"` for balanced work, `"high"` for complex analysis
   - Gemini Pro: `"low"` for fast tasks, `"high"` for complex analysis
3. **Don't set temperature**: it belongs to the vendor, and lowering it is a
   documented cause of looping on Gemini 3 and Qwen — see
   [Temperature](#temperature-dont-set-it)
4. **Get consistency from the prompt and the schema**, not from sampling: explicit
   rules in the system instruction, plus `generate_structured()`
5. **Log your config**: Always log the configuration used for reproducibility
6. **Use structured outputs**: For NER, classification, and data extraction, prefer `generate_structured()` over parsing JSON manually

## Adding New Models

To add a new model to the registry:

1. Update `MODEL_REGISTRY` in `llm_provider.py`
2. Set appropriate defaults (e.g., `default_thinking_level` for Gemini 3 models).
   Look up the vendor's own sampling recommendation and record it as
   `default_temperature`, with a comment citing it — leave it unset to send no
   temperature at all. Don't copy a neighbouring entry's value.
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
A: Use registry keys like `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gemini-flash`, `gemini-pro`, `mistral-large`, `ministral-14b`. Common aliases like `openai`, `luna`, `terra`, `sol`, `gemini`, `mistral` also work, as do the retired `gpt-5-mini` / `gpt-5.1` keys.

**Q: How do I restrict which models a pipeline can use?**  
A: Use `allowed_keys` in `get_model_option()`:
```python
model_option = get_model_option(args.model, allowed_keys=["gemini-flash", "gemini-pro"])
```
