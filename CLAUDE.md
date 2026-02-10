# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IWAC AI Pipelines is a collection of AI-powered document processing workflows for the Islam West Africa Collection (IWAC) — a digital archive of 14,500+ items. The project automates OCR extraction, text correction, summarization, named entity recognition, transcription, and handwritten text recognition using multiple LLM providers (Gemini, OpenAI, Mistral).

## Running Pipelines

Each AI pipeline follows a numbered script pattern. Run scripts sequentially within each directory:

```bash
cd AI_<pipeline_name>/
python 01_script.py    # Step 1 (typically: download from Omeka)
python 02_script.py    # Step 2 (typically: AI processing)
python 03_script.py    # Step 3 (typically: update Omeka)
```

Most scripts support both interactive mode (prompts guide selection) and CLI flags:
```bash
python 02_AI_generate_summaries.py --model gemini-flash
python 01_NER_AI.py --item-set-id 123 --model gpt-5-mini --async --batch-size 20
```

## Architecture

### Shared Omeka Client

**All** pipelines accessing Omeka S **must** use `common/omeka_client.py`:

```python
from common.omeka_client import OmekaClient

client = OmekaClient.from_env()
items = client.get_items(item_set_id=123)
item = client.get_item(456)
client.update_item(456, data)
item_set = client.get_item_set(789)
```

**Never** use raw `requests.get/patch` with Omeka credentials directly in pipeline scripts.

### Shared LLM Provider Pattern

Text-only pipelines **must** route through `common/llm_provider.py`:

```python
from common.llm_provider import build_llm_client, get_model_option, LLMConfig, summary_from_option

model_option = get_model_option(args.model, allowed_keys=["gemini-flash", "gpt-5-mini"])
config = LLMConfig(reasoning_effort="medium", thinking_level="low")
llm_client = build_llm_client(model_option, config)
response = llm_client.generate(system_prompt, user_prompt)
```

**Never** instantiate `openai.OpenAI()`, `google.genai.Client()`, or `mistralai.Mistral()` directly in text scripts.

### Shared Rate Limiter

Multimodal pipelines **must** use `common/rate_limiter.py` for quota-aware error handling:

```python
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted

# In __init__: create rate limiter (None = no throttling, set RPM for free tier)
self.rate_limiter = RateLimiter(requests_per_minute=None)

# Before each API call:
self.rate_limiter.wait()
response = client.models.generate_content(...)

# In error handlers:
except APIError as e:
    if is_quota_exhausted(e):
        raise QuotaExhaustedError(str(e))  # stops pipeline immediately
```

- `is_quota_exhausted()` distinguishes daily quota exhaustion (stop) from transient rate limits (retry)
- `QuotaExhaustedError` propagates up to save partial results then halt
- `RateLimiter` optionally throttles requests to stay under RPM limits

### Multimodal Pipelines (Exception)

Audio, vision, HTR, and OCR scripts use provider clients directly because they require special capabilities not available through the shared provider. They still use `OmekaClient` for all Omeka S API access and `RateLimiter` for quota handling.

### Model Registry

| Key | Provider | Use Case |
|-----|----------|----------|
| `gpt-5-mini` | OpenAI | Fast, cost-effective text processing |
| `gpt-5.1` | OpenAI | Higher quality, slower |
| `gemini-flash` | Google | Fast multimodal, cost-effective |
| `gemini-pro` | Google | Best quality, more expensive |
| `mistral-large` | Mistral | Good quality, moderate cost |
| `ministral-14b` | Mistral | Budget option ($0.2/M tokens) |

Aliases: `openai` → `gpt-5-mini`, `gemini` → `gemini-flash`, `mistral` → `mistral-large`

### LLMConfig Parameters

- **OpenAI:** `reasoning_effort` ("low"/"medium"/"high"), `text_verbosity` ("low"/"medium"/"high")
- **Gemini 3:** `thinking_level` — Flash: "minimal"/"low"/"medium"/"high"; Pro: "low"/"high" — cannot be disabled
- **Mistral:** `temperature` (0.0-1.0)

## Pipeline Categories

### Text-Only Pipelines (use shared `llm_provider.py` + `omeka_client.py`)
- `AI_summary/` — French document summarization
- `AI_NER/` — Named entity recognition with authority reconciliation
- `AI_ocr_correction/` — OCR error correction (plain text or ALTO XML)
- `AI_sentiment_analysis/` — Sentiment analysis with all 3 providers concurrently
- `NotebookLM/` — Export to Google NotebookLM (OmekaClient only, no LLM)

### Multimodal Pipelines (use provider APIs directly)
- `AI_audio_summary/` — Audio/video transcription (Gemini multimodal)
- `AI_htr_extraction/` — Handwritten text recognition (Gemini vision)
- `AI_ocr_extraction/` — PDF OCR (Gemini native PDF or Mistral Document AI)
- `AI_video_summary/` — Video processing with visual descriptions
- `AI_summary_issue/` — Magazine article extraction (Gemini/Mistral structured outputs)

## Environment Configuration

Required in `.env`:
```
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# At least one AI provider
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

## Code Conventions

### Console Output
All pipelines use `rich` library for consistent terminal UI:
- Welcome panels, configuration tables, progress bars
- Color-coded: success `[green]✓[/]`, errors `[red]✗[/]`, info `[cyan]...[/]`

### Prompt Templates
Keep prompts in pipeline directories as `.md` files loaded at runtime.

### Structured Outputs
Use `generate_structured()` with Pydantic models for data extraction:
```python
class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")

response = llm_client.generate_structured(system_prompt, user_prompt, response_schema=Entity)
```

### Gemini API (Multimodal Scripts)
- Use `system_instruction` in `GenerateContentConfig` (not concatenated to user prompt)
- Use `types.Part.from_bytes()` for PDF processing
- Pass Pydantic `BaseModel` directly to `response_schema`

## Checklists for New Scripts

### Text-Only Scripts
- [ ] Use `OmekaClient.from_env()` for all Omeka S API access
- [ ] Import from `common.llm_provider`: `build_llm_client`, `get_model_option`, `LLMConfig`, `summary_from_option`
- [ ] Provide `--model` flag with `choices=[...]` or use `allowed_keys`
- [ ] Create task-appropriate `LLMConfig`
- [ ] Log `summary_from_option(model_option)` and config
- [ ] Load prompts from sibling `.md` files
- [ ] Skip empty text before LLM call
- [ ] Use `rich` library for output

### Multimodal Scripts
- [ ] Use `OmekaClient.from_env()` for all Omeka S API access
- [ ] Use appropriate provider client directly for AI processing
- [ ] Use `system_instruction` in `GenerateContentConfig` (Gemini)
- [ ] Use `RateLimiter` from `common.rate_limiter` with `wait()` before each API call
- [ ] Detect quota exhaustion with `is_quota_exhausted()` and raise `QuotaExhaustedError`
- [ ] Catch `QuotaExhaustedError` in processing loops to save partial results and stop
- [ ] Handle transient API errors with retry logic (exponential backoff + jitter)
- [ ] Log selected model before processing
- [ ] Use `rich` library for output

## Adding New Models

1. Add to `MODEL_REGISTRY` and `MODEL_ALIASES` in `common/llm_provider.py`
2. Set appropriate defaults for reasoning, verbosity, thinking
3. Update README files and pipeline `--model` choices
4. No duplication of provider-specific code elsewhere
