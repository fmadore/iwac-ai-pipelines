# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IWAC AI Pipelines is a collection of AI-powered document processing workflows for the Islam West Africa Collection (IWAC) — a digital archive of 14,500+ items. The project automates OCR extraction, text correction, summarization, named entity recognition, transcription, and handwritten text recognition using multiple LLM providers (Gemini, OpenAI, Mistral).

## Virtual Environment

**Always** use the project virtual environment at `.venv/` when running scripts or installing packages:

```bash
.venv\Scripts\activate   # Windows
# or
.venv/bin/activate       # Linux/macOS
```

When running commands via CLI, use `.venv\Scripts\python` (or `.venv/bin/python`) directly if the venv is not already activated.

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
python 01_NER_AI.py --item-set-id 123 --model gpt-5.6-luna --async --batch-size 20
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

model_option = get_model_option(args.model, allowed_keys=["gemini-flash", "gpt-5.6-luna"])
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

### Shared Retry Decorator

`common/retry.py` provides `retry_with_backoff` for wrapping functions with exponential backoff. Automatically skips retry on `QuotaExhaustedError`. An optional `is_retryable` predicate can re-raise non-retryable errors immediately.

### Shared Gemini Utilities

`common/gemini_utils.py` centralizes Gemini API plumbing for multimodal scripts:
- `build_generation_config(model_name, ...)` — builds a `GenerateContentConfig` with the correct thinking/temperature settings per model family
- `upload_and_wait_active(client, source, ...)` — uploads a file via the Files API and polls until it is ACTIVE
- `extract_text_from_response(response)` — safely extracts text, skipping `thought=True` parts returned by thinking models

### Shared Prompt Loader

`common/prompt_loader.py` provides prompt discovery and interactive selection (`discover_prompts`, `load_prompt_md`, `select_prompt_interactive`) for pipelines that keep multiple alternative `.md` prompts in a `prompts/` directory (audio transcription, video summaries). Prompt files are named `<number>_<description>.md`.

### Shared PDF Downloader

`common/pdf_downloader.py` is the shared Omeka PDF download step used by `AI_ocr_extraction/01` and `AI_summary_issue/01`: fetches items from an item set, resolves media PDF URLs, and downloads them with consistent naming and progress output.

### IWAC Instance Configuration

`common/iwac_config.py` holds constants specific to the IWAC Omeka S instance so they live in one place instead of being copy-pasted across scripts:
- Authority item sets: `SPATIAL_AUTHORITY_ITEM_SETS`, `SUBJECT_AUTHORITY_ITEM_SETS`, `TOPIC_AUTHORITY_ITEM_SETS`
- Property IDs: `DCTERMS_TITLE_PROPERTY_ID`, `DCTERMS_SUBJECT_PROPERTY_ID`, `DCTERMS_TYPE_PROPERTY_ID`, `DCTERMS_SPATIAL_PROPERTY_ID`, `BIBO_CONTENT_PROPERTY_ID`, etc.
- `AI_MODEL_ITEMS`, `select_model_key()` and `model_annotation_value()` — the authority items, interactive picker, and value object used to annotate which AI model produced content (`iwac:ocrModel` / `iwac:summaryModel`)
- `item_api_url(base_url, item_id)` — builds an item's API `@id` (never hardcode `https://islam.zmo.de/...` URLs)

### Other Shared Helpers

- `common/pdf_utils.py` — page-by-page PDF helpers (`extract_pdf_page`, `get_pdf_page_count`) used by OCR, HTR, and magazine extraction
- `common/ffmpeg_utils.py` — FFmpeg discovery, pydub setup, video-to-audio conversion, audio splitting, and cleanup for audio/video pipelines

### Multimodal Pipelines (Exception)

Audio, vision, HTR, and OCR scripts use provider clients directly because they require special capabilities not available through the shared provider. Pipelines with Omeka download/upload steps use `OmekaClient`; standalone processors (HTR, video) operate on local files only. All multimodal scripts use `RateLimiter` for quota handling.

### Model Registry

| Key | Provider | Use Case |
|-----|----------|----------|
| `gpt-5.6-luna` | OpenAI | GPT-5.6 Luna — cost-optimized tier, fast, high-volume text processing ($1/$6 per 1M tokens) |
| `gpt-5.6-terra` | OpenAI | GPT-5.6 Terra — balanced tier ($2.50/$15 per 1M tokens) |
| `gpt-5.6-sol` | OpenAI | GPT-5.6 Sol — flagship tier, highest quality, slower ($5/$30 per 1M tokens) |
| `gemini-flash` | Google | Gemini Flash — fast multimodal, cost-effective |
| `gemini-flash-lite` | Google | Gemini Flash-Lite — cheapest, lowest latency |
| `gemini-pro` | Google | Best quality (latest Pro), more expensive |
| `gemma-4` | Google | Gemma 4 31B open-weights, served via the Gemini API (shares `GEMINI_API_KEY`); text + image only, thinking "minimal"/"high" only |
| `mistral-large` | Mistral | Good quality, moderate cost |
| `ministral-14b` | Mistral | Budget option ($0.2/M tokens) |

Aliases: `openai` → `gpt-5.6-luna`, `luna`/`terra`/`sol` → the matching tier, `gpt-5.6` → `gpt-5.6-sol`, `gemini` → `gemini-flash`, `gemma` → `gemma-4`, `mistral` → `mistral-large`

The retired OpenAI keys still resolve — `gpt-5-mini` → `gpt-5.6-luna`, `gpt-5.1`/`gpt-5` → `gpt-5.6-sol` — but their snapshots shut down on 2026-10-23, so use the tier keys in new code.

### LLMConfig Parameters

- **OpenAI:** `reasoning_effort` (GPT-5.6 accepts "none"/"low"/"medium"/"high"/"xhigh"/"max"; API default "medium", this project defaults to "low"), `text_verbosity` ("low"/"medium"/"high")
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
- `AI_audio_summary/` — Audio/video transcription (Gemini multimodal or Mistral Voxtral)
- `AI_htr_extraction/` — Handwritten text recognition (Gemini vision, standalone processor)
- `AI_ocr_extraction/` — PDF OCR (Gemini native PDF or Mistral Document AI)
- `AI_video_summary/` — Video processing with visual descriptions (standalone processor)
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
- [ ] Use `OmekaClient.from_env()` for Omeka S API access (in download/upload steps; standalone processors like HTR and video skip this)
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

## Development

```bash
pip install -e ".[dev]"   # install with pytest + ruff
pytest tests/             # run the test suite
ruff check .              # lint
```
