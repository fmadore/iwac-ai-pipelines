# Shared Utilities

This directory contains shared modules used across all pipelines.

## Omeka S Client (`omeka_client.py`)

Authenticated client for the Omeka S REST API with automatic retry and pagination.

### Quick Start

```python
from common.omeka_client import OmekaClient

client = OmekaClient.from_env()  # Reads OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
items = client.get_items(item_set_id=123)
articles = client.get_items(resource_class_id=36, modified_after="2026-08-01")
total = client.count_items(resource_class_id=36)
item = client.get_item(456)
client.update_item(456, item_data)
```

### API Reference

| Method | Description |
|--------|-------------|
| `OmekaClient.from_env()` | Create client from `.env` variables |
| `get_items(item_set_id=None, **filters)` | Every item matching any `/api/items` filter (`item_set_id`, `resource_class_id`, `resource_template_id`, `modified_after`, `property[0][…]` …); follows `Omeka-S-Total-Results` and warns if the count moves during the walk |
| `iter_items(...)` | The same, streamed a page at a time — for a corpus whose `bibo:content` is too much to hold |
| `count_items(**filters)` | The match count from the header, one request |
| `list_page(page, per_page, **filters)` | One page, for samplers that pick pages at random |
| `get_items_by_ids(ids)` | Many items in pages of `id[]`, as `{id: item}` — a write step pre-fetches this way instead of one GET per item |
| `search_items_by_property(property_id, value)` | Items with an exact value; raises `OmekaRequestError` on a transport failure rather than answering "no match" |
| `get_item(item_id)` | Fetch a single item by ID |
| `get_item_set(item_set_id)` | Fetch item set metadata |
| `get_property_id(term)` | Resolve a vocabulary term to its property id at runtime |
| `update_item(item_id, data)` | PATCH an item (returns `True`/`False`); the payload must be the whole item |
| `create_item(data)` | POST a new item |
| `get_resource(url)` | GET any Omeka S resource URL |
| `append_resource_links(...)` | Add `resource:item` links to a payload, skipping ids already present |
| `upsert_property_value(...)` | **Deprecated** — drops `@annotation`, ignores `@language`; use `omeka_text_updater.apply_text_value` |

The client includes:
- **Automatic retry** on transient errors (429, 500-504) with `backoff_factor=1`
  (0, 2, 4, 8, 16 s) and `Retry-After` honoured — Omeka core never rate-limits,
  so this is for an overloaded PHP host, not for quota
- **Deterministic pagination** — Omeka sorts by id, and the walk stops at the
  announced total rather than on the first short page
- **`server_version`** — the `Omeka-S-Version` header from the last response,
  for run provenance
- **Environment-based configuration** via `python-dotenv`
- Credentials go in the query string (Omeka offers no header alternative), and
  `requests` echoes the URL in error messages: every entry point calls
  `common.log_redaction.install_credential_redaction()` so logs stay masked

### Environment Variables

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential
```

---

## Link Update and Reconciliation Runs (`link_update_cli.py`, `reconciliation_cli.py`)

NER and reference indexing end the same way: reconcile `Spatial AI` / `Subject
AI` terms against the authority records, then append the reconciled ids to
Omeka as `dcterms:spatial` / `dcterms:subject` links. Both runs live here once;
`AI_NER/02`, `AI_NER/03`, `AI_reference_indexing/03` and `05` are entry points
that name their folder, banner and file tags.

- `reconciliation_cli.run_reconciliation(client, csv, subject_tag=...)` —
  spatial pass, subject + topic pass (built together so a term in both sets is
  reported as ambiguous), fuzzy candidates for the rest.
- `link_update_cli.run_link_update(args, output_dir=..., banner=..., backup_label=...)`
  — model provenance from the checkpoint (`--model` overrides, prompt otherwise),
  batch pre-fetch of every item named by the CSV, one `iwac:nerModel`
  annotation per link added, `WriteGuard` gate and pre-write dump.

---

## Omeka Resource-Link Updater (`omeka_link_updater.py`)

Shared idempotent fetch/mutate/PATCH transaction for pipelines that append
`resource:item` values such as `dcterms:subject` and `dcterms:spatial`. It
deduplicates existing links and reports `updated`, `would_update`, `unchanged`,
`not_found`, `failed`, or `invalid_id`, so a failed PATCH never inflates
persisted-link counts and a dry run reports the totals a live run would produce.

`on_pre_write` fires once per item that is *about to change*, with the item
exactly as fetched. That snapshot is the only pre-write state that exists.

```python
from common.omeka_link_updater import ResourceLinkSpec, update_item_resource_links

pre_write = []
result = update_item_resource_links(client, item_id, [
    ResourceLinkSpec("dcterms:subject", subject_property_id, subject_ids, "Subject"),
], dry_run=guard.dry_run, on_pre_write=pre_write.append)
```

---

## Write Guard (`write_guard.py`)

The gate every Omeka write entry point must pass through: `--dry-run`, `--yes`,
`--backup-dir`, `--no-backup`, a blast-radius panel, and a pre-write payload dump.

This exists because of a real incident. On 2026-08-02 `AI_NER/03_Omeka_update.py`
had no argument parser, so a `--help` invocation was not recognised as a request
for help — it fell straight through to the real update and PATCHed 630 live items
before it was killed. **Ignoring argv is the dangerous part.** A write script must
refuse an argument it does not understand rather than treat it as consent.

```python
parser = argparse.ArgumentParser(description="...")
add_write_guard_args(parser, default_backup_dir=OUTPUT_DIR)
args = parser.parse_args()
guard = WriteGuard.from_args(args, default_backup_dir=OUTPUT_DIR)

if not guard.confirm(console, action="Append subject links",
                     base_url=client.base_url, item_count=len(rows)):
    return 1
...
guard.dump_backup(pre_write, label="ner_links")
```

A closed or non-interactive stdin counts as declining: an unattended run has to
pass `--yes` on purpose rather than inherit consent from an EOF.

---

## Durable Checkpoints (`checkpoint.py`)

`JsonCheckpoint` records both completed entries and an exact provenance context
(model ID, prompt hash, input scope). A resume refuses incompatible context
instead of silently mixing two runs. `atomic_write_text()` writes through a
temporary file and `os.replace`, so interrupted writes do not leave a partial
artifact at the final path.

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
| `probe_duration_seconds(path)` | Media duration via ffprobe; `None` (never raises) when unavailable |
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

Detection is based on the HTTP status (429, or 402 for a billing stop) plus message patterns that name a *daily* or *billing* quota — `"exceeded your current quota"`, `"requests_per_model_per_day"` and the like. The bare `RESOURCE_EXHAUSTED` status is deliberately **not** enough: Gemini uses it for per-minute throttling too, and a throttle is not an exhausted quota. A 429 that carries a `retryDelay` is treated as transient. `is_mistral_quota_exhausted()` applies the same idea to Mistral's error shape.

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
`AI_ocr_correction`, `AI_audio_summary`, `AI_youtube_transcription`,
`AI_publication_extraction` and `AI_summary_issue`. Every one of them takes the
same `--dry-run` / `--yes` / `--backup-dir` / `--no-backup` flags from
`common/write_guard.add_write_guard_args()`.

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
trimmed payload), `@annotation` attached to the value just written, unchanged
items skipped rather than re-PATCHed, and a `--dry-run` plus confirmation gate.
`updates_from_directory` reads `<item_id>.txt` files (`texts_from_directory`
returns the same as an `{item_id: text}` map); build `TextUpdate` objects
yourself when items are matched some other way (the transcription updater
resolves `dcterms:identifier` first).

### Several values on one property

`PropertyTarget.language` writes `@language` and, more importantly, decides
**which literal the write owns**. Without it a second write would clobber the
first: `OmekaClient.upsert_property_value` matches the first literal on a
property whatever its language, which is why `apply_text_value` no longer
delegates to it. A target with `language=None` keeps that language-blind rule,
so the OCR, correction and transcription updaters are unaffected.

`TextUpdate.extra_values` carries additional `(target, text)` pairs applied in
the **same PATCH**. AI_summary writes its French and English summaries this way:

```python
french = PropertyTarget(..., language="fr", adopt_untagged=True)
english = PropertyTarget(..., language="en")

update = TextUpdate("2231.txt", 2231, "Résumé…", extra_values=[(english, "Summary…")])
run_text_updates(client, [update], french, console=console)
```

`adopt_untagged` claims a pre-existing literal that carries no `@language`,
tagging it on the way past. Set it on the language that owns the legacy values —
IWAC's French summaries predate the tag — so a bilingual run upgrades them
instead of appending a second French value. Never set it on more than one target
of the same property: the first write would take the untagged literal and the
second would take it back.

An empty text is skipped rather than written, so a missing translation cannot
blank a value Omeka already holds; an item counts as `empty` only when *every*
one of its values is blank.

### Pre-write backup

Pass `backup_dir=` and every item's pre-write JSON is appended to a timestamped
`.jsonl` there — **flushed before its PATCH**, and only for items that actually
change:

```python
run_text_updates(client, updates, target, backup_dir=Path("backups"),
                 backup_label="summaries")
```

This is deliberately not `write_guard.WriteGuard.dump_backup`, which buffers
every payload and writes once at the end. That is right for a few hundred items
and wrong for a corpus pass: it holds ~50 MB of OCR in memory for 12k articles,
and a crash at item 7,000 leaves no backup at all — exactly when one is needed.
`open_backup()` is the streaming equivalent and yields `None` on a dry run,
which callers pass straight through.

---

## Console Utilities (`console_utils.py`)

One definition of the rich furniture every pipeline prints.

```python
from common.console_utils import count_table, key_value_table, print_file_table, standard_progress

with standard_progress(console) as progress:
    task = progress.add_task("[cyan]Working...", total=len(items))
    ...
    progress.update(task, advance=1)

console.print(key_value_table([("Model", "gemini-3.7-flash"), ("Items", "42")]))
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

This guide explains how to use `llm_registry.py` for model configuration and
`llm_provider.py` for provider calls. The provider re-exports the registry API,
so existing imports remain compatible.

## Overview

The `LLMConfig` class allows individual scripts to customize AI behavior without modifying the shared provider code. You can now configure:

- **OpenAI**: `reasoning_effort` and `text_verbosity`
- **Gemini / Gemma**: `thinking_level` ("minimal", "low", "medium", or "high"),
  clamped per model to the rungs that model actually has
- **Mistral**: no per-script parameters
- **OpenRouter**: `reasoning_effort` on the models that accept one

`temperature` is deliberately absent from that list. See
[Temperature](#temperature-dont-set-it) below.

## Available Models

The provider supports these models via the `MODEL_REGISTRY`:

| Key | Provider | Model ID | Label | Description |
|-----|----------|----------|-------|-------------|
| `gpt-5.6-luna` | OpenAI | `gpt-5.6-luna` | ChatGPT (GPT-5.6 Luna) | Cost-optimized tier, $0.20/$0.02/$1.20 per 1M |
| `gpt-5.6-terra` | OpenAI | `gpt-5.6-terra` | ChatGPT (GPT-5.6 Terra) | Balanced tier, $2/$0.20/$12 per 1M |
| `gpt-5.6-sol` | OpenAI | `gpt-5.6-sol` | ChatGPT (GPT-5.6 Sol) | Flagship tier, $5/$0.50/$30 per 1M |
| `gemini-3.7-flash` | Gemini | `gemini-3.7-flash` | Gemini 3.7 Flash | **The Flash every tier offers**; version-pinned, `LOW`/`MEDIUM`/`HIGH` thinking only |
| `gemini-flash` | Gemini | `gemini-flash-latest` | Gemini Flash | Rolling alias, currently 3.7; in no tier — use the pinned key unless the run stamps nothing |
| `gemini-flash-lite` | Gemini | `gemini-flash-lite-latest` | Gemini Flash-Lite | Most cost-effective, lowest latency |
| `gemini-pro` | Gemini | `gemini-pro-latest` | Gemini Pro | Highest quality; rolling, so absent from the OCR document tier |
| `gemini-3.6-flash` | Gemini | `gemini-3.6-flash` | Gemini 3.6 Flash | Version-pinned; superseded by 3.7, kept for the backlog it already annotated |
| `gemini-3.5-flash-lite` | Gemini | `gemini-3.5-flash-lite` | Gemini 3.5 Flash-Lite | Version-pinned; the generation-2 sentiment panel's Gemini seat |
| `gemini-3.1-flash-lite` | Gemini | `gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite | Version-pinned |
| `gemini-3.1-pro` | Gemini | `gemini-3.1-pro-preview` | Gemini 3.1 Pro | Version-pinned quality tier |
| `gemma-4` | Gemini | `gemma-4-31b-it` | Gemma 4 31B | Dense open-weights flagship, served on `GEMINI_API_KEY`; text + image only, `MINIMAL`/`HIGH` thinking only |
| `mistral-large` | Mistral | `mistral-large-2512` | Mistral Large 3 | 41B active params MoE |
| `ministral-14b` | Mistral | `ministral-14b-2512` | Ministral 3 14B | Fast, cost-effective |
| `mistral-small` | Mistral | `mistral-small-2603` | Mistral Small 4 | Hybrid reasoning model |
| `qwen3.5-moe` | OpenRouter | `qwen/qwen3.5-122b-a10b` | Qwen3.5 122B-A10B | Apache-2.0 open weights, MoE 10B active, $0.26/$2.08 per 1M tokens |
| `qwen3.5-moe-small` | OpenRouter | `qwen/qwen3.5-35b-a3b` | Qwen3.5 35B-A3B | Apache-2.0 open weights, MoE 3B active, $0.14/$1.00 per 1M tokens |
| `qwen3.5-dense` | OpenRouter | `qwen/qwen3.5-27b` | Qwen3.5 27B | Apache-2.0 open weights, dense, $0.195/$1.56 per 1M tokens |
| `deepseek-v4-flash-0731` | OpenRouter | `deepseek/deepseek-v4-flash-0731` | DeepSeek V4 Flash 0731 | **Default text model**; official release, 284B/13B active, 1M context, from $0.09/$0.18 per 1M tokens |
| `deepseek-v4-flash` | OpenRouter | `deepseek/deepseek-v4-flash` | DeepSeek V4 Flash Preview | **Archive only** — in no tier, so no pipeline offers it; see below |
| `deepseek-v4-pro` | OpenRouter | `deepseek/deepseek-v4-pro` | DeepSeek V4 Pro | 1.6T/49B active MoE flagship, $0.435/$0.87 per 1M tokens |
| `qwen3.8-27b-selfhosted` | Self-hosted | `Qwen/Qwen3.8-27B` | Qwen3.8 27B (self-hosted) | Apache-2.0, dense 27.8B; served from your own vLLM endpoint — see [`serving/`](../serving/README.md). Reasoning `low`/`medium`/`xhigh` |
| `qwen3.8-27b-openrouter` | OpenRouter | `qwen/qwen3.8-27b` | Qwen3.8 27B (OpenRouter) | The same weights, hosted. **In no tier** — it exists to measure one route against the other, at $0.45/$3.20 per 1M |

### Model Aliases

For convenience, these aliases are also supported:

| Alias | Resolves To |
|-------|-------------|
| `openai` | `gpt-5.6-luna` |
| `luna` | `gpt-5.6-luna` |
| `terra` | `gpt-5.6-terra` |
| `sol` | `gpt-5.6-sol` |
| `gpt-5.6` | `gpt-5.6-sol` |
| `gemini` | `gemini-3.7-flash` |
| `flash` | `gemini-3.7-flash` |
| `mistral` | `mistral-large` |
| `ministral` | `ministral-14b` |
| `qwen` | `qwen3.5-moe` |
| `deepseek` | `deepseek-v4-flash-0731` |
| `deepseek-pro` | `deepseek-v4-pro` |

OpenRouter slugs resolve as-is too, so a model id copied off openrouter.ai
(`qwen/qwen3.5-122b-a10b`) works without translation.

Where the same weights are reachable two ways, the vendor-prefixed slug names
the hosted route and the bare name the local one: `qwen3.8` is the self-hosted
entry, `qwen/qwen3.8-27b` the OpenRouter one. Note that the Hugging Face repo id
`Qwen/Qwen3.8-27B` lowercases to exactly that slug, so pasting it gets you the
hosted route — ask for the self-hosted entry by its short name.

The retired OpenAI keys still resolve: `gpt-5-mini` → `gpt-5.6-luna`, and
`gpt-5.1` / `gpt-5` → `gpt-5.6-sol`. Their underlying snapshots shut down on
2026-10-23, so prefer the new tier keys in new code.

See `MODEL_ALIASES` in `llm_registry.py` for the full list of legacy aliases.

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
| OpenRouter / Self-hosted | `chat.completions.create(response_format=type_to_response_format_param(Model))` |

This matters for OpenAI in particular. Its `strict` mode requires
`additionalProperties: false` on every object and *every* property listed in
`required`, and `model_json_schema()` emits neither — it drops any field that has
a default from `required`. Passing that raw schema with `strict: true` is rejected
by the API, and a caller's retry loop will report it as a generic failure.
`responses.parse()` runs the SDK's own `to_strict_json_schema()` transform, so the
schema is always valid.

The two open-model clients run **the same transform** — via the SDK's
`type_to_response_format_param`, so the request is byte-identical to what
`parse()` sent — but issue it through `create()` and validate the response
themselves. The reason is in the next paragraph, and it is not a style
preference: `parse()` validates `message.content` and *raises* before returning,
so a model that fences its JSON produced a `ValidationError` that no fallback
could catch. Reintroducing `parse()` on these two clients silently removes the
recovery below; `test_structured_output_never_delegates_parsing_to_the_sdk`
guards against it.

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

### Usage totals

Every client keeps `client.usage` (`UsageTotals`): requests, input/output
tokens, cached and reasoning tokens where the provider reports them, and cost
where the provider states it (OpenRouter's `usage.cost`; elsewhere `None`
rather than a guess from a rate card). `usage.summary()` is one line for a run
summary, which NER, summarization and reference enrichment print.

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

Gemini and Gemma models use `thinking_level` to control how much reasoning the
model does before answering. Thinking cannot be disabled — these models always
reason to some degree.

**Which rungs exist is per-model and changes between releases.** Google returns a
400 `INVALID_ARGUMENT` for a level a model does not have, so this is not a soft
preference. Each `ModelOption` declares `supported_thinking_levels` (empty = all
four), and `clamp_thinking_level()` snaps a request to the nearest one that
exists, rounding **up** on a tie — a level the model cannot serve becomes more
deliberation, never a silent drop to none.

That means a pipeline can go on asking for `"minimal"` to mean "as little as this
model offers" and stay correct across a vendor's ladder change. Gemini 3.7 Flash
dropping `MINIMAL` — while `gemini-flash-latest` rolled onto it the same day —
would otherwise have broken OCR, HTR, audio, video and every text tier at once.

| Model | Thinking Levels | Default | Best For |
|-------|----------------|---------|----------|
| Gemini 3.7 Flash | `"low"`, `"medium"`, `"high"` | `"low"` | Fast processing, bulk tasks |
| Gemini 3.6 Flash / Flash-Lite | `"minimal"`, `"low"`, `"medium"`, `"high"` | `"minimal"` | Cheapest bulk work |
| Gemini Pro | `"low"`, `"medium"`, `"high"` | `"low"` | Complex analysis, higher accuracy |
| Gemma 4 31B | `"minimal"`, `"high"` | `"high"` | Open-weights alternative |

Verified against the live API on 2026-08-14. Re-probe rather than infer when
adding a model: nothing in a model's name predicts which rungs it kept.

### Mistral Parameters

Mistral takes no per-script parameters; each model's `temperature` comes from
`MODEL_REGISTRY` (`0.2` for Large 3 and Ministral, `0.3` for Small 4).

**Available Mistral Models**:
- **`mistral-large`**: Mistral Large 3 — flagship 41B active params MoE model
- **`ministral-14b`**: Ministral 3 14B — fast, cost-effective ($0.2/M tokens)
- **`mistral-small`**: Mistral Small 4 — hybrid reasoning; accepts `reasoning_effort` `none` or `high`

**Note**: Structured output uses `client.chat.parse()`; when a reasoning effort is
requested on Small 4 the adapter calls `chat.complete()` and parses the JSON
itself, because `parse()` does not document how it interacts with thinking chunks.

### OpenRouter Parameters

OpenRouter is a router in front of open-weights models, not a lab. One
`OPENROUTER_API_KEY` covers all of them, and because the endpoint is
OpenAI-compatible it reuses the already-installed `openai` SDK — no extra
dependency.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_effort` | `str` | per model | Only sent to models that accept one; see below |

Every request carries `provider: {data_collection: "deny", require_parameters: true}`
— no backend that trains on prompts, and only backends that support every
parameter sent (the `json_schema` response format above all). That is a
*training* opt-out, not a storage one: set `OPENROUTER_ZDR=1` in the
environment to add OpenRouter's `zdr` flag and route only to endpoints that
store nothing, at the price of a much shorter provider list (for DeepSeek it can
leave none, and the request fails with a 503).

**Available OpenRouter models**:
- **`qwen3.5-moe` / `qwen3.5-moe-small` / `qwen3.5-dense`**: Qwen3.5 122B-A10B, 35B-A3B and 27B, all Apache-2.0 open weights. `qwen3.5-moe` was re-pointed from 35B-A3B to 122B-A10B on 2026-07-31 so the sentiment panel's open-weights members sit at comparable active-parameter counts (10B vs DeepSeek V4 Flash's 13B; 35B-A3B activates only 3B)
  (deliberately not the Flash/Plus/Max hosted tiers, which publish no weights). Accept
  `reasoning_effort` minimal/low/medium/high/xhigh, normalised by OpenRouter.
- **`deepseek-v4-flash-0731`**: official DeepSeek V4 Flash release and the shared text default; accepts exactly `"low"`, `"high"`, or `"max"` reasoning (bulk pipelines use low; sentiment uses high).
- **`deepseek-v4-flash`**: the April preview — **archive only**. Removed from every tier on 2026-08-07, so no pipeline offers it and no `--model` accepts it; every DeepSeek Flash run goes to 0731. The `MODEL_REGISTRY` entry survives only so the slug still resolves where it turns up in an old pilot payload; the sentiment values it wrote were deleted from Omeka the same day. `test_the_deepseek_preview_is_archive_only` keeps it out of the tiers.
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

**Structured output has a fallback.** Open models routinely return schema-valid
JSON as a plain string, sometimes inside a ``` fence or after a sentence of
preamble. `OpenRouterClient.generate_structured()` extracts the JSON document
from whatever packaging it arrives in and validates that, so a well-formed
answer is not thrown away over its wrapping.

This only works because the client parses the response itself. Delegating to the
SDK's `parse()` helper — which it did until 2026-08-16 — meant the SDK validated
`message.content` and raised a `ValidationError` first, making the fallback
unreachable in production: a fenced answer consumed a retry and, once retries
ran out, was recorded as an `analysis_error`. The unit tests did not catch it
because mocking `parse()` returns content the real SDK would never hand back.
If you add a provider that speaks the OpenAI wire format, parse the response
yourself.

### Self-hosted Parameters

An OpenAI-compatible endpoint you run yourself — vLLM on a GPU cluster, or
llama.cpp / LM Studio / TGI on anything smaller. `SelfHostedClient` subclasses
`OpenRouterClient`, because open models behave the same way wherever they are
served and the tolerant structured-output recovery above is worth strictly more
here: there is no router in front filtering for backends that honour
`response_format`.

| Variable | Required | Description |
|---|---|---|
| `SELFHOSTED_LLM_BASE_URL` | yes | e.g. `http://localhost:8000/v1`, typically an SSH tunnel to a compute node |
| `SELFHOSTED_LLM_API_KEY` | no | Matches the server's `--api-key`; falls back to vLLM's `EMPTY` convention |

**The endpoint is not in the catalog.** A `MODEL_REGISTRY` entry describes a
model; where it is served today is deployment state, and on a cluster it changes
with every job. So the URL is read from the environment by the adapter, exactly
as every other client reads its key, and no pipeline passes one through. A
missing URL fails at client construction — which is what lets the sentiment
pilot report the model as *skipped* on a machine with no tunnel open instead of
aborting a run, and what lets CI import everything with no endpoint at all.

**Nothing is sent about routing.** No `provider` preferences, no attribution
headers. `data_collection: "deny"` is a contract with a third party, and on this
route there is none — the text reaches a machine you control and stops there.
The guarantee is physical rather than contractual, which is stronger, and a
strict server may reject unknown body fields anyway.

**Reasoning depth travels in `chat_template_kwargs`.** That is how vLLM passes
arguments into a model's chat template: Qwen3.8 reads `reasoning_effort` there
(`low`/`medium`/`xhigh`). The same clamping applies as for OpenRouter — a level
the model does not declare degrades to its default rather than being forwarded.

Setting one up, and the reasoning-depth probe that should precede trusting one,
are documented in [`serving/README.md`](../serving/README.md).

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
    thinking_level="medium",       # Gemini: balanced thinking
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
    thinking_level="minimal",      # Gemini: least reasoning the model offers
)
```

`"minimal"` is safe to write even for models that dropped that rung — the clamp
turns it into their shallowest (`"low"` on Gemini 3.7 Flash and every Pro).

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
    parser.add_argument("--model", help="Model key (e.g., openai, gemini-3.7-flash)")
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
   - Gemini Flash: `"minimal"` for fast tasks (clamped to `"low"` on 3.7), `"low"`/`"medium"` for balanced work, `"high"` for complex analysis
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

1. Update `MODEL_REGISTRY` in `llm_registry.py`
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
A: Use registry keys like `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gemini-3.7-flash`, `gemini-pro`, `mistral-large`, `ministral-14b`. Common aliases like `openai`, `luna`, `terra`, `sol`, `gemini`, `mistral` also work, as do the retired `gpt-5-mini` / `gpt-5.1` keys.

**Q: How do I restrict which models a pipeline can use?**  
A: Use `allowed_keys` in `get_model_option()`:
```python
model_option = get_model_option(args.model, allowed_keys=["gemini-3.7-flash", "gemini-pro"])
```
