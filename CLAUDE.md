# CLAUDE.md

AI document-processing pipelines for the Islam West Africa Collection (IWAC), a
digital archive of 14,500+ items in Omeka S. Each pipeline automates one task —
OCR extraction and correction, summarization, NER, transcription, HTR, sentiment —
against Gemini, OpenAI, Mistral, or the open-weights models (Qwen, DeepSeek)
reached through OpenRouter.

## Setup

Always use the project venv at `.venv/`. If it is not activated, invoke it directly:
`.venv\Scripts\python` on Windows, `.venv/bin/python` elsewhere.

```bash
pip install -e ".[dev]"   # pytest + ruff
pytest tests/
ruff check .
```

The test suite is green on Windows and Linux; a red test is a real signal.

## Layout

Pipelines are `AI_<name>/` directories of numbered scripts run in sequence —
typically `01` downloads from Omeka, `02` calls the AI, `03` writes back. Most
accept CLI flags and fall back to interactive prompts.

Which architecture rules apply depends on the pipeline's category:

- **Text-only** — `AI_summary`, `AI_NER`, `AI_ocr_correction`,
  `AI_sentiment_analysis`, `NotebookLM`
- **Multimodal** — `AI_audio_summary`, `AI_htr_extraction`, `AI_ocr_extraction`,
  `AI_video_summary`, `AI_summary_issue`
- **Agent-driven** — `AI_reference_indexing` (orchestrated by the
  `reference-indexing` skill rather than a numbered run)

Shared code in `common/`. `common/README.md` covers `omeka_client`, `llm_provider`,
`rate_limiter`, `retry` and `ffmpeg_utils` in depth; the rest are only described here:

| module | purpose |
|---|---|
| `iwac_config.py` | IWAC-instance constants: property IDs, authority item sets, `AI_MODEL_ITEMS` |
| `gemini_utils.py` | Gemini plumbing for multimodal scripts: generation config, Files API upload, text extraction that skips `thought` parts |
| `gemini_page_processor.py` | The page-by-page Gemini PDF loop shared by OCR and HTR: inline→Files-API fallback, retry policy, `finish_reason` handling, page markers, batch driver |
| `omeka_text_updater.py` | The `03` write step shared by summary/OCR/correction/transcription: change detection, `@annotation` attachment, `@language`-tagged values, several values per item in one PATCH, `--dry-run`, confirmation gate |
| `omeka_link_updater.py` | The same idempotent write for *resource-link* properties (`dcterms:subject`, `dcterms:spatial`): dedup against existing links, whole-item PATCH, `dry_run`, pre-write snapshot |
| `write_guard.py` | The gate in front of every bulk write: `--dry-run`, `--yes`, pre-write payload dump, confirmation panel |
| `checkpoint.py` | Atomic JSON checkpoints for resumable runs: a stored fingerprint of model, prompt and input decides resume vs. regenerate |
| `console_utils.py` | `standard_progress()`, `key_value_table()`, `count_table()` — one definition of the rich furniture every pipeline prints |
| `downloader.py` | `stream_download()` — streaming download via a `.part` temp file, used by the PDF and media downloaders |
| `prompt_loader.py` | Discovery and interactive selection for pipelines holding several `prompts/*.md` |
| `pdf_downloader.py` | Shared Omeka PDF download step (`AI_ocr_extraction/01`, `AI_summary_issue/01`) |
| `pdf_utils.py` | `PdfPageSource` (parse once, serve many pages) plus one-off page extraction and page counts |
| `reconciliation.py` | Fuzzy matching of extracted entities against authority records |

## Architecture rules

**All Omeka S access goes through `common/omeka_client.py`.** Never use raw
`requests` with Omeka credentials in a pipeline script. Do not modify
`omeka_client.py` without asking — every pipeline depends on it.

**Text-only pipelines route every LLM call through `common/llm_provider.py`.**
Never instantiate `openai.OpenAI()`, `google.genai.Client()`, or
`mistralai.Mistral()` in a text script. SDK adapters live in that module; the
dependency-free model catalog and config defaults live in
`common/llm_registry.py`. Add models there first so every pipeline picks them up.

**Pipelines pick a model *tier*, not a list of keys.** `TEXT_ECONOMY_MODELS`,
`TEXT_EXTENDED_MODELS`, `TEXT_FULL_MODELS`, `TEXT_OPEN_MODELS` and
`GEMINI_DOCUMENT_MODELS` live in `llm_registry`; a pipeline's `ALLOWED_MODELS`
should be one of them. Retiring a model is then a one-line change instead of a
grep across five pipelines.

**The shared text default is `DEFAULT_TEXT_MODEL_KEY`.** It currently points to
the pinned `deepseek-v4-flash-0731` release. Text-only entry points use it when
`--model` is omitted; multimodal extraction stages do not pretend a text-only
model can consume their source media.

**Never set `temperature` in a pipeline.** It belongs to the vendor and lives once
in `MODEL_REGISTRY`: nothing at all for Gemini 3 / Gemma, `1.0` for DeepSeek V4,
`0.7` for Qwen3.5, and the model-specific Mistral default. A pipeline picks a *tier*, so it cannot know
whose model the run will land on, and the values are not interchangeable — Google
and Alibaba both document a lowered temperature as a cause of looping, which here
means a transcript repeating a paragraph through a 90-minute interview or OCR
stalling on one line. `top_p` / `top_k` are never set either. Constrain output with
system-prompt rules and `generate_structured()` instead.

**OpenRouter models carry a routing policy, not just a key.** Every request is
pinned to `data_collection: "deny"` and `require_parameters` via
`OPENROUTER_PROVIDER_PREFS` — the first because these pipelines send whole
archival documents to third-party backends, the second because `json_schema`
support varies by backend. Add models in `common/llm_registry.py`, never by
letting a pipeline pass an arbitrary `vendor/model` slug through.

**Multimodal pipelines are the exception** and call provider SDKs directly, because
they need capabilities the shared provider does not expose. They must still use
`common/gemini_utils.build_gemini_client()` so transport calls have a finite
deadline, plus `common/rate_limiter.py`: call `wait()` before each request, and translate provider
errors with `is_quota_exhausted()` so daily-quota exhaustion raises
`QuotaExhaustedError` (save partial results, stop) instead of being retried like a
transient 429.

## Omeka gotchas

For anything about the archive itself — resource classes, templates, property
semantics, the Hugging Face export and how Omeka fields map onto it — invoke the
`iwac-data` skill rather than inferring from the pipelines. It is the source of
truth for that, and it is where archive knowledge should be added. (Personal skill,
so it may be absent in a fresh clone; the rules below stand on their own.)

The rest of this section is destructive if got wrong, and none of it is apparent
from the code.

**PATCH the whole item, always.** Omeka treats RDF properties as one block: any
property missing from the payload is deleted. `isPartial=1` does not protect them.
Fetch with `get_item()`, mutate, send the full object back. Never trim fields to
reduce payload size — a timeout is the better problem to have.

**A write script must parse `argv`.** On 2026-08-02 `AI_NER/03_Omeka_update.py` had
no argument parser, so `--help` was not recognised as a request for help: the script
ran its real update and PATCHed 630 live items before it was killed. A script that
ignores arguments treats a typo as consent. Every entry point that PATCHes or POSTs
now goes through `common/write_guard.py` — argument parsing, `--dry-run`, a pre-write
payload dump, and a confirmation gate — and `tests/test_write_guard.py` fails if a new
write script skips it. Never run one of these scripts to "see what it does".

**`upsert_property_value()` drops `@annotation` and ignores `@language`.** It rebuilds
the value object from five keys when appending to a property that has no literal yet,
so value annotations (`iwac:summaryModel`, `iwac:ocrModel` — which AI model produced
the content) are silently lost; and it matches the *first* literal on a property
whatever its language, so calling it once per language makes the second write clobber
the first. `common/omeka_text_updater.apply_text_value()` handles both and is what the
`03` steps use — it no longer delegates to `upsert_property_value()` for exactly this
reason. Call `upsert_property_value()` directly only for a single untagged literal
whose annotation you re-attach yourself. Before any bulk write, dump the pre-write
payloads to JSON; that backup is the only route back.

**AI summaries go in `bibo:shortDescription`**, exported to Hugging Face as
`descriptionAI`. Not `dcterms:abstract`, which holds publisher abstracts on issues
and scholarly references. When unsure which property a pipeline should target, count
live field population per resource class through the API rather than trusting a
docstring.

**Since 2026-08-06 that property carries TWO literals**, tagged `@language` `fr` and
`en`, both annotated with the one model that produced them. The ~12,300 summaries
written before then carry no language tag at all, so the French `PropertyTarget` sets
`adopt_untagged=True` to claim and tag the existing literal instead of appending a
second French value beside it — never set that flag on more than one target of the
same property. The HF export pipe-joins multi-values, so `descriptionAI` becomes
`"résumé|summary"` until the IWAC-Hugging-Face mapper learns to split by language.

Instance-specific constants — property IDs, authority item sets, the `AI_MODEL_ITEMS`
model-provenance registry — belong in `common/iwac_config.py`, not inline in scripts.
Adding a model there means creating its Omeka authority item first (class 244,
template 3, item set 267, `dcterms:type` → "Notice d'autorité").

## Conventions

Prompts live beside their pipeline as `.md` files, loaded at runtime. Structured
extraction uses `generate_structured()` with a Pydantic schema rather than parsing
JSON by hand. Terminal output uses `rich`, via `common/console_utils.py` for
progress bars and the standard tables. Beyond that, match the surrounding code.

Scripts put the repo root on `sys.path` with one canonical line —
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`. `insert`, not
`append`: with `append`, a same-named module earlier on the path shadows `common`.
