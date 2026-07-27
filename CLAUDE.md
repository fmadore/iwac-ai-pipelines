# CLAUDE.md

AI document-processing pipelines for the Islam West Africa Collection (IWAC), a
digital archive of 14,500+ items in Omeka S. Each pipeline automates one task —
OCR extraction and correction, summarization, NER, transcription, HTR, sentiment —
against Gemini, OpenAI, or Mistral.

## Setup

Always use the project venv at `.venv/`. If it is not activated, invoke it directly:
`.venv\Scripts\python` on Windows, `.venv/bin/python` elsewhere.

```bash
pip install -e ".[dev]"   # pytest + ruff
pytest tests/
ruff check .
```

Three tests in `tests/test_pdf_downloader.py` fail on Windows — a helper opens
`/dev/null`. Pre-existing, unrelated to whatever you are changing.

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
| `prompt_loader.py` | Discovery and interactive selection for pipelines holding several `prompts/*.md` |
| `pdf_downloader.py` | Shared Omeka PDF download step (`AI_ocr_extraction/01`, `AI_summary_issue/01`) |
| `pdf_utils.py` | Page-by-page PDF extraction and page counts |
| `reconciliation.py` | Fuzzy matching of extracted entities against authority records |

## Architecture rules

**All Omeka S access goes through `common/omeka_client.py`.** Never use raw
`requests` with Omeka credentials in a pipeline script. Do not modify
`omeka_client.py` without asking — every pipeline depends on it.

**Text-only pipelines route every LLM call through `common/llm_provider.py`.**
Never instantiate `openai.OpenAI()`, `google.genai.Client()`, or
`mistralai.Mistral()` in a text script. Models and their config defaults live in
`MODEL_REGISTRY` / `MODEL_ALIASES` — read them there rather than from a table in a
doc, and add new models there first so every pipeline picks them up.

**Multimodal pipelines are the exception** and call provider SDKs directly, because
they need capabilities the shared provider does not expose. They must still use
`common/rate_limiter.py`: call `wait()` before each request, and translate provider
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

**`upsert_property_value()` drops `@annotation`.** It rebuilds the value object from
five keys when appending to a property that has no literal yet, so value annotations
(`iwac:summaryModel`, `iwac:ocrModel` — which AI model produced the content) are
silently lost. Re-attach them explicitly after calling it. Before any bulk write,
dump the pre-write payloads to JSON; that backup is the only route back.

**AI summaries go in `bibo:shortDescription`**, exported to Hugging Face as
`descriptionAI`. Not `dcterms:abstract`, which holds publisher abstracts on issues
and scholarly references. When unsure which property a pipeline should target, count
live field population per resource class through the API rather than trusting a
docstring.

Instance-specific constants — property IDs, authority item sets, the `AI_MODEL_ITEMS`
model-provenance registry — belong in `common/iwac_config.py`, not inline in scripts.
Adding a model there means creating its Omeka authority item first (class 244,
template 3, item set 267, `dcterms:type` → "Notice d'autorité").

## Conventions

Prompts live beside their pipeline as `.md` files, loaded at runtime. Structured
extraction uses `generate_structured()` with a Pydantic schema rather than parsing
JSON by hand. Terminal output uses `rich`. Beyond that, match the surrounding code.

Model keys are duplicated in each pipeline's `--model` choices, so a registry change
means grepping for the old key across pipelines, READMEs, and `.env.example`.
