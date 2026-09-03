# Changelog

Releases are tagged on GitHub and archived on Zenodo (concept DOI
[10.5281/zenodo.21804210](https://doi.org/10.5281/zenodo.21804210)). The
second half of this file holds the operational history that used to live as
comments beside the constants it explains — kept because an annotation on the
archive can only be read back with it, moved here so the code states rules and
the changelog tells stories.

## 1.1.0 — 2026-09-02

Pre-publication release accompanying *When AI Meets the Archive*.

### Provenance
- Every `dcterms:subject` / `dcterms:spatial` link written by NER or
  reference indexing carries an `iwac:nerModel` value annotation (property 314)
  naming the model that proposed it. Links already on an item are never
  re-stamped. The write steps read the model from the checkpoint the AI step
  left beside its CSV; `--model` overrides, and they ask when neither applies.
- `AI_ocr_extraction/02_mistral_ocr_processor.py` runs the pinned
  `mistral-ocr-4-1` through the shared `common/mistral_ocr.py` client instead
  of a copy of it addressed to the rolling `mistral-ocr-latest` alias. A test
  forbids the alias in any pipeline script.
- `AI_summary_issue/03` asks which model consolidated the index rather than
  stamping the registry default silently.

### Python only
- The `.claude/` agents and skills are gone. Reference-indexing step 2 is
  `AI_reference_indexing/02_enrich_references.py` (structured output through
  the shared provider, resumable, keyword summary); the magazine pipeline is
  its Gemini and Mistral scripts.

### Write safety
- Every step that writes to Omeka now shares one gate
  (`common/write_guard.py`): `--dry-run`, `--yes`, `--backup-dir`,
  `--no-backup`, the same confirmation panel, and a pre-write dump of every
  item before its PATCH. Three text write steps and the sentiment panel had no
  backup at all; the magazine TOC step also had no `--yes` and ignored failures.
- `AI_publication_extraction/04`'s "unchanged" check compared against Omeka's
  echoed value objects and never matched, so every rerun re-PATCHed.

### Shared code
- `OmekaClient.get_items()` takes any `/api/items` filter (class, template,
  `modified_after` …), follows `Omeka-S-Total-Results`, and warns if the count
  moves during the walk; `iter_items()` streams, `count_items()` reads the
  header, `get_items_by_ids()` batch-fetches, `list_page()` serves samplers,
  and the client remembers `Omeka-S-Version`. Five hand-rolled paging loops
  are gone. `search_items_by_property()` raises on a transport failure instead
  of answering "no match". Session backoff is `1.0`. `upsert_property_value()`
  is deprecated.
- `common/link_update_cli.py` and `common/reconciliation_cli.py` hold the
  write and reconciliation runs that NER and reference indexing had each
  copied; the four scripts are entry points. The two PDF downloaders are one
  `run_cli()` with `--item-set-id`.
- `common/reconciliation.py` prepares each string once and gates the expensive
  `SequenceMatcher.ratio()` behind its cheap upper bounds.
- `common/llm_provider.py` records tokens (and cost where the provider states
  it) on every client as `client.usage`; `LLMConfig.service_tier` reaches
  OpenAI (`AI_summary/02 --service-tier flex`); `OPENROUTER_ZDR=1` adds
  OpenRouter's zero-retention routing.
- OCR correction no longer picks a thinking level by model name.

### Interfaces
- `--prompt N` on the audio and video transcribers, `--no-split` on the
  audio one, `--item-set-id` / `--item-id` on the three downloaders that only
  asked; the audio downloader stops producing filenames the transcriber then
  refuses. Every `main()` returns an exit code. One repo-root idiom.
- Input folders (`PDF/`, `Audio/`, `video/`, `TXT/`, `ALTO/`) ship empty with a
  `.gitkeep`; output, cache, backup and `serving/work/` are git-ignored.

### Documentation
- Around thirty README statements corrected against the code, including flags
  argparse rejected, a wrong expansion of IWAC, a sentiment annotation that is
  not written, and the audio pipeline's provenance claim.

## 1.0.0 — 2026-08-05

First archived release.

---

# Operational history

What the constants used to explain in place. Dates are when the archive changed.

## AI model authority items (`common/iwac_config.AI_MODEL_ITEMS`)

- **2026-07-27 / 07-31** — Item 79608 "Gemini 3.6 flash" was a duplicate of
  79611 and item 79609 "GPT-5.6 Luna" was deleted upstream and replaced by
  79610; both are in `RETIRED_AI_MODEL_ITEM_IDS` and must not be annotated with.
- **2026-07-31** — The Gemini key was `gemini-flash`, the registry key of the
  rolling `gemini-flash-latest`, which reports its version as literally
  "Gemini Flash Latest"; annotating such a run as "Gemini 3.6 flash" asserted a
  version the run could not confirm. Re-keyed to the pinned `gemini-3.6-flash`.
- **2026-08** — `gemini-flash-lite` was the same bug and outlived the fix: the
  key of the rolling `gemini-flash-lite-latest` claiming item 78631 "Gemini 3.1
  flash lite", which resolved to 3.5 once 3.5 Flash-Lite shipped. Both
  Flash-Lite generations now have pinned keys and their own items; the rolling
  key is deliberately absent. `gemini-pro` was the third case, found by the
  guard added for the second (`test_no_registry_key_is_a_rolling_alias`), and
  is re-keyed to `gemini-3.1-pro`; the rolling `gemini-pro` stays in the
  registry for pipelines that stamp nothing.
- **2026-08-07** — `claude-opus` split into `claude-opus-4.6` (item 78528) and
  `claude-opus-5` (79615), so an operator asserting which Claude read a
  magazine could not name last year's release by accident. Since 2026-09-02 no
  pipeline uses a Claude agent; the keys stay so the annotations resolve.
- **2026-08-14** — Three `display_title` values corrected to the live item
  titles ("Gemini 3.1 Pro", "Gemini 3.6 Flash", "Gemini 3.1 Flash Lite").
  Omeka regenerates the key on read, so nothing stored was wrong; it is what
  the confirmation panel shows.
- **2026-08-18** — `mistral-ocr-4-1` (item 111889) created for
  `AI_publication_extraction`; the 425 `bibo:content` values already on the
  reference corpus carry no provenance annotation.
- **2026-08-25** — `qwen3.8-27b-selfhosted` (item 111933): keyed on the
  self-hosted route because that is what produced the annotations; the
  OpenRouter twin has written nothing and would need its own item.
- **2026-08-27** — `gemini-3.5-transcribe` (item 113077) for
  `AI_audio_summary/02c`; the audio pipeline stamped no transcription
  provenance before it.

## Model registry (`common/llm_registry.py`)

- **Gemma 4 on two routes.** `gemma-4` (Gemini API) is free of charge because
  it is served on the free tier, and Google's pricing page states that
  free-tier content is used to improve its products — the thing
  `data_collection: "deny"` exists to prevent when whole archival articles are
  sent. `gemma-4-openrouter` is the key for anything that sends archive text;
  `gemma-4` stays for multimodal document work the chat API cannot do. Gemma
  has two thinking levels, MINIMAL and HIGH; a request for "medium" lands on
  `high`.
- **Qwen3.8 27B on two routes.** Same weights, and the route is half of what a
  provenance record claims. Both carry temperature 1.0, Qwen's thinking-mode
  recipe (the model card's 0.7 is the non-thinking one). The self-hosted
  default is `low`, not the vendor's `xhigh`: an unconfigured bulk run
  reasoning as hard as it can on shared university GPUs is the accident to
  design against. The hosted twin (`qwen3.8-27b-openrouter`) is in no tier and
  exists to measure one route against the other.
- **DeepSeek V4 Flash.** The dated `deepseek-v4-flash-0731` is the shared text
  default; the April preview `deepseek-v4-flash` is in no tier and survives
  only so annotations it wrote still resolve.
- **2026-08-14** — `gemini-flash-latest` rolled onto Gemini 3.7 Flash, which
  dropped the `MINIMAL` thinking level, and every pipeline that hardcoded
  `minimal` began failing with a 400. `supported_thinking_levels` and
  `clamp_thinking_level()` date from that day.

## Sentiment panel (`AI_sentiment_analysis/sentiment_core.PANEL`)

- **2026-07-31** — Generation 2 goes live: six properties per member named for
  the model, never the vendor, because vendor-keyed generation 1 could not be
  attributed without git archaeology. The Gemini slot moved off
  `gemini-3.6-flash` the same day: at $1.50/$7.50 per 1M it cost five to
  seventeen times the others, so any disagreement read as "the expensive model
  knows better". Nothing was ever written to `iwac:gemini36Flash*`.
- **2026-08-05** — Qwen3.5 122B-A10B dropped before annotating anything: with
  `require_parameters` OpenRouter left four endpoints and a median call of
  104 s against 4–6 s for the rest. `iwac:qwen35A10b*` holds zero values.
- **2026-08-07** — Generation 1 (`iwac:gemini*`, `iwac:chatgpt*`,
  `iwac:mistral*`, 12,286 items each) and the April DeepSeek preview (11,482)
  deleted from Omeka after confirmation on the Hugging Face full mirror. The
  Hub keeps generation 1 frozen with `omeka_prefix=None`; that freeze is the
  archive.
- **2026-08-14** — The Google slot became Gemma 4 31B, replacing
  `gemini-3.5-flash-lite`, which had held it since 07-31 and annotated nothing
  (verified at 0 items). Gemma replaces rather than joins Gemini: same lab and
  pretraining family, so running both buys correlated annotator error. It is
  routed through OpenRouter (see the registry note); the Gemini route is also
  capped at 16,000 input tokens per minute for this model. OpenRouter lists 19
  endpoints, 16 with structured outputs, so it does not repeat Qwen3.5's
  starvation. Its graduated thinking levels collapse to on/off across
  third-party backends: measured 2026-08-14, `medium` and `high` are
  indistinguishable in latency and reasoning length.
- **2026-08-25** — Qwen3.8 27B joined as a fifth voice, annotated on
  university hardware: the first member whose requested `medium` is a rung
  the model has. Its coverage is 12,098 of 12,251 and stays there: 153
  articles were attempted four times and retired, 145 failing the schema's
  cross-field rule because the model declines subjectivité when Islam is
  peripheral where the prompt licenses declining only when it is absent.
  Recorded, not repaired; its missing subjectivité is not missing at random.
- **Reasoning depth, verified 2026-07-29/31 and 08-14** — GPT-5.6 Luna accepts
  none/low/medium/high/xhigh/max; Gemma 4 only MINIMAL/HIGH; DeepSeek 0731
  only low/high/max; Mistral Small 4 only none/high. Only Luna and Qwen3.8 sit
  at a genuine middle; the others are rounded up to `high`, which any write-up
  of the panel must state.
