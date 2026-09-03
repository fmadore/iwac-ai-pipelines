# Document Summarization

Generate **bilingual (French + English)** summaries of text content from Omeka S collections using AI.

## Why This Tool?

Large document collections are difficult to browse. Researchers need to open each item to understand its contents. AI-generated summaries enable quick scanning and improve search relevance—users can find documents by concept rather than exact keyword matches.

The collection is francophone but its readership is not. An English summary beside the French one makes an item findable by a researcher who does not read French, without translating the source itself.

## How It Works

```
Omeka S → Extract text → AI summarization (fr + en) → Update database with both
```

1. **Extract** (`01_extract_omeka_content.py`): Pull OCR text from Omeka S items
2. **Summarize** (`02_AI_generate_summaries.py`): Generate the French and English summaries
3. **Update** (`03_omeka_update_summaries.py`): Store both in Omeka S

Each document is read **once** and rendered **twice**. The two summaries are neither independent passes nor a word-for-word translation: they report the same facts, the same actors and the same figures, each idiomatic in its own language. A fact present in one is present in the other, so an item is equally findable from either.

## Quick Start

```bash
python 01_extract_omeka_content.py --item-set 2201    # Extract text
python 02_AI_generate_summaries.py                    # Generate summaries
python 03_omeka_update_summaries.py --dry-run         # Preview, then drop --dry-run
```

### Running the whole corpus

Articles span **58 item sets and 39 belong to none**, so select them by class, not by item set:

```bash
python 01_extract_omeka_content.py --resource-class
```

That fetches all 12,356 `bibo:Article` items and drops the **51** outside French/English (Ewé 32, Kabiyè 11, Dendi 2, plus 6 with no language value), leaving 12,305 — the same scope the sentiment panel uses. For a representative pilot, use `--sample N` rather than `--limit N`: items come back in ID order, so the first N are one newspaper's consecutive issues.

```bash
python 01_extract_omeka_content.py --resource-class --sample 200
python 02_AI_generate_summaries.py --workers 6
python 01_extract_omeka_content.py --resource-class --modified-after 2026-08-01   # incremental re-run
python 02_AI_generate_summaries.py --service-tier flex   # OpenAI at ~half price, slower; for an overnight pass
python 03_omeka_update_summaries.py --dry-run
```

**Measured on a 200-article pilot** (GPT-5.6 Luna, 6 workers): 2 m 22 s, 0 failures, mean 546 FR / 519 EN characters. Extrapolated to 12,305 articles: **~2.4 hours** and **~$7**. Serial — `--workers 1` — would be ~14 hours.

The cost breakdown matters, because two things make it much cheaper than a naive estimate:

| | tokens | rate | cost |
|---|---:|---|---:|
| fresh input | 13.3M | $0.20/1M | $2.67 |
| **cached** input | 16.6M | $0.02/1M | $0.33 |
| output | 3.5M | $1.20/1M | $4.19 |
| | | | **$7.19** |

**55% of input is cached** — the 5,113-character system prompt is the shared prefix of every request, and 39 of 40 calls hit it. And only ~8% of output is reasoning at `effort=low`, unlike the sentiment panel where reasoning dominates.

> Verify the rate card before quoting a figure. An earlier estimate here said **$50** because it trusted a stale `$1/$6` in `llm_registry.py` (real Luna is `$0.20/$1.20`) and ignored caching — 7× too high. Prices live in the model descriptions in `common/llm_registry.py` and are checked against [OpenAI's pricing page](https://developers.openai.com/api/docs/pricing), not inferred from a tier name.

> **Do not delete the existing summaries first.** `adopt_untagged=True` overwrites them in place, so there is nothing left over to delete; and deleting first turns any mid-run failure into 12,300 articles with no summary at all, where the pipeline as written simply leaves the old one standing.

The summarization script runs on GPT-5.6 Luna unless `--model` names another
registry model (step 03, which uploads, is the one that asks which model wrote
the summaries when `--model` is omitted):

```bash
python 02_AI_generate_summaries.py  # GPT-5.6 Luna by default
```

Successful files are checkpointed with the exact model ID, prompt hash, and
source-text hash. Re-running skips exact matches and regenerates only changed
or previously failed inputs. **Both** language files must be on disk for an item
to resume — a run interrupted between the two writes regenerates rather than
shipping an item in one language. If existing output has no matching provenance,
the script stops instead of mixing runs; use `--force` to replace it.

## Supported Models

| Model | Provider | Speed | Cost per 1M (in / cached / out) | Full corpus |
|-------|----------|-------|------|---|
| `gpt-5.6-luna` | OpenAI | Fast | $0.20 / $0.02 / $1.20 | **~$7** (default) |
| `deepseek-v4-flash-0731` | DeepSeek via OpenRouter | Slow | $0.09 / — / $0.18 | ~$3.25 |
| `gemini-3.7-flash` | Google | Fast | see registry | — |
| `ministral-14b` | Mistral | Fast | see registry | — |

All models produce comparable summary quality for this task. Luna is the default for
throughput: measured over the sentiment panel's full-corpus passes, Luna ran 2.7 h
against DeepSeek V4 Flash 0731's 31.5 h — 0731 has no middle reasoning level, so the
panel rounds it up to `high`. Since Luna's real price makes the whole corpus ~$7, the
~$4 saved by DeepSeek does not buy back a ~12× slower run.

This pipeline is the one text entry point that does **not** default to the shared
`DEFAULT_TEXT_MODEL_KEY`; every other one still does.

## Output

Summaries are saved as `.txt` files to `Summaries_FR_TXT/` and `Summaries_EN_TXT/`, keyed by item ID, then uploaded to the `bibo:shortDescription` field in Omeka S — the AI-summary property for articles and documents, exported to Hugging Face as `descriptionAI`.

Both land on that one property as two `@language`-tagged literals, `fr` and `en`, in a **single PATCH per item**:

```json
"bibo:shortDescription": [
  {"type": "literal", "@language": "fr", "@value": "Le Centre culturel islamique…",
   "@annotation": {"iwac:summaryModel": [{"value_resource_id": 79610}]}},
  {"type": "literal", "@language": "en", "@value": "The Centre culturel islamique…",
   "@annotation": {"iwac:summaryModel": [{"value_resource_id": 79610}]}}
]
```

> `dcterms:abstract` is a different field: it holds publisher/author abstracts on issues and scholarly references, and the HF `documents` subset exports it as a separate `abstract` column. Do not write generated summaries there.

Each summary carries an `iwac:summaryModel` value annotation naming the model that produced it, linked to its authority item (class 244, item set 267). One model produces both renderings, so both literals carry it. Step 03 prompts for the model, or takes `--model`:

```bash
python 03_omeka_update_summaries.py --model gpt-5.6-luna --dry-run
```

Available keys come from `AI_MODEL_ITEMS` in `common/iwac_config.py`. Add a new one there after creating its authority item in Omeka.

### Rollback

Every item's **pre-write JSON is appended to `backups/_pre_write_summaries_<timestamp>.jsonl` and flushed before its PATCH** — one object per line, only for items that actually change. An interrupted run therefore still has a complete record of everything it overwrote, which a buffered end-of-run dump would not. Restoring is a read of that file and a PATCH of each object back.

```bash
python 03_omeka_update_summaries.py --backup-dir /some/other/path
python 03_omeka_update_summaries.py --no-backup      # not recommended
```

### Legacy untagged summaries

The ~12,300 French summaries written before this pipeline became bilingual carry **no `@language` at all**. The French target sets `adopt_untagged=True`, so step 03 claims that existing literal and tags it `fr` on the way past, rather than appending a second French value beside it. The English target deliberately does not: an English write must never claim a value that predates this pipeline.

This only touches items you actually regenerate — step 03 writes what is in the two folders. Items never re-run keep their untagged French summary and gain nothing.

### Downstream: Hugging Face

[IWAC-Hugging-Face](https://github.com/fmadore/IWAC-Hugging-Face) exports the two literals as `descriptionAI` (fr) and `descriptionAI_en`, selecting one value per language rather than pipe-joining them.

The one constraint that falls on **this** pipeline: never write two literals of the same language on one item. The export takes the first and drops the rest, so a duplicate does not merge — it makes the exported summary depend on whatever order Omeka returned. That is what `adopt_untagged` on the French target prevents.

## Limitations

**Summarization is lossy**: Summaries compress information. Important details may be omitted, especially from long or complex documents.

**Hallucination risk**: AI may occasionally include information not present in the source text. Summaries are aids for discovery, not substitutes for reading originals.

**Language**: Summaries are generated in French and English regardless of source language. The prompt is written in French and assumes French input — the collection holds ~45 Ewé, Kabiyè and Dendi items for which a French-prompted model returns confident but unreliable output. Step 01 drops them by default (`--language` keeps only Français/Anglais); passing `--language` with no value disables that filter and puts them back in scope.

**Fidelity**: the prompt forbids adding any fact, place or date the source does not state — including the obvious ones. This is enforcement against a real failure: on a 338-character stub, GPT-5.6 Luna added the city "à Ouagadougou?", question mark included, inferring the organization's seat and flagging its own doubt inside the summary. Both the invented location and uncertainty markers are now explicitly prohibited. Spot-check short and OCR-degraded documents anyway.

**Written for discovery, not for RAG**: these summaries are never embedded — they are read by keyword search and by agents deciding which items are worth opening in full. That is why the prompt no longer says "keyword-rich, not narrative": the corpus shares its vocabulary (*islam*, *musulmans*, *imam*, country names), so a keyword-dense abstract is indistinguishable from forty others and triage collapses. The prompt asks instead for dense, concrete prose carrying the particulars — figures, decisions, named roles — and tells the model **not** to paraphrase the title, subject or spatial fields, which consumers already have beside the summary.

**Length is a budget.** Live summaries average **501 characters** (measured over 300 articles). The prompt targets 400–600 characters *per version* so that each language stands on its own within that, rather than the pair costing double wherever a summary is listed.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential

# At least one AI provider
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
MISTRAL_API_KEY=your_key
OPENROUTER_API_KEY=your_key  # Default DeepSeek model
```

## Customization

Edit `summary_prompt.md` to adjust:
- Summary languages
- Length and style
- What to emphasize (people, events, themes)
- Fidelity rules

The `{text}` placeholder is replaced with document content at runtime; everything before it becomes the system prompt. Its hash is part of the checkpoint, so editing the prompt invalidates existing output — that is deliberate, and `--force` is how you accept it.

Output *shape* is not set here: it comes from the `BilingualSummary` Pydantic schema in step 02, sent through `generate_structured()`. Adding a third language means adding a field there, a folder, and a `PropertyTarget` in step 03.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty summaries | Check source text exists in `TXT/` folder |
| Missing prompt error | Ensure `summary_prompt.md` exists in script directory |
| API authentication | Verify correct API key in `.env` |
| Items not updating | Check Omeka S credentials have write access |
| "no English summary — French only" | Step 02 failed on those items; re-run it before step 03 |
| Item ends up with two French values | The French target lost `adopt_untagged`; check `PropertyTarget` in step 03 |
| Checkpoint provenance error | Expected after editing the prompt or switching model — `--force` to accept |
