# Named Entity Recognition (NER)

Extract people, places, organizations, and subjects from Omeka S collection items using AI, then reconcile them against existing authority records.

## Why This Tool?

Manual entity tagging at scale is impractical. A single newspaper archive may mention thousands of people and places. This pipeline automates extraction while connecting entities to controlled vocabularies—enabling faceted search and network analysis across the collection.

## How It Works

```
Omeka S items → AI entity extraction → Authority reconciliation → Database update
```

1. **Extract** (`01_NER_AI.py`): AI identifies persons, organizations, locations, and subjects
2. **Reconcile** (`02_NER_reconciliation_Omeka.py`): Match entities against existing authority records
3. **Update** (`03_Omeka_update.py`): Link reconciled entities back to Omeka S items

## Quick Start

```bash
# Extract entities from an item set
python 01_NER_AI.py --item-set-id 123  # DeepSeek V4 Flash 0731 by default

# Or use Google's open-weights flagship
python 01_NER_AI.py --item-set-id 123 --model gemma-4

# Match against authority records (newest CSV in output/, or --input)
python 02_NER_reconciliation_Omeka.py

# Preview the Omeka writes — no PATCH is sent
python 03_Omeka_update.py --dry-run

# Update Omeka S database (asks before the first write)
python 03_Omeka_update.py
```

Step 3 writes to the live archive. It reports what would change under
`--dry-run`, dumps every pre-write payload to `output/_pre_write_ner_links_*.json`,
and asks for confirmation before the first PATCH. `--yes` skips the prompt for
unattended runs; `--input` applies a specific CSV instead of the newest one.

Every link step 3 adds carries an `iwac:nerModel` value annotation naming the
model that extracted the entity, so AI-assigned subjects and places can be told
apart from hand-catalogued ones. The model is read from the checkpoint step 1
left beside its CSV; `--model` overrides it, and the script asks when neither
is available (for instance after a run on `gemma-4`, whose Gemini route has no
authority item). Links already on an item are never re-stamped.

The extraction CSV is resumable. Each completed row is flushed immediately,
and a sidecar checkpoint records the exact model, prompt, item-set scope, and
spatial filter. Re-running skips IDs already present in a compatible CSV and
retries failed items. The pipeline refuses to append to output with missing or
different provenance; use `--force` to replace it deliberately.

## Supported Models

| Model | Provider | Speed | Cost |
|-------|----------|-------|------|
| `gemini-3.7-flash` | Google | Fast | Low |
| `gemma-4` | Google (Gemma 4 31B, open-weights via Gemini API) | Fast | Low |
| `gpt-5.6-luna` | OpenAI | Fast | Low |
| `mistral-large` | Mistral | Medium | Medium |
| `ministral-14b` | Mistral | Fast | Low |
| `mistral-small` | Mistral (hybrid reasoning) | Fast | Low |
| `qwen3.5-moe` | Alibaba (open weights, Apache-2.0, via OpenRouter) | Fast | Lowest |
| `deepseek-v4-flash-0731` | DeepSeek (open weights via OpenRouter) | Fast | Lowest (default) |

All models use the same French-language prompt (`ner_system_prompt.md`) optimized for West African Islamic contexts.

`gemma-4` uses the same `GEMINI_API_KEY` as the Gemini models. Thinking level is `minimal` by default (Gemma 4 accepts only `MINIMAL` or `HIGH`), which matches the low-cost entity-extraction budget used by `gemini-3.7-flash`.

`qwen3.5-moe` and `deepseek-v4-flash-0731` share one `OPENROUTER_API_KEY` and cost
roughly a tenth of `gpt-5.6-luna`, which is what makes a full-corpus pass
affordable. Qwen accepts the NER pipeline's `medium` reasoning request directly;
DeepSeek 0731 supports only `low`, `high`, and `max`, so the shared adapter uses
its cost-conscious `low` default for NER.
Because these are open models behind a router, treat a first run as an
evaluation — compare its entities against a known-good model on the same item
set before committing to a bulk pass.

## Entity Categories

The pipeline extracts four entity types:

| Category | Examples | Formatting Rules |
|----------|----------|------------------|
| **Persons** | Amadou Hampâté Bâ | Full names, no titles (Sheikh, Dr.) |
| **Organizations** | Union Musulmane du Togo | Full names, no acronyms |
| **Locations** | Ouagadougou, Zinder | Simple names, no political qualifiers |
| **Subjects** | islam, hajj, laïcité | Thematic keywords (max 8 per text) |

## Output Files

After extraction:
- `item_set_<ID>_processed_<model>.csv` — Extracted entities per item
- `item_set_<ID>_processed_<model>.csv.checkpoint.json` — Model, prompt and scope the CSV was made with

After reconciliation:
- `*_reconciled.csv` — Entities matched to authority IDs
- `*_unreconciled_*.csv` — Entities needing manual review
- `*_potential_reconciliation_*.csv` — Fuzzy-match suggestions for the unreconciled terms
- `*_ambiguous_*.csv` — Terms matching multiple authorities

After the Omeka update:
- `_pre_write_ner_links_*.json` — Pre-write payload dump, the only route back

## Validating Extracted Entities

Before reconciliation, you may want to manually review the AI-extracted entities for accuracy. Use the [AI-NER-Validator](https://github.com/fmadore/AI-NER-Validator) web application to:

- Review articles with highlighted entities side by side
- Validate or reject each entity with a single click
- Add missing entities manually
- Export a clean CSV with only validated entities

This step is optional but recommended for quality control, especially when working with new document types or testing different AI models.

## Limitations

**Entity boundaries**: AI may split compound names incorrectly or merge separate entities. Names like "Cheikh Amadou Bamba" may lose the honorific as intended, or incorrectly.

**Authority matching**: Fuzzy matching catches spelling variants but may miss entities with very different surface forms (nicknames, transliterations).

**Ambiguous terms**: Common terms like "Union" or "Association" may match multiple authority records. These are flagged for manual review rather than auto-linked.

**West African focus**: The prompt is optimized for Francophone West African Islamic contexts. Other regions may need prompt adjustments.

**Hallucinated entities**: AI may occasionally extract entities not present in the source text. Cross-reference against originals for critical work.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential

# The default model (DeepSeek V4 Flash 0731) is reached through OpenRouter
OPENROUTER_API_KEY=your_key
# Other --model choices need their own provider key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
MISTRAL_API_KEY=your_key
```

## Reconciliation Tuning

The reconciliation script uses conservative defaults to minimize false matches. To adjust:

| Goal | Change |
|------|--------|
| Fewer suggestions | Raise similarity thresholds in script constants |
| More suggestions | Lower `MIN_TOKEN_OVERLAP`, raise `DEFAULT_MAX_CANDIDATES` |
| Skip suggestions entirely | Set `DEFAULT_MAX_CANDIDATES = 0` |

The constants live in `common/reconciliation.py` (`BASE_MIN_SIMILARITY`,
`MIN_TOKEN_OVERLAP`, `DEFAULT_MAX_CANDIDATES`), shared with the
reference-indexing pipeline so both reconcile the same way. Steps 2 and 3 are
entry points for `common/reconciliation_cli.py` and `common/link_update_cli.py`;
the reference-indexing pipeline's steps 3 and 5 run the same code.

## Customization

Edit `ner_system_prompt.md` to modify:
- Entity categories and formatting rules
- Subject vocabulary (religion, society, politics, economy)
- Language-specific examples
- Output format

Changes take effect on the next extraction run.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty extractions | Check item has `bibo:content` field with text |
| API errors | Verify API key and quota in `.env` |
| Too many unreconciled | Review `*_unreconciled_*.csv`, add missing authorities |
| Wrong entity splits | Adjust examples in `ner_system_prompt.md` |
