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
3. **Check in context** (`02b_context_check.py`, optional): a model reads each article and keeps, drops or settles every link
4. **Update** (`03_Omeka_update.py`): Link reconciled entities back to Omeka S items

## Quick Start

```bash
# Extract entities from an item set
python 01_NER_AI.py --item-set-id 123  # DeepSeek V4 Flash 0731 by default

# Or Google's open-weights Gemma 4, through OpenRouter
python 01_NER_AI.py --item-set-id 123 --model gemma-4-openrouter

# Or every press article from one country's newspapers
python 01_NER_AI.py --country "Burkina Faso" --model claude-haiku-5-5 --async

# Match against authority records (newest CSV in output/, or --input)
python 02_NER_reconciliation_Omeka.py

# Optional: check every link against its article (a 100-article pilot first)
python 02b_context_check.py --input output/x_reconciled.csv --limit 100
python 02b_context_check.py --input output/x_reconciled.csv

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
is available. Links already on an item are never re-stamped.

The extraction CSV is resumable. Each completed row is flushed immediately,
and a sidecar checkpoint records the exact model, prompt, item-set scope, and
spatial filter. Re-running skips IDs already present in a compatible CSV and
retries failed items. The pipeline refuses to append to output with missing or
different provenance; use `--force` to replace it deliberately.

## Supported Models

| Model | Provider | Speed | Cost |
|-------|----------|-------|------|
| `gemini-3.7-flash` | Google | Fast | Low |
| `gemma-4-openrouter` | Google (Gemma 4 31B, open weights via OpenRouter) | Slow | Low |
| `gpt-6-luna` | OpenAI | Fast | Low |
| `mistral-small` | Mistral (hybrid reasoning) | Fast | Low |
| `qwen3.5-moe` | Alibaba (open weights, Apache-2.0, via OpenRouter) | Fast | Lowest |
| `deepseek-v4-flash-0731` | DeepSeek (open weights via OpenRouter) | Fast | Lowest (default) |
| `claude-haiku-5-5` | Anthropic (Claude Haiku 5.5) | Fast | Low |

All models use the same French-language prompt (`ner_system_prompt.md`) optimized for West African Islamic contexts.

Every model offered has an Omeka authority item, so step 3 can stamp it.
Gemma is offered only on its OpenRouter route: Google serves it on the Gemini
API only on a free tier whose content it uses. Gemma 4 has two reasoning levels,
`minimal` and `high`, so the NER pipeline's `medium` request is sent as `high`.

`gemma-4-openrouter`, `qwen3.5-moe` and `deepseek-v4-flash-0731` share one `OPENROUTER_API_KEY`. Compare current rates and measured token usage
when budgeting a full-corpus pass. Qwen accepts the NER pipeline's `medium` reasoning request directly;
DeepSeek 0731 supports only `low`, `high`, and `max`, so the shared adapter uses
its cost-conscious `low` default for NER.
Because these are open models behind a router, treat a first run as an
evaluation — compare its entities against a known-good model on the same item
set before committing to a bulk pass.

`claude-haiku-5-5` needs `ANTHROPIC_API_KEY` and receives the `medium` request
as its effort level, which is also its default. On 2026-10-20 it ran over
the 8,045 Burkinabè articles in 1 h 38 with `--async` and no failure: 35.5M
tokens in and 5.9M out (4.4M of them reasoning), about $6.50.

`--country` replaces `--item-set-id` with a country name, matched without regard
to case or accents (`benin` finds `Bénin`). It selects the item sets whose
`dcterms:spatial` links that country — its newspapers, and also its photograph,
reference, document and YouTube sets — and keeps only their press articles
(class 36). A newspaper's country is therefore the article's, as in the
`articles.country` column of the Hugging Face export; an article from a
Burkinabè paper about Mali is in, a Malian article about Burkina is not.

## Entity Categories

The pipeline extracts four entity types:

| Category | Examples | Formatting Rules |
|----------|----------|------------------|
| **Persons** | Amadou Hampâté Bâ | Full names, no titles (Sheikh, Dr.), no scriptural figures cited in a sermon |
| **Organizations** | Union Musulmane du Togo | Full names, no acronyms; not the outlet or agency an article is sourced from |
| **Locations** | Ouagadougou, Zinder | Simple names, no political qualifiers, no bare compass points |
| **Subjects** | Hadj, laïcité, dialogue interreligieux | 1–6 themes of the kind the subject vocabulary (item set 1) holds; never "islam", "religion" or catch-all nouns |

Every entity is then recased by `clean_entity()`: an all-lowercase word is
capitalised, a word the model already capitalised is kept, so acronyms
(`UEMOA`) survive; each piece of a hyphenated word is capitalised
(`Ki-Zerbo`), particles and elisions stay lowercase inside a name (`Côte
d'Ivoire`), and in person names a surname in capitals (`TRAORÉ`) is recased.

## Output Files

After extraction:
- `item_set_<ID>_processed_<model>.csv` — Extracted entities per item
  (`country_<country>_articles_processed_<model>.csv` with `--country`)
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

## Checking links in context

Step 02 matches by name alone, so a shared name is taken for one entity: three
persons called Abdoulaye Ouédraogo, six countries' health ministries, a
village and the clashes named after it. `02b_context_check.py` sends the model
one call per article with its text and every entry step 02 produced — the
match it made, the candidates of an ambiguous term, the fuzzy candidates of an
unreconciled one — each with the record's type, country and description. The
model returns, per entry, one of the offered ids or none; any other id is
refused. The rules it follows are `context_check_prompt.md`.

It writes `*_reconciled_checked.csv` (same columns; pass it to `03` with
`--input`, and its provenance still traces to the NER checkpoint) and
`*_context_decisions.csv`, one row per entry with the model's reason. With
`--limit` the checked CSV holds only the sample, so a pilot cannot be written
in place of the full run. Decisions are journalled, so a re-run resumes.

The 100-article pilot of 2026-10-21 (Burkinabè run, Haiku 5.5): of 950
matches 930 kept and 20 dropped — "Foi" the word read as the Front de la Oumma
Islamique, the Mouhoun river as the region, Kuwait's foreign ministry as
Burkina's, bylines, a hallucinated "Niger", themes the text never treats —
and 7 fuzzy terms linked (EIGS, Balai Citoyen, Médine). About $0.08 and one
minute per 100 articles with 8 workers. Generic themes (Santé, Sécurité) are
the decisions that vary between runs.

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
reference-indexing pipeline. Steps 2 and 3 are entry points for
`common/reconciliation_cli.py` and `common/link_update_cli.py`; the
reference-indexing pipeline's steps 3 and 5 run the same code.

NER alone reconciles with `extended_matching`. After the exact keys miss, a
term may still match on its words in another order ("Kassoum Coulibaly" →
"Coulibaly Kassoum", never for initials), and a subject other than a person
on a trailing s/x ("Confrérie" → "Confréries", accents kept). A subject term that names a place
in set 268 ("Université de Ouagadougou") is written to the spatial column, as
the catalogue links places.

An acronym name (2–6 capitals, dots allowed) matches only a term written in
capitals: "FOI" reaches the Front de la Oumma Islamique, "Foi" and "foi" do
not. Twelve of the archive's acronyms are common French words (UN, AU, CAR,
CET, FOI, AIDE, AIT, UNIR…), and the word "foi" had been linked to the FOI 317
times. The rule relies on `01`'s `clean_entity()`, which keeps capitals since
2026-10-20; output made before then title-cased its acronyms ("Cedeao",
"Cerfi"), so reconciling it again would lose them — 995 matches on the
Burkinabè run, 736 of which the context check had kept.

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

## Publication workflow

See the [pipeline contract](../docs/PIPELINE_CONTRACTS.md) for inputs, outputs,
resume identity, incomplete-output handling and Omeka write semantics, and the
[publication guide](../docs/PUBLICATION.md) for legacy-artifact migration and
release validation.
