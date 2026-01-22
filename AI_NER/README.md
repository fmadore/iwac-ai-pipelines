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
python 01_NER_AI.py --item-set-id 123 --model gemini-flash

# Match against authority records
python 02_NER_reconciliation_Omeka.py

# Update Omeka S database
python 03_Omeka_update.py
```

## Supported Models

| Model | Provider | Speed | Cost |
|-------|----------|-------|------|
| `gemini-flash` | Google | Fast | Low |
| `gpt-5-mini` | OpenAI | Fast | Low |
| `mistral-large` | Mistral | Medium | Medium |
| `ministral-14b` | Mistral | Fast | Low |

All models use the same French-language prompt (`ner_system_prompt.md`) optimized for West African Islamic contexts.

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

After reconciliation:
- `*_reconciled.csv` — Entities matched to authority IDs
- `*_unreconciled_*.csv` — Entities needing manual review
- `*_ambiguous_*.csv` — Terms matching multiple authorities

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

# At least one AI provider
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

See tuning constants at the top of `02_NER_reconciliation_Omeka.py`.

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
