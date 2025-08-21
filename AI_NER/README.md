# AI Named Entity Recognition (NER) Module

This module provides a complete Named Entity Recognition pipeline for Omeka S Collections. A **single unified script** (`01_NER_AI.py`) now supports both **Google Gemini** and **OpenAI ChatGPT (gpt-5-mini)**. You can select the provider interactively (prompt 1 or 2) or via the `--model` flag. The remainder of the pipeline (reconciliation + update) is model-agnostic and consumes the CSV produced by the unified extraction script. The pipeline consists of three sequential stages that extract, reconcile, and update named entities in the IWAC (Islam West Africa Collection) database.

## Pipeline Overview

The NER module operates as a three-stage pipeline:

1. **Entity Extraction** - Extract named entities from Omeka S collection items using AI
2. **Authority Reconciliation** - Match extracted entities against existing authority records
3. **Database Update** - Update Omeka S items with reconciled entity links

## Files

### Core Scripts
- `01_NER_AI.py` - **Entity Extraction (Gemini or OpenAI)**: Unified script that extracts named entities (persons, organizations, locations, subjects) from Omeka S collection items. Choose provider interactively or via `--model gemini|openai`.
- `02_NER_reconciliation_Omeka.py` - **Authority Reconciliation**: Matches AI-extracted entities against existing Omeka S authority records (spatial coverage, subjects)
- `03_Omeka_update.py` - **Database Update**: Updates Omeka S items with reconciled entity links, preserving existing data and avoiding duplicates

### Configuration
- `ner_system_prompt.md` - **NER System Prompt**: Detailed French-language prompt template defining extraction rules for the IWAC collection context

## Features

### Script 1: Unified Entity Extraction `01_NER_AI.py`
- Supports both Google Gemini and OpenAI ChatGPT via flag or interactive selection
- Asynchronous and synchronous modes
- Configurable batch size & concurrency
- Rate limiting, retries (tenacity), and progress statistics
- Single shared French prompt (`ner_system_prompt.md`) for consistent extraction rules
- Spatial coverage filtering (removes collection-wide spatial term from per-item results)
- Produces provider-specific output filename suffix (e.g., `_processed_openai.csv`, `_processed_gemini.csv`)
- OpenAI path preserves required Responses API settings block
- Gemini path requests JSON-style output (parsed with fallback regex)

### Script 2: Authority Reconciliation (`02_NER_reconciliation_Omeka.py`)
- Multi-stage reconciliation process (spatial → combined subject+topic)
- Authority dictionary building from multiple item sets
- Ambiguous term detection & export (values mapping to >1 authority ID are NOT auto-linked)
- Unreconciled entity tracking and export
- Conservative fuzzy suggestion engine (token overlap + similarity thresholds)
- Configurable tuning constants to control strictness & volume of suggestions
- Comprehensive matching statistics
- Iterative update preserves original CSV columns

### Script 3: Database Update (`03_Omeka_update.py`)
- Safe item updating with duplicate prevention
- Preserves existing metadata and relationships
- Processes reconciled CSV output from script 2
- Comprehensive error handling and logging
- Updates both spatial coverage (dcterms:spatial) and subjects (dcterms:subject)

## NER System Prompt

The NER processing uses a sophisticated modular prompt system where detailed extraction instructions are stored in `ner_system_prompt.md`. This French-language prompt is specifically designed for the IWAC (Islam West Africa Collection) context.

### Prompt Benefits
- Easy modification of extraction rules without changing code
- Better version control for prompt changes
- Collaborative editing of prompts
- Clear separation of concerns between logic and instructions
- Context-specific rules for West African Islamic collections

### Prompt Structure

The prompt file contains detailed specifications for the IWAC collection:

1. **Context Definition**: Specific to Islam and Muslims in Burkina Faso, Benin, Niger, Nigeria, Togo, and Côte d'Ivoire
2. **Four Entity Categories** with precise formatting rules:
   - **Personnes (Persons)**: Full names without titles (religious, civil, honorific)
   - **Organisations**: Full organization names without acronyms, excluding parishes
   - **Lieux (Places)**: Simplified geographical names without political qualifiers
   - **Sujets (Subjects)**: Simple thematic keywords (max 8 per text) covering religion, society, politics, economy
3. **Extensive Examples**: Specific to West African context and Islamic terminology
4. **Output Format**: JSON structure specification
5. **Text Placeholder**: `{text_content}` for dynamic content insertion

### Subject Categories Covered
- **Religion**: islam, fiqh, sharia, hajj, ramadan, salafisme, soufisme, confrérie, etc.
- **Society**: mariage, éducation, famille, jeunesse, femmes, etc.
- **Politics**: laïcité, démocratie, élections, radicalisation, etc.
- **Economy**: commerce, agriculture, développement, etc.

### Modifying the Prompt

To modify the NER extraction behavior:
1. Edit the `ner_system_prompt.md` file
2. Update the formatting rules, examples, or subject categories as needed
3. The changes will be automatically picked up on the next script run

## Workflow

### Choosing the NER Provider

You now select the provider inside the unified script:

Interactive (script will prompt):
```bash
python 01_NER_AI.py --item-set-id 123
```

Explicit provider flag:
```bash
python 01_NER_AI.py --item-set-id 123 --model gemini --async --batch-size 20
python 01_NER_AI.py --item-set-id 123 --model openai --async --batch-size 20
```

Comparison:

| Aspect | Gemini (via unified) | OpenAI (via unified) |
|--------|----------------------|----------------------|
| Model name | `gemini-2.5-flash-preview-05-20` | `gpt-5-mini` |
| Env var | `GEMINI_API_KEY` | `OPENAI_API_KEY` |
| Dependency | `google-genai` | `openai` |
| Output filename suffix | `_processed_gemini.csv` | `_processed_openai.csv` |
| Prompt | Shared `ner_system_prompt.md` | Shared `ner_system_prompt.md` |

### Complete Pipeline Execution

1. **Run Entity Extraction (unified)**:
   ```bash
   # interactive model choice
   python 01_NER_AI.py --item-set-id 123

   # or specify provider
   python 01_NER_AI.py --item-set-id 123 --model gemini --async --batch-size 20
   python 01_NER_AI.py --item-set-id 123 --model openai --async --batch-size 20
   ```
   Output filename pattern: `item_set_<ID>_processed_<provider>.csv`

2. **Run Authority Reconciliation**:
   ```bash
   python 02_NER_reconciliation_Omeka.py
   ```
   Automatically finds the latest provider CSV.

3. **Update Database**:
   ```bash
   python 03_Omeka_update.py
   ```
   Applies reconciled entities to Omeka S items.

### Output Files

- **After Script 1**: `item_set_<ID>_processed_<provider>.csv` (e.g., `_processed_openai.csv`, `_processed_gemini.csv`)
- **After Script 2** (reconciliation): 
   - `*_reconciled.csv` – Main CSV augmented with `Spatial AI Reconciled ID` & `Subject AI Reconciled ID`
   - `*_unreconciled_spatial.csv` – Unmatched spatial values + counts
   - `*_unreconciled_subject_and_topic.csv` – Unmatched subject/topic values + counts
   - `*_ambiguous_authorities_spatial.csv` – Spatial terms that map to multiple authority records
   - `*_ambiguous_authorities_subject_and_topic.csv` – Subject/topic ambiguous terms
   - `*_potential_reconciliation_spatial.csv` – (Optional) Suggested spatial matches above thresholds
   - `*_potential_reconciliation_subject_and_topic.csv` – Suggested subject/topic matches
- **After Script 3**: Database updated (no new files created)

> Note: The suggestion CSVs are intentionally small after recent tightening (from tens of thousands down to a few hundred). Further tuning instructions below.

### Reconciliation Tuning (Script 2)

The reconciliation script exposes **tunable constants** near the top of `02_NER_reconciliation_Omeka.py` so you can adjust recall vs precision without diving into function bodies.

| Constant | Purpose | Typical Effect When Increased |
|----------|---------|--------------------------------|
| `BASE_MIN_SIMILARITY` | Minimum similarity for single-word values | Fewer suggestions overall |
| `MULTI_WORD_MIN_SIMILARITY` | Baseline similarity for multi-word values | Stricter multi-token matching |
| `STRONG_MATCH_THRESHOLD` | Threshold considered a “very strong” match (narrows band) | Narrows kept candidate band |
| `MIN_TOKEN_OVERLAP` | Jaccard-like required overlap after stopword removal | Removes matches sharing only generic tokens |
| `DEFAULT_MAX_CANDIDATES` | Max suggestions per unreconciled value | Caps review workload |
| `GENERIC_TERMS` | Set of broad words that only allow near-exact matches | Blocks noisy generic suggestions |
| `STOPWORDS` | Tokens excluded before overlap calc | Adjust if domain terms wrongly discarded |

Core heuristics now in place:
- Require token overlap (after stopword & diacritic normalization) for multi-word comparisons.
- Generic / thematic vocabulary (e.g., *religion*, *transport*, *formation*) is suppressed unless near-exact.
- Only the best variant per authority item is kept (deduplicated by item ID).
- A narrow score band (±0.02–0.03) retains only candidates nearly as good as the top one.
- Default: **1 suggestion per unreconciled value** to minimize triage effort.

#### Common Adjustments

| Goal | Recommended Change |
|------|--------------------|
| Even fewer suggestions | Raise `BASE_MIN_SIMILARITY` to 0.85–0.90; raise `MULTI_WORD_MIN_SIMILARITY` to 0.92–0.95 |
| Allow more variants | Lower `MIN_TOKEN_OVERLAP` to 0.4 and raise `DEFAULT_MAX_CANDIDATES` to 2–3 |
| Eliminate generic one-word suggestions | Add them to `GENERIC_TERMS` or set their branch threshold to 1.0 |
| Too many near-identical org names | Raise `STRONG_MATCH_THRESHOLD` to 0.95 and keep `band` narrow |
| Missing true matches due to token filtering | Remove that token from `STOPWORDS` |

#### Adding a Frequency Gate (Optional)
If you only want suggestions for unreconciled terms appearing at least N times, you can insert a check in `create_potential_reconciliation_csv` after reading the `Count` column:

```python
MIN_COUNT_FOR_SUGGESTION = 3
if int(count) < MIN_COUNT_FOR_SUGGESTION:
      continue
```

#### Disabling Suggestions Entirely
Set `DEFAULT_MAX_CANDIDATES = 0` (or short‑circuit `create_potential_reconciliation_csv` early) to skip producing the potential reconciliation CSVs.

### Ambiguous Terms Handling

Any value that resolves to **multiple authority IDs** (same normalized forms) is written to the `*_ambiguous_authorities_*.csv` files and **not** auto-reconciled. These should be manually reviewed—typically by refining authority records (adding/clarifying alternative titles) or pruning duplicates.

## Requirements

### Environment Setup
- Python 3.7+
- Dependencies listed in `requirements.txt`
- Environment variables (choose provider-specific key):
   - `OMEKA_BASE_URL` - Base URL for Omeka S instance
   - `OMEKA_KEY_IDENTITY` - Omeka S API key identity  
   - `OMEKA_KEY_CREDENTIAL` - Omeka S API key credential
   - `GEMINI_API_KEY` (if selecting Gemini)
   - `OPENAI_API_KEY` (if selecting OpenAI)

### Required Files
- `ner_system_prompt.md` - Shared by both extraction scripts
- `.env` file with whichever API key you are using

### Authority Item Sets (for Script 2)
The reconciliation script expects specific Omeka S item sets:
- **Spatial authorities**: Item set 268 (spatial coverage terms)
- **Subject authorities**: Item sets 854, 2, 266 (subject/topic terms) — combined during processing with topic set(s)

## Notes

- Unified extraction simplifies maintenance (legacy `01_NER_AI_Gemini.py` and `01_NER_AI_ChatGPT.py` can be removed)
- Comprehensive logging, retries, and rate limiting remain
- Batch processing & async improve throughput
- French language support throughout
- Designed specifically for Islamic studies collections in West Africa
