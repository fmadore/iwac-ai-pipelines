# AI Reference Indexing

Assign controlled subject (`dcterms:subject`) and spatial (`dcterms:spatial`) keywords to IWAC scholarly references by analysing their full text (`bibo:content`). Keywords are reconciled against existing authority records and linked as Omeka S resource references, each link annotated with the model that proposed it.

## Scope

This pipeline targets **references** — book chapters, journal articles, reports, and other scholarly publications. It is not designed for newspaper articles (which use the NER pipeline) or other item types.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_fetch_references.py` | Download items from Omeka and export authority indices |
| 2 | `02_enrich_references.py` | Read each text with an LLM and assign Subject AI + Spatial AI keywords |
| 3 | `03_reconcile_metadata.py` | Fuzzy-match keywords against authority records |
| 4 | `04_create_index_items.py` | Batch-create new authority items (optional) |
| 5 | `05_update_omeka.py` | Update Omeka items with reconciled resource links |

Step 2 applies the rules in `02_enrichment_prompt.md` through the shared model registry (`common/llm_provider.py`), with structured output so every answer is two keyword lists rather than free text. All keywords are assigned **in French** regardless of document language. Steps 3 and 5 are entry points for `common/reconciliation_cli.py` and `common/link_update_cli.py`, which the NER pipeline's steps 2 and 3 share.

## Usage

```bash
# Step 1: Fetch references and authority indices
python 01_fetch_references.py --item-set-id 78405

# Step 2: Assign keywords (newest items_*.csv, shared default text model)
python 02_enrich_references.py
python 02_enrich_references.py --model gpt-5.6-luna      # another registry model
python 02_enrich_references.py --reindex                 # also items that already have both link sets

# Step 3: Reconcile keywords against authorities (newest items_enriched_*.csv, or --input)
python 03_reconcile_metadata.py

# Step 4 (optional): Create new authority items from unreconciled terms
python 04_create_index_items.py \
    --input-csv output/items_enriched_..._unreconciled_subject.csv \
    --type subject

# Step 5: Update Omeka with reconciled links (preview first)
python 05_update_omeka.py --dry-run
python 05_update_omeka.py
```

### What step 2 does with existing keywords

Items with no `bibo:content` get empty keyword columns. Items that already carry **both** subject and spatial links are skipped unless `--reindex` is passed. Items with only one of the two are processed, and their existing links are resolved to names through `index_subject.csv` / `index_spatial.csv` and shown to the model, so new keywords complement rather than repeat them.

Output is durable and resumable: each row is flushed as it is written, and a `.checkpoint.json` beside the CSV records the model, prompt and input. Re-running continues where it stopped; a run with a different model or prompt refuses to append to the old file unless `--force` is passed.

### Reviewing fuzzy matches

Step 3 writes `*_potential_reconciliation_*.csv` with near-matches for unreconciled terms (accents, spelling). Correct the term in the enriched CSV to the authority's spelling and re-run step 3; what remains is genuinely new and is what step 4 offers to create.

### Provenance

Step 5 stamps every link it adds with an `iwac:nerModel` value annotation naming the model that proposed the keyword, so AI-assigned subjects can be told apart from hand-catalogued ones. The model is read from the checkpoint step 2 left beside its CSV; pass `--model` to override, or answer the prompt when no checkpoint is found. Links already on an item are never re-stamped.

## Authority Item Sets

When **creating** new authority items, the `--type` parameter routes to the correct item set, resource template, and resource class:

| `--type` | Item Set | Template | Class | Use for |
|----------|----------|----------|-------|---------|
| `subject` | 1 (Sujets) | 3 | 244 | Topics/themes |
| `spatial` | 268 (Emplacements) | 6 | 9 | Locations |
| `association` | 854 (Associations) | 7 | 96 | Organizations |
| `individu` | 266 (Individus) | 5 | 94 | People |
| `event` | 2 (Événements) | 2 | 54 | Events |

All created items automatically include `dcterms:type` → "Notice d'autorité" (linked item 67568).

For **reconciliation**, the subject index is built from item sets 1, 2, 266, and 854 combined.

Some terms the model files as subjects are really places — parks (`Parc national du W`), regions (`Sahel`). Check the unreconciled subject list against the spatial index before creating them with `--type subject`.

## Keyword Assignment Rules

Defined in `02_enrichment_prompt.md`:

- **Subject AI**: 5–8 thematic keywords per document in French
- **Spatial AI**: geographic locations mentioned or implied
- Prefer existing index terms; new terms are allowed when no match exists
- Avoid `Islam` and `Musulmans` (too generic for the IWAC context) — the script drops them again if the model ignores the rule
- Persons: full name without titles; Organizations: full name without acronyms
- Spatial: cities before countries, no continents, standardized names
- Geographic features (parks, regions) belong in Spatial AI, not Subject AI

## Output Files

| File | Description |
|------|-------------|
| `items_{ids}_{date}.csv` | Items with full text from Step 1 |
| `index_subject.csv` | Subject authority index (id, title, alternatives) |
| `index_spatial.csv` | Spatial authority index |
| `items_enriched_{ids}_{date}.csv` | Items with Subject AI + Spatial AI columns |
| `items_enriched_*.csv.checkpoint.json` | Model, prompt and input the enriched file was made with |
| `keyword_summary_{timestamp}.csv` | Unique terms with type and count |
| `*_reconciled.csv` | Enriched CSV with reconciled Omeka IDs |
| `*_unreconciled_*.csv` | Terms not found in authority records |
| `*_potential_reconciliation_*.csv` | Fuzzy-match suggestions for unreconciled terms |
| `newly_created_items_*.csv` | Mapping of newly created authority items |
| `_pre_write_reference_links_*.json` | Pre-write payload dump from Step 5 — the only route back |

Everything lives in `output/`, which is git-ignored; delete it once step 5 has run.

## Configuration

```bash
OMEKA_BASE_URL=https://your-instance.org/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
OPENROUTER_API_KEY=your_key    # the default text model (DeepSeek V4 Flash 0731)
# or GEMINI_API_KEY / OPENAI_API_KEY / MISTRAL_API_KEY for another --model
```
