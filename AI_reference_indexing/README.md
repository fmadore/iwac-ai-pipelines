# AI Reference Indexing

Assign controlled subject (`dcterms:subject`) and spatial (`dcterms:spatial`) keywords to IWAC scholarly references by analyzing their full text (`bibo:content`). Keywords are reconciled against existing authority records and linked as Omeka S resource references.

## Scope

This pipeline targets **references** — book chapters, journal articles, reports, and other scholarly publications. It is not designed for newspaper articles (which use the NER pipeline) or other item types.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_fetch_references.py` | Download items from Omeka and export authority indices |
| 2 | Claude agent / skill | Analyze text and assign Subject AI + Spatial AI keywords |
| 3 | `03_reconcile_metadata.py` | Fuzzy-match keywords against authority records |
| 4 | `04_create_index_items.py` | Batch-create new authority items (optional) |
| 5 | `05_update_omeka.py` | Update Omeka items with reconciled resource links |

Step 2 is performed by the `reference-indexing` Claude agent (`.claude/agents/reference-indexing.md`), which reads the document text, applies the enrichment prompt guidelines, and writes an enriched CSV. All keywords are assigned **in French** regardless of document language.

## Usage

### Full pipeline via Claude agent

The recommended workflow is to invoke the `enrich-metadata` agent, which orchestrates all steps:

```
Use the enrich-metadata agent to process item set 78405
```

The agent will:
1. Ask for the item set ID (if not provided)
2. Run Step 1 (or reuse existing files)
3. Perform AI enrichment (Step 2)
4. Run reconciliation (Step 3)
5. Report unreconciled terms and ask about Step 4
6. Confirm before updating Omeka (Step 5)

### Manual step-by-step

```bash
# Step 1: Fetch references and authority indices
python 01_fetch_references.py --item-set-id 78405

# Step 2: Run the enrich-metadata skill in Claude Code

# Step 3: Reconcile keywords against authorities
python 03_reconcile_metadata.py

# Step 4 (optional): Create new authority items from unreconciled terms
python 04_create_index_items.py \
    --input-csv output/items_enriched_..._unreconciled_subject.csv \
    --type subject

# Step 5: Update Omeka with reconciled links
python 05_update_omeka.py
```

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

## Keyword Assignment Rules

Defined in `02_enrichment_prompt.md`:

- **Subject AI**: 5–8 thematic keywords per document in French
- **Spatial AI**: geographic locations mentioned or implied
- Prefer existing index terms; new terms are allowed when no match exists
- Avoid `Islam` and `Musulmans` (too generic for the IWAC context)
- Persons: full name without titles; Organizations: full name without acronyms
- Spatial: cities before countries, no continents, standardized names
- Geographic features (parks, regions) belong in Spatial AI, not Subject AI

## Output Files

| File | Description |
|------|-------------|
| `items_{ids}_{date}.csv` | Items with full text from Step 1 |
| `index_subject.csv` | Subject authority index |
| `index_spatial.csv` | Spatial authority index |
| `items_enriched_{timestamp}.csv` | Items with Subject AI + Spatial AI columns |
| `keyword_summary_{timestamp}.csv` | Unique terms with counts and index match status |
| `*_reconciled.csv` | Enriched CSV with reconciled Omeka IDs |
| `*_unreconciled_*.csv` | Terms not found in authority records |
| `*_potential_reconciliation_*.csv` | Fuzzy-match suggestions for unreconciled terms |
| `newly_created_items_*.csv` | Mapping of newly created authority items |
