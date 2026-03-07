---
name: reference-indexing
description: Assign controlled subject and spatial keywords to IWAC scholarly references (book chapters, journal articles, reports) by analyzing their bibo:content text. Use when indexing references with dcterms:subject and dcterms:spatial keywords, enriching reference metadata, or assigning thematic and geographic terms to scholarly items in the IWAC collection. Not for newspaper articles or other item types.
---

# Reference Indexing

Assign `Subject AI` (5–8 thematic keywords) and `Spatial AI` (geographic locations) to scholarly references by analyzing their full text.

## Prerequisites

Run `AI_reference_indexing/01_fetch_references.py` first to produce:
- `AI_reference_indexing/output/items_*.csv` — items with `bibo:content`
- `AI_reference_indexing/output/index_subject.csv` — subject/topic authority terms
- `AI_reference_indexing/output/index_spatial.csv` — spatial authority terms

## Workflow

1. Read the keyword assignment rules from `AI_reference_indexing/02_enrichment_prompt.md`.

2. Read authority indices to learn the existing vocabulary. Exclude individuals (item set 266) to keep context manageable — focus on subject/topic terms. Persons can still be assigned as keywords; reconciliation handles matching.

3. Find the latest `items_*.csv` in `AI_reference_indexing/output/` (exclude `_enriched` files).

4. For each item:
   - **Skip** if `bibo:content` is empty.
   - **Skip** if both `Existing Subject IDs` and `Existing Spatial IDs` are already populated.
   - Assign keywords following the prompt rules. Use existing index terms when they fit, but freely create new ones — they will be reconciled later.
   - **All keywords in French**, regardless of document language.

5. Write `AI_reference_indexing/output/items_enriched_{YYYYMMDD_HHMMSS}.csv` — all original columns plus `Subject AI` and `Spatial AI` (pipe-separated).

6. Write `AI_reference_indexing/output/keyword_summary_{YYYYMMDD_HHMMSS}.csv` with columns: `Term`, `Type` (subject/spatial), `Count`, `In Index` (yes/no), `Index ID` (if matched).
