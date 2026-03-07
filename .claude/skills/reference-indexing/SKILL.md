---
name: reference-indexing
description: Assign controlled subject and spatial keywords to IWAC scholarly references (book chapters, journal articles, reports) by analyzing their bibo:content text. Use when indexing references with dcterms:subject and dcterms:spatial keywords, enriching reference metadata, or assigning thematic and geographic terms to scholarly items in the IWAC collection. Not for newspaper articles or other item types.
---

# Reference Indexing

Assign `Subject AI` (5–8 thematic keywords) and `Spatial AI` (geographic locations) to scholarly references by analyzing their full text.

## Prerequisites

Run `AI_reference_indexing/01_fetch_references.py` first to produce:
- `AI_reference_indexing/output/items_*.csv` — items with `bibo:content`

Authority index CSVs are also exported but are **only used by Python reconciliation** (Step 3), not by the AI enrichment step.

## Workflow

1. Read the keyword assignment rules from `AI_reference_indexing/02_enrichment_prompt.md`.

2. Find the latest `items_*.csv` in `AI_reference_indexing/output/` (exclude `_enriched` files). Also read `index_subject.csv` and `index_spatial.csv` **only to build an ID→title lookup** (do NOT load the full indices into sub-agent context).

3. Process items using **one sub-agent per item** (Agent tool, `subagent_type=general-purpose`). Each sub-agent receives:
   - The enrichment prompt
   - The item's title and `bibo:content`
   - The item's **existing keywords resolved to human-readable names** (look up `Existing Subject IDs` and `Existing Spatial IDs` in the ID→title maps built from step 2)

   This keeps each agent's context clean and enables parallel processing.
   - **Skip** if `bibo:content` is empty.
   - **Skip** if both `Existing Subject IDs` and `Existing Spatial IDs` are already populated.
   - Assign keywords following the prompt rules, complementing (not duplicating) existing keywords.
   - **All keywords in French**, regardless of document language.

4. Write `AI_reference_indexing/output/items_enriched_{YYYYMMDD_HHMMSS}.csv` — all original columns plus `Subject AI` and `Spatial AI` (pipe-separated).

5. Write `AI_reference_indexing/output/keyword_summary_{YYYYMMDD_HHMMSS}.csv` with columns: `Term`, `Type` (subject/spatial), `Count`.
