---
name: issue-indexing
description: Extract article-level table of contents from digitized Islamic magazine PDFs and update Omeka S items with dcterms:tableOfContents. Claude reads PDFs directly — no LLM API calls needed. Use when indexing magazine issues, extracting article listings from PDF scans, or populating table of contents metadata for IWAC periodicals.
---

# Magazine Issue Indexing

Extract structured table of contents from digitized Islamic magazine PDFs (e.g., Le CERFIste, Plume Libre) and update Omeka S.

## What it does

1. Downloads PDFs from an Omeka S item set
2. Reads each PDF directly (no LLM API calls — Claude does the extraction)
3. Identifies articles: titles, authors, page numbers, summaries
4. Formats a `dcterms:tableOfContents` entry per magazine issue
5. Updates Omeka S items after user review

## Usage

Provide the item set ID containing magazine issues to index. The agent handles the full pipeline.

## Output format

Each item gets a table of contents like:

```
p. 1-3 : [Éditorial] L'Islam et la modernité (Imam Ismaël Tiendrébéogo)
Réflexion sur la compatibilité entre les valeurs islamiques et les défis contemporains.

p. 4-5 : Les enjeux de l'éducation coranique au Burkina Faso
Analyse des défis auxquels font face les écoles coraniques.
```
