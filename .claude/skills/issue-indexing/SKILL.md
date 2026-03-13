---
name: issue-indexing
description: Extract article-level table of contents from digitized Islamic magazine PDFs and generate a CSV for Omeka S CSV Import. Claude reads PDFs directly — no LLM API calls needed. Use when indexing magazine issues, extracting article listings from PDF scans, or populating table of contents metadata for IWAC periodicals.
---

# Magazine Issue Indexing

Extract structured table of contents from digitized Islamic magazine PDFs (e.g., Le CERFIste, Plume Libre, Alif) and generate a CSV for Omeka S CSV Import.

## CRITICAL: Follow the existing Python pipeline

You MUST use the existing scripts in `AI_summary_issue/`. Do NOT create temp folders, custom download scripts, or any ad-hoc workflow.

### Pipeline steps

1. **Download PDFs** — Run `AI_summary_issue/01_omeka_pdf_downloader.py` to download PDFs to `AI_summary_issue/PDF/` (only `bibo:Issue` items)
2. **Extract ToC** — Spawn `issue-indexing` sub-agents to read PDFs and extract articles (Claude reads PDFs directly — no LLM API calls)
3. **Save results** — Write extraction results to `AI_summary_issue/Magazine_Extractions/{item_id}/{item_id}_final_index.json`
4. **Review** — Present results to the user for approval
5. **Generate CSV** — Run `AI_summary_issue/03_update_omeka_toc.py` to generate `toc_import.csv` (the user will upload this via Omeka S CSV Import themselves)

### Sub-agent instructions

When spawning `issue-indexing` sub-agents for extraction, you MUST:

- Include the FULL content of `extraction_prompt.md` (in this skill directory) in the sub-agent prompt
- Explicitly instruct sub-agents to use PyPDF2 via Bash (`.venv/Scripts/python.exe`) to extract text page by page, since the Read tool may lack poppler for PDF rendering
- Tell sub-agents to write output to a specific batch file in `AI_summary_issue/`
- **ACCENTS**: Emphasize in the prompt that ALL output MUST have proper French accents. The OCR text is often unaccented — the agent must ADD correct accents. Include the accent rules and common proper name list from `extraction_prompt.md`. This is the #1 source of errors.

## Output format

Each item gets a table of contents like:

```
p. 1-3 : [Éditorial] L'Islam et la modernité (Imam Ismaël Tiendrébéogo)
Réflexion sur la compatibilité entre les valeurs islamiques et les défis contemporains.

p. 4-5 : Les enjeux de l'éducation coranique au Burkina Faso
Analyse des défis auxquels font face les écoles coraniques.
```
