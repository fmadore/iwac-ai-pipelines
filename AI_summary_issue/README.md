# Magazine Article Extraction

Extract and index individual articles from digitized Islamic magazines using AI vision models.

## Why This Tool?

Digitized magazine PDFs are searchable as whole issues but not as individual articles. Researchers looking for specific topics must manually browse each page. This pipeline identifies article boundaries, extracts titles, and generates summaries—making magazine contents discoverable.

## How It Works

```
PDF Magazine → Page-by-page AI extraction → Article consolidation → Searchable index
```

1. **Extract**: AI identifies articles on each page, noting titles and continuation markers ("suite page X")
2. **Consolidate**: Fragments are merged across pages, duplicates eliminated
3. **Output**: JSON + Markdown index with titles, page ranges, and summaries

## Quick Start

```bash
# Place PDFs in the PDF/ folder, then run:
python 02_AI_generate_summaries_issue.py            # prompts you to choose a profile
python 02_AI_generate_summaries_issue.py --light    # skip the prompt, use the light profile
python 02_AI_generate_summaries_issue.py --profile standard
```

The script auto-detects PDFs and processes them sequentially. On start it asks
which **model profile** to run (or pass `--profile` / `--light` to skip the prompt):

| Choice | Profile | Step 1 — per page | Step 2 — consolidation | Best for |
|--------|---------|-------------------|------------------------|----------|
| `1` / `a` | standard | Gemini Pro | Gemini Flash (latest) | Best quality |
| `2` / `b` | light | Gemini Flash (latest) | Gemini Flash-Lite (latest) | Cheaper & faster |

Step 1 (per-page extraction) is the quality-critical step, so the light profile
uses Flash there and the cheaper Flash-Lite for the simpler consolidation step.

### With Omeka S Integration

```bash
python 01_omeka_pdf_downloader.py            # Download PDFs from collection (bibo:Issue only)
python 02_AI_generate_summaries_issue.py     # Extract articles
python 03_update_omeka_toc.py --dry-run      # Preview the Omeka changes (writes nothing)
python 03_update_omeka_toc.py                # Write the TOC to Omeka via the API
```

`03_update_omeka_toc.py` writes `dcterms:tableOfContents` to each item **directly
via the Omeka REST API** and records which model produced it as an
`iwac:summaryModel` value annotation. It is safe for existing metadata: for each
item it fetches the full record, modifies only the table-of-contents property,
and PATCHes the whole record back — aborting that item if any existing property
would be lost. Run `--dry-run` first to review; the live run asks for confirmation.
Annotation-model choices:

| # | Annotation model | Omeka item |
|---|------------------|------------|
| 1 | Claude Opus 4.6 | 78528 |
| 2 | Gemini 3.1 pro | 78536 |
| 3 | Gemini 3.5 flash | 78630 |
| 4 | Gemini 3.1 flash lite | 78631 |

### Claude Agent (Alternative)

The issue-indexing Claude agent reads PDFs directly without LLM API calls and handles the full pipeline (download, extraction, Omeka update). Use it via the `/issue-indexing` skill in Claude Code.

## Supported Models

| Provider | Model | Best For |
|----------|-------|----------|
| **Claude Agent** (recommended) | Opus 4.6 | Reads PDFs directly, no API costs, best quality |
| **Gemini** (standard profile) | Gemini Pro + Flash | Good extraction quality, accurate article detection |
| **Gemini** (light profile) | Gemini Flash + Flash-Lite | Cheaper & faster; minor quality trade-off |
| Mistral | OCR + Small | Alternative if Gemini unavailable |

Gemini model IDs come from the shared registry (`common/llm_provider.py`) via the
rolling `gemini-flash-latest` / `gemini-flash-lite-latest` aliases, so they always
track the newest stable Flash models. The Gemini version produces significantly
better results than Mistral; use Mistral only for experimentation.

## Output

```
Magazine_Extractions/
└── 12345/                          # PDF filename/Omeka ID
    ├── 12345_final_index.json      # Machine-readable index
    └── 12345_final_index.md        # Human-readable index
```

Step 3 (`03_update_omeka_toc.py`) consumes the `*_final_index.json` files and
writes the result straight to Omeka via the API — no intermediate CSV.

**Sample output:**

```markdown
# Index des articles du magazine

## The Islamic Revolution and Its Impact
- Pages: 1, 3-5
- Résumé: This article examines the Islamic Revolution from political,
  social, and religious perspectives...

## Youth Education in Muslim Communities
- Pages: 2
- Résumé: Analysis of educational challenges faced by Muslim youth...
```

## Limitations

**Article boundary detection**: AI may miss articles that span unusual layouts (sidebars, pull quotes) or misidentify advertisements as articles.

**Title accuracy**: Decorative or ambiguous headers may be incorrectly identified as article titles.

**Language assumption**: Prompts are optimized for French/Arabic magazines. Other languages may need prompt adjustments.

**Page-by-page processing**: Articles split across pages are consolidated algorithmically. Complex multi-part series may not merge correctly.

## Configuration

Create `.env` in project root:

```bash
GEMINI_API_KEY=your_key    # For Gemini version
MISTRAL_API_KEY=your_key   # For Mistral version

# Optional: Omeka S integration
OMEKA_BASE_URL=https://your-instance.org
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
```

## Error Recovery

- **Automatic retry**: Failed pages retry 3 times with exponential backoff
- **Progressive saving**: Each page saved immediately; interrupted runs resume from last checkpoint
- **Failed pages**: Marked with placeholder text, don't block the pipeline

## Customization

Edit the prompt files to adjust extraction behavior:
- `summary_prompt_issue.md` — Page extraction rules
- `consolidation_prompt_issue.md` — Article merging logic

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No PDFs detected | Ensure files are in `PDF/` folder with `.pdf` extension |
| API key errors | Check `.env` file exists with valid keys |
| Empty extractions | PDF may be image-only; ensure it's not password-protected |
| Rate limits | Script handles automatically; increase `RETRY_DELAY` if persistent |
