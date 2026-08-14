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
| `1` / `a` | standard | Gemini Pro | DeepSeek V4 Flash 0731 | Best extraction quality |
| `2` / `b` | light | Gemini 3.7 Flash | DeepSeek V4 Flash 0731 | Cheaper extraction |

The profile controls only the quality-critical visual extraction. Both variants
send the resulting typed page JSON to the shared default text model,
`deepseek-v4-flash-0731`, for consolidation. The Mistral variant likewise uses
Mistral OCR for step 1 and DeepSeek for step 2.

### With Omeka S Integration

```bash
python 01_omeka_pdf_downloader.py            # Download PDFs from collection (bibo:Issue only)
python 02_AI_generate_summaries_issue.py     # Extract articles
python 03_update_omeka_toc.py --dry-run      # Preview; records DeepSeek 0731 provenance
python 03_update_omeka_toc.py                # Write the TOC to Omeka via the API
```

`03_update_omeka_toc.py` writes `dcterms:tableOfContents` to each item **directly
via the Omeka REST API** and records which model produced it as an
`iwac:summaryModel` value annotation. It is safe for existing metadata: for each
item it fetches the full record, modifies only the table-of-contents property,
and PATCHes the whole record back — aborting that item if any existing property
would be lost. Run `--dry-run` first to review; the live run asks for confirmation.
The updater defaults to the matching `DeepSeek V4 Flash 0731` authority item
(83261). Use `--model` only when uploading an index produced by another model.

### Claude Agent (Alternative)

The issue-indexing Claude agent reads PDFs directly without LLM API calls and handles the full pipeline (download, extraction, Omeka update). Use it via the `/issue-indexing` skill in Claude Code.

On that path the provenance to record is the Claude model that did the reading, not a registry model — `--model claude-opus-5` for Opus 5. Nothing in this repo can observe which model Claude Code is running, so the operator asserts it; naming last year's release is the easy mistake, which is why the key carries the version.

## Supported Models

| Provider | Model | Best For |
|----------|-------|----------|
| **Claude Agent** (recommended) | Opus 5 | Reads PDFs directly, no API costs, best quality |
| **Gemini** (standard profile) | Gemini Pro → DeepSeek V4 Flash 0731 | Good extraction quality, accurate article detection |
| **Gemini** (light profile) | Gemini 3.7 Flash → DeepSeek V4 Flash 0731 | Cheaper visual extraction |
| Mistral | OCR → DeepSeek V4 Flash 0731 | Alternative if Gemini unavailable |

Gemini model IDs come from the shared registry (`common/llm_registry.py`). The
standard profile takes the rolling `gemini-pro-latest`, which always tracks the
newest stable Pro; the light profile is pinned to `gemini-3.7-flash`, because
`gemini-flash-latest` rolled onto 3.7 on 2026-08-14 and 3.7 had dropped the
`MINIMAL` thinking level, changing the model under a run that named no version.
The Gemini version produces significantly better results than Mistral; use
Mistral only for experimentation.

The Mistral version extracts each page in a single request, passing the schema as
`document_annotation_format` on `ocr.process`. It previously called `ocr.process`
for markdown and then `chat.parse` to structure it — twice the latency and twice
the rate-limit pressure per page, with the layout information flattened away in
between.

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
OPENROUTER_API_KEY=your_key # DeepSeek consolidation in both versions

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
