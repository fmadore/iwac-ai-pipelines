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
python 02_AI_generate_summaries_issue.py
```

The script auto-detects PDFs and processes them sequentially.

### With Omeka S Integration

```bash
python 01_omeka_pdf_downloader.py   # Download PDFs from collection
python 02_AI_generate_summaries_issue.py   # Extract articles
```

## Supported Models

| Provider | Model | Best For |
|----------|-------|----------|
| **Gemini** (recommended) | Pro + Flash | Best extraction quality, accurate article detection |
| Mistral | OCR + Small | Alternative if Gemini unavailable |

The Gemini version produces significantly better results. Use Mistral only for experimentation.

## Output

```
Magazine_Extractions/
└── 12345/                          # PDF filename/Omeka ID
    ├── 12345_index.json            # Machine-readable index
    └── 12345_final_index.md        # Human-readable index
```

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
