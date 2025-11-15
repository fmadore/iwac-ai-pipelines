# Magazine Article Extraction Pipeline

A two-step AI-powered pipeline for extracting and consolidating articles from digitized Islamic magazines using Google Gemini models.

## Overview

This pipeline processes PDF magazines to:
1. **Extract articles page-by-page** using Gemini 2.5 Pro
2. **Consolidate fragmented articles** across multiple pages using Gemini 2.5 Flash
3. Generate a comprehensive article index with titles, page ranges, and summaries

## Features

- ✅ **Automatic PDF text extraction** using PyPDF2
- ✅ **Robust error handling** with automatic retry (max 3 attempts)
- ✅ **Progressive saving** - resume from cached results if interrupted
- ✅ **Smart page numbering** - controlled by script, not AI (more reliable)
- ✅ **Dual-prompt system** - separate prompts for extraction and consolidation
- ✅ **Omeka S integration** - download PDFs directly from Omeka collections

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Download PDF from Omeka S (optional)                │
│ → 01_omeka_pdf_downloader.py                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Page-by-Page Extraction (Gemini 2.5 Pro)           │
│ → Extract articles from each page                          │
│ → Identify exact titles                                    │
│ → Detect continuation markers ("suite page X")             │
│ → Generate brief summaries (2-3 sentences)                 │
│ Output: step1_page_extractions/*.md                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Consolidation (Gemini 2.5 Flash)                   │
│ → Merge fragmented articles across pages                   │
│ → Eliminate duplicates                                     │
│ → Aggregate page ranges (e.g., 1–3, 5–7)                   │
│ → Generate comprehensive summaries (4-6 sentences)         │
│ Output: {omeka_id}_final_index.md                          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `google-genai` - Google Gemini API client
- `PyPDF2` - PDF text extraction
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars
- `requests` - HTTP requests (for Omeka downloader)

2. **Configure environment variables:**

Create a `.env` file in the project root:
```bash
# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Google Application Credentials for ADC
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# For Omeka downloader (optional)
OMEKA_BASE_URL=https://your-omeka-instance.org
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential
```

## Usage

### Option A: Start with Omeka S Download

1. **Download PDF from Omeka S collection:**
```bash
python 01_omeka_pdf_downloader.py
```
- Enter the Omeka S item set ID when prompted
- PDFs are downloaded to `PDF/` folder
- Files are named using Omeka item ID (e.g., `12345.pdf`)

2. **Run the extraction pipeline:**
```bash
python 02_AI_generate_summaries_issue.py
```
- Automatically detects PDFs in `PDF/` folder
- Uses PDF filename as Omeka ID
- No manual input required (fully automated)

### Option B: Use Existing PDFs

1. Place your PDF file(s) in the `PDF/` folder
2. Run the pipeline:
```bash
python 02_AI_generate_summaries_issue.py
```

### Manual Path Selection

If you prefer to specify a custom path:
```bash
python 02_AI_generate_summaries_issue.py
# Press Enter when prompted to use default PDF/ folder
# Or enter custom path to PDF file or directory
```

## Output Structure

```
AI_summary_issue/
├── PDF/
│   └── 12345.pdf                          # Source PDF
├── Magazine_Extractions/
│   └── 12345/                             # Omeka ID
│       ├── step1_page_extractions/
│       │   ├── page_001.md               # Individual page extractions
│       │   ├── page_002.md
│       │   └── page_003.md
│       ├── 12345_step1_consolidated.md   # All pages combined
│       └── 12345_final_index.md          # Final consolidated index
```

## Output Format

### Step 1 - Page Extraction
Each page extraction contains:
```markdown
## Page : 1

### Article 1
- Titre exact : "The Islamic Revolution and Its Impact"
- Continuation : suite page 3
- Résumé :
  This article discusses the early stages of the Islamic Revolution...

### Article 2
- Titre exact : "Youth Education in Muslim Communities"
- Continuation : aucune
- Résumé :
  The article explores educational challenges...
```

### Step 2 - Final Index
Consolidated output:
```markdown
# Index des articles du magazine

## Article 1
- Titre exact : "The Islamic Revolution and Its Impact"
- Pages : 1, 3–5
- Résumé consolidé :
  This comprehensive article examines the Islamic Revolution from 
  multiple perspectives, including political, social, and religious 
  dimensions. The author traces the historical roots...

## Article 2
- Titre exact : "Youth Education in Muslim Communities"
- Pages : 2
- Résumé consolidé :
  A focused analysis of educational challenges faced by Muslim youth...
```

## Prompt Files

The pipeline uses two separate prompt templates:

1. **`summary_prompt_issue.md`** - Step 1 extraction instructions
   - Identifies articles on each page
   - Detects continuation markers
   - Generates brief summaries

2. **`consolidation_prompt_issue.md`** - Step 2 consolidation instructions
   - Merges fragmented articles
   - Eliminates duplicates
   - Produces comprehensive summaries

You can customize these prompts to adjust extraction behavior.

## Error Handling & Recovery

### Automatic Retry
- API errors are automatically retried (max 3 attempts)
- Exponential backoff (2s, 4s, 8s delays)
- Handles connection errors and rate limits

### Progressive Caching
- Each page extraction is saved immediately
- If interrupted, restart will skip already-processed pages
- Delete specific page files in `step1_page_extractions/` to re-process them

### Error Placeholders
If a page fails after all retries, a placeholder is created:
```markdown
## Page : 5

Erreur lors du traitement de cette page.
```

This prevents pipeline blocking while flagging problematic pages.

## Configuration

Edit `02_AI_generate_summaries_issue.py` to adjust:

```python
# AI Models
GEMINI_MODEL_STEP1 = "gemini-2.5-pro"      # Extraction (more accurate)
GEMINI_MODEL_STEP2 = "gemini-2.5-flash"    # Consolidation (faster)

# Retry Settings
MAX_RETRIES = 3          # Number of retry attempts
RETRY_DELAY = 2          # Initial delay in seconds
RETRY_BACKOFF = 2        # Delay multiplier for exponential backoff
```

## Architecture Decisions

### Why Script-Controlled Page Numbers?
Page numbers are injected by the script (not AI) for reliability:
- ✅ **Guaranteed accuracy** - no AI hallucination risk
- ✅ **Simplifies prompts** - AI focuses on content analysis
- ✅ **Easier consolidation** - page numbers are trustworthy

### Why Two Different Models?
- **Gemini 2.5 Pro (Step 1)**: Higher accuracy for initial extraction
- **Gemini 2.5 Flash (Step 2)**: Faster consolidation of pre-extracted data

### Why Separate Prompt Files?
- **Modularity** - Easy to modify extraction vs consolidation logic
- **Version control** - Track prompt changes independently
- **Reusability** - Prompts can be shared across projects

## Troubleshooting

### No PDFs detected
```bash
# Error: "No pages extracted from input"
```
**Solution**: Ensure PDF files are in `PDF/` folder and have `.pdf` extension

### API Key errors
```bash
# Error: "GEMINI_API_KEY not set and ADC not available"
```
**Solution**: Add `GEMINI_API_KEY` to `.env` file

### Rate limit errors
The retry mechanism handles most rate limits automatically. If persistent:
- Increase `RETRY_DELAY` in configuration
- Reduce concurrent processing (already set to 1 thread for sequential processing)

### Empty extractions
If AI returns no content:
- Check PDF text extraction quality (some PDFs may be scanned images)
- Review `summary_prompt_issue.md` for clarity
- Ensure PDF language matches prompt expectations

## License

Part of the IWAC AI Pipelines project.

## Contributing

1. Test prompt changes on sample magazines first
2. Document any configuration changes
3. Preserve error handling and retry logic
4. Maintain backwards compatibility with existing output format
