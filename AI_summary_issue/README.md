# Magazine Article Extraction Pipeline

A two-step AI-powered pipeline for extracting and consolidating articles from digitized Islamic magazines using Google Gemini models.

## Overview

This pipeline processes PDF magazines to:
1. **Extract articles page-by-page** using Gemini 2.5 Pro with native PDF understanding
2. **Consolidate fragmented articles** across multiple pages using Gemini 2.5 Flash
3. Generate a comprehensive article index with titles, page ranges, and summaries

## Features

- ✅ **Native PDF processing** - Each page extracted and sent individually to Gemini
- ✅ **No image conversion needed** - Leverages Gemini's native PDF understanding
- ✅ **Robust error handling** with automatic retry (max 3 attempts)
- ✅ **Progressive saving** - resume from cached results if interrupted
- ✅ **Smart page numbering** - controlled by script, not AI (more reliable)
- ✅ **Dual-prompt system** - separate prompts for extraction and consolidation
- ✅ **Omeka S integration** - download PDFs directly from Omeka collections
- ✅ **Gemini-only** - Optimized for Google's Gemini models (Pro + Flash)

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Download PDF from Omeka S (optional)                │
│ → 01_omeka_pdf_downloader.py                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Page-by-Page Extraction (Gemini 2.5 Pro)           │
│ → Extract individual pages from PDF using PyPDF2           │
│ → Send each page separately to Gemini as PDF bytes         │
│ → Identify exact titles on each page                       │
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
- `PyPDF2` - PDF page extraction (not text extraction)
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars
- `requests` - HTTP requests (for Omeka downloader)

2. **Configure environment variables:**

Create a `.env` file in the project root:
```bash
# Gemini API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

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
- Extracts each page individually using PyPDF2
- Sends individual page PDFs to Gemini for analysis
- Uses PDF filename as magazine ID
- Fully automated (Gemini Pro → Flash pipeline)

### Option B: Use Existing PDFs

1. Place your PDF file(s) in the `PDF/` folder
2. Run the pipeline:
```bash
python 02_AI_generate_summaries_issue.py
```

### Automatic Processing

The pipeline automatically:
1. Detects all PDFs in the `PDF/` folder
2. Extracts individual pages from each PDF
3. Processes each page with Gemini Pro
4. Consolidates results with Gemini Flash
5. Saves outputs to `Magazine_Extractions/{pdf_id}/`

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

### Why Extract Individual Pages?
Each page is extracted separately and sent to Gemini as a standalone PDF:
- ✅ **Better accuracy** - Gemini focuses on one page at a time
- ✅ **More efficient** - Smaller payloads per API request
- ✅ **Better error handling** - Failed pages don't block the entire document
- ✅ **Native PDF understanding** - Gemini processes PDFs directly, no image conversion

### Why Script-Controlled Page Numbers?
Page numbers are injected by the script (not AI) for reliability:
- ✅ **Guaranteed accuracy** - no AI hallucination risk
- ✅ **Simplifies prompts** - AI focuses on content analysis
- ✅ **Easier consolidation** - page numbers are trustworthy

### Why Two Different Models?
- **Gemini 2.5 Pro (Step 1)**: Higher accuracy for initial extraction with thinking enabled
- **Gemini 2.5 Flash (Step 2)**: Faster consolidation with thinking disabled

### Why Gemini Only?
- **Native PDF processing** - Gemini can read PDFs directly without OCR
- **Vision capabilities** - Handles images, diagrams, and complex layouts
- **Better document understanding** - Maintains context of visual structure
- **No OpenAI support** - ChatGPT lacks native PDF understanding

### Why Separate Prompt Files?
- **Modularity** - Easy to modify extraction vs consolidation logic
- **Version control** - Track prompt changes independently
- **Reusability** - Prompts can be shared across projects

## Troubleshooting

### No PDFs detected
```bash
# Error: "No PDF files found in the PDF directory!"
```
**Solution**: Ensure PDF files are in `PDF/` folder and have `.pdf` extension

### API Key errors
```bash
# Error: "GEMINI_API_KEY not found in environment variables"
```
**Solution**: Add `GEMINI_API_KEY` to `.env` file

### Page extraction errors
```bash
# Error: "Error extracting page X from PDF"
```
**Solution**: 
- Ensure PDF is not corrupted or password-protected
- Check PDF is readable by PyPDF2
- Try re-downloading the PDF if from Omeka

### Rate limit errors
The retry mechanism handles most rate limits automatically. If persistent:
- Increase `RETRY_DELAY` in configuration
- Reduce concurrent processing (already set to 1 thread for sequential processing)

### Empty extractions
If AI returns no content:
- Gemini should handle scanned images (has vision capabilities)
- Review `summary_prompt_issue.md` for clarity
- Ensure PDF language matches prompt expectations (French/Arabic)
- Check if PDF pages are actually extractable by PyPDF2
- Verify individual page PDFs are being created correctly

## License

Part of the IWAC AI Pipelines project.

## Contributing

1. Test prompt changes on sample magazines first
2. Document any configuration changes
3. Preserve error handling and retry logic
4. Maintain backwards compatibility with existing output format
