# Magazine Article Extraction Pipeline

A two-step AI-powered pipeline for extracting and consolidating articles from digitized Islamic magazines.

## Available Implementations

### 1. **Gemini Version** (Recommended) - `02_AI_generate_summaries_issue.py`
Uses Google Gemini's native PDF understanding with individual page extraction.

**Models:**
- Gemini 3.0 Pro (Step 1: page-by-page extraction with thinking_level)
- Gemini 2.5 Flash (Step 2: consolidation with thinking_budget)

**Features:**
- ✅ Structured outputs - Pydantic models for guaranteed JSON schema compliance
- ✅ Native PDF processing - Each page extracted and sent individually
- ✅ Modern API patterns - Uses `system_instruction` in GenerateContentConfig
- ✅ Rich console output - Beautiful progress bars, tables, and status indicators
- ✅ Dual output format - JSON for programmatic access + Markdown for readability

### 2. **Mistral Version** - `02_Mistral_generate_summaries_issue.py`
Uses Mistral's Document AI with OCR specialization for document understanding.

**Models:**
- Mistral OCR (Step 1: page-by-page extraction with document understanding)
- Mistral Small (Step 2: cost-effective consolidation with structured outputs)

**Features:**
- ✅ Structured outputs - Pydantic models with `client.chat.parse()` for type-safe extraction
- ✅ Document OCR specialization - Extracts markdown text from PDFs
- ✅ Single PDF upload with signed URL workflow
- ✅ Dual output format - JSON for programmatic access + Markdown for readability

## Overview

This pipeline processes PDF magazines to extract and consolidate articles using AI vision models.

**Common Pipeline:**
1. **Extract articles page-by-page** using vision-capable AI models
2. **Consolidate fragmented articles** across multiple pages
3. Generate a comprehensive article index with titles, page ranges, and summaries

**Choose Your Implementation:**
- **Gemini**: Best for general use, individual page processing
- **Mistral**: OCR specialization, single upload with signed URLs

## Features

### Gemini Version
- ✅ **Structured outputs** - Pydantic models with `response_schema` for guaranteed JSON compliance
- ✅ **Native PDF processing** - Each page extracted and sent individually to Gemini
- ✅ **Modern API patterns** - Uses `system_instruction` parameter (not concatenated prompts)
- ✅ **Rich console UI** - Color-coded progress bars, tables, panels, and status indicators
- ✅ **PyPDF2 page extraction** - Individual pages sent as separate PDFs
- ✅ **Smart thinking config** - Uses `thinking_level` for Gemini 3, `thinking_budget` for 2.5

### Mistral Version
- ✅ **Structured outputs** - Pydantic models with `client.chat.parse()` for type-safe extraction
- ✅ **Document OCR specialization** - Mistral OCR model extracts markdown text
- ✅ **Single upload workflow** - Upload once, get signed URL, reference per page
- ✅ **Cost-optimized** - Uses Small model for consolidation step

### Common Features
- ✅ **Structured outputs** - Type-safe extraction with Pydantic models
- ✅ **Dual output format** - JSON files for programmatic access + Markdown for human readability
- ✅ **Robust error handling** with automatic retry (max 3 attempts)
- ✅ **Progressive saving** - JSON caching allows resumption from interruptions
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
│ Step 1: Page-by-Page Extraction (Gemini 3.0 Pro)           │
│ → Extract individual pages from PDF using PyPDF2           │
│ → Send each page separately to Gemini as PDF bytes         │
│ → Structured output: PageExtraction Pydantic model         │
│ → Identify exact titles on each page                       │
│ → Detect continuation markers ("suite page X")             │
│ → Generate brief summaries (2-3 sentences)                 │
│ Output: *_extractions.json + *_step1_consolidated.md       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Consolidation (Gemini 2.5 Flash)                   │
│ → Merge fragmented articles across pages                   │
│ → Structured output: MagazineIndex Pydantic model          │
│ → Eliminate duplicates                                     │
│ → Aggregate page ranges (e.g., 1–3, 5–7)                   │
│ → Generate comprehensive summaries (4-6 sentences)         │
│ Output: *_index.json + *_final_index.md                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `google-genai` - Google Gemini API client (for Gemini version)
- `mistralai` - Mistral AI API client (for Mistral version)
- `PyPDF2` - PDF page extraction and metadata
- `python-dotenv` - Environment variable management
- `rich` - Beautiful console output with colors and formatting
- `requests` - HTTP requests (for Omeka downloader)

2. **Configure environment variables:**

Create a `.env` file in the project root:
```bash
# Choose your provider (or configure both)

# For Gemini version
GEMINI_API_KEY=your_gemini_api_key_here

# For Mistral version  
MISTRAL_API_KEY=your_mistral_api_key_here

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

**Gemini version (recommended):**
```bash
python 02_AI_generate_summaries_issue.py
```

**Mistral version:**
```bash
python 02_Mistral_generate_summaries_issue.py
```

Both versions:
- Automatically detect PDFs in `PDF/` folder
- Use PDF filename as magazine ID
- Fully automated processing

### Option B: Use Existing PDFs

1. Place your PDF file(s) in the `PDF/` folder
2. Run the pipeline:

**Gemini version:**
```bash
python 02_AI_generate_summaries_issue.py
```

**Mistral version:**
```bash
python 02_Mistral_generate_summaries_issue.py
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
│       │   ├── page_001.json             # Individual page extractions (JSON)
│       │   ├── page_001.md               # Individual page extractions (Markdown)
│       │   ├── page_002.json
│       │   ├── page_002.md
│       │   └── ...
│       ├── 12345_extractions.json        # All extractions combined (JSON)
│       ├── 12345_step1_consolidated.md   # All pages combined (Markdown)
│       ├── 12345_index.json              # Final index (JSON)
│       └── 12345_final_index.md          # Final index (Markdown)
```

## Output Format

### Step 1 - Page Extraction

**JSON format** (for programmatic access):
```json
{
  "page_number": 1,
  "is_cover": false,
  "has_articles": true,
  "articles": [
    {
      "titre": "The Islamic Revolution and Its Impact",
      "continuation": "suite page 3",
      "resume": "This article discusses the early stages of the Islamic Revolution..."
    },
    {
      "titre": "Youth Education in Muslim Communities",
      "continuation": null,
      "resume": "The article explores educational challenges..."
    }
  ],
  "other_content": null
}
```

**Markdown format** (for human readability):
```markdown
## Page : 1

### Article 1
- Titre : The Islamic Revolution and Its Impact
- Continuation : suite page 3
- Résumé : This article discusses the early stages of the Islamic Revolution...

### Article 2
- Titre : Youth Education in Muslim Communities
- Continuation : aucune
- Résumé : The article explores educational challenges...
```

### Step 2 - Final Index

**JSON format** (for programmatic access):
```json
{
  "articles": [
    {
      "titre": "The Islamic Revolution and Its Impact",
      "pages": "1, 3-5",
      "resume": "This comprehensive article examines the Islamic Revolution from multiple perspectives..."
    },
    {
      "titre": "Youth Education in Muslim Communities",
      "pages": "2",
      "resume": "A focused analysis of educational challenges faced by Muslim youth..."
    }
  ]
}
```

**Markdown format** (for human readability):
```markdown
# Index des articles du magazine

## The Islamic Revolution and Its Impact
- Pages : 1, 3-5
- Résumé :
  This comprehensive article examines the Islamic Revolution from 
  multiple perspectives, including political, social, and religious 
  dimensions. The author traces the historical roots...

## Youth Education in Muslim Communities
- Pages : 2
- Résumé :
  A focused analysis of educational challenges faced by Muslim youth...
```

## Prompt Files

The pipeline uses two separate prompt templates. With structured outputs, prompts focus on **extraction logic** while Pydantic models define the **output schema**.

1. **`summary_prompt_issue.md`** - Step 1 extraction instructions
   - Identifies articles on each page
   - Detects continuation markers
   - Generates brief summaries
   - Schema: `PageExtraction` → `PageArticle` models

2. **`consolidation_prompt_issue.md`** - Step 2 consolidation instructions
   - Merges fragmented articles
   - Eliminates duplicates
   - Produces comprehensive summaries
   - Schema: `MagazineIndex` → `ConsolidatedArticle` models

You can customize these prompts to adjust extraction behavior. The output structure is defined by Pydantic models in the Python scripts.

## Error Handling & Recovery

### Automatic Retry
- API errors are automatically retried (max 3 attempts)
- Exponential backoff (2s, 4s, 8s delays)
- Handles connection errors and rate limits

### Progressive Caching
- Each page extraction is saved immediately as JSON
- If interrupted, restart will skip already-processed pages (checks for `.json` files)
- Delete specific page `.json` files in `step1_page_extractions/` to re-process them

### Error Placeholders
If a page fails after all retries, a placeholder is created:
```markdown
## Page : 5

Erreur lors du traitement de cette page.
```

This prevents pipeline blocking while flagging problematic pages.

## Pydantic Models (Structured Outputs)

Both implementations use the same Pydantic models for type-safe extraction:

```python
class PageArticle(BaseModel):
    """An article found on a single page."""
    titre: str = Field(description="Titre exact de l'article")
    continuation: Optional[str] = Field(default=None, description="'suite page X' ou null")
    resume: str = Field(description="Résumé bref de 2-3 phrases")

class PageExtraction(BaseModel):
    """Extraction result for a single PDF page."""
    page_number: int
    is_cover: bool = False
    has_articles: bool = True
    articles: List[PageArticle] = []
    other_content: Optional[str] = None

class ConsolidatedArticle(BaseModel):
    """A consolidated article after merging fragmented pages."""
    titre: str = Field(description="Titre exact complet")
    pages: str = Field(description="'1-3' ou '1, 3, 5'")
    resume: str = Field(description="Résumé global de 4-6 phrases")

class MagazineIndex(BaseModel):
    """Final consolidated index of all articles."""
    articles: List[ConsolidatedArticle]
```

## Configuration

The Gemini version uses `common/llm_provider.py` for model selection. Models are configured automatically:

```python
# Model selection (automatic via llm_provider)
model_step1 = get_model_option("gemini-pro")   # Gemini 3 Pro for extraction
model_step2 = get_model_option("gemini-flash") # Gemini 2.5 Flash for consolidation

# LLM Configuration
config_step1 = LLMConfig(
    thinking_level="low",  # Gemini 3 Pro uses thinking_level
    temperature=0.2
)
config_step2 = LLMConfig(
    thinking_budget=0,     # Gemini 2.5 Flash uses thinking_budget (0=disabled)
    temperature=0.3
)

# Retry Settings
MAX_RETRIES = 3          # Number of retry attempts
RETRY_DELAY = 2          # Initial delay in seconds
RETRY_BACKOFF = 2        # Delay multiplier for exponential backoff
```

### Console Output

The Gemini version features rich console output:
- 🎨 **Color-coded status** - Green for success, red for errors, cyan for info
- 📊 **Progress bars** - Real-time progress with elapsed time
- 📋 **Tables** - Clean display of model configuration and file lists
- 🔲 **Panels** - Highlighted sections for important information
- ✓/✗ **Status indicators** - Clear success/failure markers

## Architecture Decisions

### Why Structured Outputs?
Both implementations use Pydantic models with native API support:
- ✅ **Guaranteed JSON compliance** - APIs enforce schema validation
- ✅ **Type safety** - Pydantic provides runtime validation and IDE support
- ✅ **No parsing errors** - No regex or manual JSON extraction needed
- ✅ **Dual outputs** - JSON for code, Markdown for humans from the same data
- ✅ **Resumable** - JSON caching enables interruption recovery

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
- **Gemini 3.0 Pro (Step 1)**: Higher accuracy for initial extraction with `thinking_level` enabled
- **Gemini 2.5 Flash (Step 2)**: Faster consolidation with `thinking_budget=0` (disabled)

### Why `system_instruction` Pattern?
The modern Google Gen AI SDK uses `system_instruction` in `GenerateContentConfig`:
- ✅ **Cleaner separation** - System guidance separate from user content
- ✅ **Better behavior** - Model consistently follows system instructions
- ✅ **API compliance** - Follows Google's recommended patterns

### Why Rich Console Output?
- ✅ **Better visibility** - Progress bars show real-time status
- ✅ **Professional appearance** - Color-coded, formatted output
- ✅ **Error clarity** - Clear distinction between success and failures
- ✅ **Summary tables** - Easy to review processing results

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

## Choosing Between Gemini and Mistral

### Use Gemini when:
- ✅ You want the most reliable general-purpose solution
- ✅ You prefer individual page processing for better error isolation
- ✅ You need excellent multilingual support
- ✅ Cost-efficiency is important (Flash model for consolidation)

### Use Mistral when:
- ✅ You want specialized OCR model for document processing
- ✅ You need cost-effective consolidation (Small model)
- ✅ You prefer a single upload workflow
- ✅ You want to experiment with Mistral's document capabilities

### Performance Comparison

| Feature | Gemini | Mistral |
|---------|--------|---------|
| **Upload** | Per-page extraction | Single upload with signed URL |
| **Page Processing** | Individual PDFs | OCR by page number |
| **Text Format** | Native PDF | Markdown from OCR |
| **Structured Outputs** | `response_schema` | `client.chat.parse()` |
| **Models** | Pro + Flash | OCR + Small |
| **Speed** | Fast (Flash consolidation) | Fast (OCR + Small) |
| **Cost** | Lower (Flash tier) | Lower (Small tier) |

Both implementations share the same Pydantic models, prompts, and output format, making it easy to switch between them or compare results.

## License

Part of the IWAC AI Pipelines project.

## Contributing

1. Test prompt changes on sample magazines first
2. Document any configuration changes
3. Preserve error handling and retry logic
4. Maintain backwards compatibility with existing output format
