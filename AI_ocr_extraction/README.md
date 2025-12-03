# AI OCR Extraction Pipeline

This pipeline provides a complete workflow for downloading PDFs from an Omeka S digital collection and performing high-precision OCR (Optical Character Recognition) using AI-powered document processing.

The pipeline supports two AI providers:
- **Google Gemini** – Uses multimodal capabilities with custom prompts for guided extraction
- **Mistral Document AI** – Uses dedicated OCR endpoint optimized for raw text extraction

Both produce research-grade, archival-quality text that can be synced back into Omeka S.

## Overview

The pipeline consists of four main scripts:

1. **`01_omeka_pdf_downloader.py`** – Downloads PDF files from Omeka S collections
2. **`02_gemini_ocr_processor.py`** – AI-powered page-by-page PDF OCR using Gemini (with system prompts)
3. **`02_mistral_ocr_processor.py`** – OCR using Mistral's dedicated Document AI endpoint
4. **`03_omeka_content_updater.py`** – Updates Omeka S items with extracted text content

## Features

- **Native PDF Processing** – Direct page-by-page PDF processing without image conversion
- **Multi-provider support** – Choose between Gemini or Mistral for OCR
- **Multi-threaded PDF downloading** with progress tracking
- **High-precision OCR** for French printed documents
- **Robust error handling** with comprehensive logging
- **Academic-grade text extraction** with proper formatting & French typography
- **Automatic database updates** preserving existing metadata
- **Configurable AI models** (Gemini Flash/Pro or Mistral OCR)
- **Intelligent copyright handling** with automatic RECITATION error recovery (Gemini)
- **Fair use compliance** with academic research-focused prompts

## Prerequisites

### Software Requirements

- Python 3.8+
- Google Gemini API access (for Gemini OCR)
- Mistral API access (for Mistral OCR)

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - HTTP requests for API calls
- `google-genai` - Google Gemini API client (for Gemini OCR)
- `mistralai` - Mistral AI client (for Mistral OCR)
- `PyPDF2` - PDF page extraction
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars
- `pathlib` - Modern path handling

## Provider Comparison

### Gemini (02_gemini_ocr_processor.py)
- **Approach**: Multimodal model with custom system prompts
- **Customization**: Full control via `ocr_system_prompt.md`
- **Best for**: Documents needing guided extraction rules (French typography, layout handling)
- **Processing**: Page-by-page for precise control
- **Copyright handling**: Has RECITATION detection, requires fallback strategies

### Mistral Document AI (02_mistral_ocr_processor.py)
- **Approach**: Dedicated OCR endpoint (`mistral-ocr-latest`)
- **Customization**: No system prompts – pure OCR extraction
- **Best for**: Fast, accurate raw text extraction without custom rules
- **Processing**: Whole document or page-by-page (user choice)
- **Output format**: Markdown with preserved structure

## Model Selection Considerations

### Gemini Models

#### Gemini 2.5 Flash (Recommended for Speed)
- **Faster processing** and lower cost
- **Good accuracy** for most documents
- **Thinking**: Can be disabled with `thinking_budget=0` for faster OCR
- **More conservative** copyright detection (higher RECITATION rate)
- **Best for**: Modern documents, clear PDFs, non-copyrighted content
- **Cost**: ~$0.54 per 450-page document
- **Fallback handling**: Automatic alternative prompts for RECITATION errors

#### Gemini 3.0 Pro (Recommended for Quality)
- **Higher accuracy** and more sophisticated reasoning
- **Slower processing** and higher cost
- **Thinking**: Uses `thinking_level` parameter ("low" or "high"), cannot be disabled
- **Less restrictive** copyright detection (lower RECITATION rate)
- **Best for**: Complex layouts, poor quality scans, historical copyrighted material
- **Cost**: ~$4.49 per 450-page document
- **Default OCR config**: `thinking_level="low"` for faster processing

### Mistral Document AI

#### mistral-ocr-latest
- **Dedicated OCR model** optimized specifically for document extraction
- **Markdown output** with preserved structure (headers, tables, lists)
- **No copyright blocking** – extracts text as-is
- **Fast processing** – can process entire PDFs at once
- **Simpler API** – no system prompts needed
- **Best for**: Fast batch processing, raw text extraction, documents not needing custom rules

### Recommendation
- **Start with Gemini Flash** for fastest OCR with custom extraction rules (French typography, layout handling)
- **Use Mistral** for raw text extraction without customization needs
- **Switch to Gemini Pro** if experiencing frequent RECITATION errors or need maximum accuracy
- **Use Gemini Pro** for challenging historical documents with complex layouts

### Thinking Configuration (Gemini)

The pipeline automatically configures thinking based on model type:

| Model | Parameter | Default for OCR | Notes |
|-------|-----------|-----------------|-------|
| Gemini 2.5 Flash | `thinking_budget` | `0` (disabled) | Can be 0-24576 |
| Gemini 3.0 Pro | `thinking_level` | `"low"` | Can be "low" or "high", cannot disable |

For OCR tasks, **low thinking** is optimal because:
- OCR is straightforward text extraction, not complex reasoning
- Lower thinking = faster processing
- Low temperature + low thinking = consistent results

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Omeka S API Configuration
OMEKA_BASE_URL=https://your-omeka-instance.com
OMEKA_KEY_IDENTITY=your_api_key_identity
OMEKA_KEY_CREDENTIAL=your_api_key_credential

# Google Gemini API Configuration (for Gemini OCR)
GEMINI_API_KEY=your_gemini_api_key

# Mistral API Configuration (for Mistral OCR)
MISTRAL_API_KEY=your_mistral_api_key
```

## Usage

### Step 1: Download PDFs

```bash
python 01_omeka_pdf_downloader.py
```

- Prompts for Omeka S item set ID
- Downloads all PDFs from the specified collection
- Saves files to `PDF/` directory
- Creates `pdf_download.log` for tracking

### Step 2: Perform OCR

Choose one of the following OCR scripts:

#### Option A: Gemini OCR (with custom prompts)

```bash
python 02_gemini_ocr_processor.py
```

- Select Gemini model (2.5 Flash or 3.0 Pro)
- Processes PDFs page-by-page directly (no image conversion)
- Uses `ocr_system_prompt.md` for extraction rules
- Configures optimal thinking for OCR:
  - **Flash**: `thinking_budget=0` (disabled for speed)
  - **Pro**: `thinking_level="low"` (minimal reasoning for consistency)
- Saves extracted text to `OCR_Results/` directory
- Creates detailed page-by-page progress tracking
- Logs to `log/ocr_gemini_pdf.log`

#### Option B: Mistral OCR (dedicated Document AI)

```bash
python 02_mistral_ocr_processor.py
```

- Uses Mistral's specialized `mistral-ocr-latest` model
- Choose processing mode: whole document or page-by-page
- No system prompts – pure OCR extraction
- Outputs Markdown with preserved structure
- Saves extracted text to `OCR_Results/` directory
- Logs to `log/ocr_mistral.log`

#### Processing Details
- Each page is extracted as a single-page PDF (Gemini page-by-page mode)
- Mistral can process entire PDFs at once for faster throughput
- Automatic fallback from inline to file upload for large files
- Page markers added between pages in output
- Failed pages are logged with error messages

### Step 3: Update Database

```bash
python 03_omeka_content_updater.py
```

- Reads all `.txt` files from `OCR_Results/`
- Updates corresponding Omeka S items with extracted text
- Preserves existing metadata and properties

## Directory Structure

```
AI_ocr_extraction/
├── 01_omeka_pdf_downloader.py   # PDF download script
├── 02_gemini_ocr_processor.py   # OCR processing script (Gemini, page-by-page with prompts)
├── 02_mistral_ocr_processor.py  # OCR processing script (Mistral Document AI)
├── 03_omeka_content_updater.py  # Database update script
├── ocr_system_prompt.md         # OCR system prompt configuration (Gemini only)
├── README.md                    # This documentation
├── PDF/                         # Downloaded PDF files
├── OCR_Results/                 # Extracted text files
└── log/
    ├── pdf_download.log         # Download activity log
    ├── ocr_gemini_pdf.log       # Gemini OCR processing log
    └── ocr_mistral.log          # Mistral OCR processing log
```

## OCR System Prompt

The `ocr_system_prompt.md` file controls the extraction behavior for French newspaper articles and documents. It defines:

### Key Features
- **Reading Zone Protocol** – Multi-column layout handling & precise reading order
- **Content Hierarchy** – Primary, secondary, tertiary zones (headers, footers, captions)
- **Semantic Integration** – Intelligent paragraph joining and line merging
- **French Typography** – Proper spacing rules for ` : `, ` ; `, ` ! `, ` ? `
- **De-hyphenation** – Removes end-of-line hyphens while preserving compounds
- **Quality Control** – Built-in self-review checklist for formatting consistency

### Processing Approach
The system prompt is loaded once and combined with page-specific instructions for each page:
```
[Full OCR System Prompt from file]

Please perform complete OCR transcription of this single page.
Extract all visible text maintaining original formatting and structure.
```

The prompt is sent **after** the PDF page content (following Google's best practices for document processing).

### Customization
Edit `ocr_system_prompt.md` to refine extraction behavior without modifying Python code. The prompt emphasizes research-grade accuracy and archival-quality output suitable for academic research.

## Error Handling

The pipeline includes comprehensive error handling:

- **Network timeouts** with automatic retries
- **API rate limiting** with exponential backoff
- **Page extraction errors** with detailed logging
- **Missing dependencies** with clear error messages
- **Partial processing** continuation after errors
- **RECITATION errors** with intelligent fallback strategies

### RECITATION Error Handling (Copyright Detection)

Gemini models may trigger `FinishReason.RECITATION` when they detect potentially copyrighted content in historical documents. The pipeline automatically handles this with multi-layered fallback strategies.

#### When RECITATION Errors Occur
- **Primary trigger**: Gemini returns `FinishReason.RECITATION` 
- **Secondary trigger**: Empty response with no content/parts
- **Common cause**: Historical newspaper content triggering copyright detection

#### Automatic Fallback Strategies

When a page hits RECITATION, the system tries three escalating strategies:

1. **Academic Fair Use**: Emphasizes legitimate research, archival preservation, and fair use principles
2. **Educational Request**: Focuses on educational and research purposes
3. **Technical Analysis**: Frames as technical OCR analysis for digitization

#### Safety Settings Configuration
The pipeline configures Gemini with permissive safety settings for legitimate OCR tasks using string-based values (per latest Google GenAI SDK):
- `HARM_CATEGORY_HARASSMENT`: `BLOCK_NONE`
- `HARM_CATEGORY_HATE_SPEECH`: `BLOCK_NONE` 
- `HARM_CATEGORY_SEXUALLY_EXPLICIT`: `BLOCK_NONE`
- `HARM_CATEGORY_DANGEROUS_CONTENT`: `BLOCK_NONE`

#### Processing Flow
1. **First attempt**: Uses full detailed system prompt
2. **RECITATION detected**: Automatically tries alternative prompts
3. **Success**: Continues with extracted text
4. **All strategies fail**: Logs error and marks page as failed

#### Monitoring RECITATION Errors
Watch for these log messages:
- `"Page X: Copyright detection triggered, trying alternative..."`
- `"Page X: Trying Academic Fair Use Request..."`
- `"Page X complete (using [Strategy Name])"`
- `"All copyright retry strategies failed"`

This ensures maximum success rate even when processing historical copyrighted material under fair use principles.

## Logging

Each script creates detailed logs in the `log/` directory:

- `log/pdf_download.log` - Download progress and errors
- `log/ocr_gemini_pdf.log` - Page-by-page OCR processing details and API responses

Log files use timestamps and severity levels for easy debugging and tracking.

## Performance Considerations

### PDF Download
- Uses 2 concurrent threads by default (configurable)
- Streams large files to minimize memory usage
- Implements connection pooling for efficiency

### OCR Processing
- Processes PDFs page-by-page for better control
- Direct PDF processing (no image conversion overhead)
- Fixed token cost: 258 tokens per page
- Retry logic for transient failures
- Automatic fallback from inline to file upload for large pages

### Cost Efficiency
- **20% cheaper** than image-based OCR approaches
- **Fixed predictable cost**: 258 tokens/page for PDF + prompt tokens
- Example: 450-page document = ~$0.54 with Gemini 2.5 Flash

### Processing Speed
- ~20-40 seconds per page (no conversion overhead)
- Faster than image-based approaches (no PDF→image conversion)
- Parallel processing could be added for multiple PDFs

## Troubleshooting

### Common Issues

1. **"PyPDF2 import error"**
   - Install PyPDF2: `pip install PyPDF2`
   - Verify it's in your requirements.txt

2. **"API key not found"**
   - Verify `.env` file exists and contains valid keys
   - Check environment variable names match

3. **"Error extracting page"**
   - PDF may be corrupted or password-protected
   - Check if PDF opens correctly in a viewer
   - Try with a different PDF to isolate the issue

4. **"Page too large for inline processing"**
   - Normal for pages > 20MB
   - Script automatically uses File API upload
   - No action needed

5. **"Response lacks content. Finish reason: FinishReason.RECITATION"**
   - Gemini detected potential copyright content
   - The pipeline automatically tries alternative prompts
   - If all strategies fail, consider:
     - Using Gemini Pro instead of Flash (less restrictive)
     - Manually reviewing content for copyright concerns
     - Checking logs for which pages consistently fail

6. **"File processing timed out"**
   - Network issues or API slowdown
   - Script automatically retries with exponential backoff
   - Check internet connection and API status

### Debug Mode

Enable verbose logging by modifying the logging level in each script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## API Rate Limits

- **Gemini API**: Respect rate limits with built-in retry logic
- **Omeka S API**: Uses reasonable request spacing
- **Concurrent processing**: Limited to prevent API overload

## Contributing

When modifying the pipeline:

1. Test with small datasets first (1-2 page PDFs)
2. Update `ocr_system_prompt.md` when changing extraction logic
3. Maintain backward compatibility with existing output formats
4. Add appropriate error handling & logging
5. Update documentation for any new features or changes

## Support

For issues or questions:
1. Check the log files in `log/` directory for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure API credentials are valid and have proper permissions
4. Test with a small single-page PDF before processing large collections

## License

This pipeline is designed for academic research and archival purposes under fair use principles. It processes French newspaper articles and documents with research-grade quality suitable for scholarly work.
