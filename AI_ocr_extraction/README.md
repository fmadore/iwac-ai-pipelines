# AI OCR & HTR Extraction Pipeline

This pipeline provides a complete workflow for downloading PDFs from an Omeka S digital collection and performing either:

1. OCR (Optical Character Recognition) for printed / typeset French documents (e.g. newspapers)
2. HTR (Handwritten Text Recognition) for French handwritten manuscripts, marginalia, annotations

Both modes use Google's Gemini AI vision models and produce research‑grade, archival-quality text which can be synced back into Omeka S.

## Overview

The pipeline consists of core scripts plus two alternative processing paths (choose OCR or HTR depending on source material):

1. **`01_omeka_pdf_downloader.py`** – Downloads PDF files from Omeka S collections
2a. **`02_gemini_ocr_processor.py`** – (Printed) AI-powered OCR using Gemini
2b. **`02_gemini_htr_processor.py`** – (Handwritten) High-precision HTR using Gemini
3. **`03_omeka_content_updater.py`** – Updates Omeka S items with extracted text content

You can keep both processing scripts; simply run the one appropriate for the collection type.

## Features

- **Multi-threaded PDF downloading** with progress tracking
- **High-precision OCR / HTR** for French printed & handwritten sources
- **Robust error handling** with comprehensive logging
- **Academic-grade text extraction** with proper formatting & French typography
- **Automatic database updates** preserving existing metadata
- **Configurable AI models** (Gemini 2.5 Flash or Pro)
- **Dedicated system prompts** for printed (`ocr_system_prompt.md`) and handwritten (`htr_system_prompt.md`) material

## Prerequisites

### Software Requirements

- Python 3.8+
- Poppler PDF utilities (for PDF to image conversion)
- Google Gemini API access

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - HTTP requests for API calls
- `google-genai` - Google Gemini API client
- `pdf2image` - PDF to image conversion
- `Pillow` - Image processing
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars
- `pathlib` - Modern path handling

### Poppler Installation (Windows)

1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler\`
3. The pipeline expects Poppler at: `C:\Program Files\poppler\Library\bin`

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Omeka S API Configuration
OMEKA_BASE_URL=https://your-omeka-instance.com
OMEKA_KEY_IDENTITY=your_api_key_identity
OMEKA_KEY_CREDENTIAL=your_api_key_credential

# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key
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

### Step 2 (Option A): Perform OCR (Printed / Typeset)

```bash
python 02_gemini_ocr_processor.py
```

- Select Gemini model (2.5 Flash or 2.5 Pro)
- Uses `ocr_system_prompt.md`
- Processes all PDFs in the `PDF/` directory
- Saves extracted text to `OCR_Results/` directory
- Logs to `ocr_gemini.log`

### Step 2 (Option B): Perform HTR (Handwritten)

```bash
python 02_gemini_htr_processor.py
```

- Select Gemini model (2.5 Flash or 2.5 Pro)
- Uses `htr_system_prompt.md` (falls back to OCR prompt if missing)
- Applies stricter line joining, de-hyphenation & zone handling for manuscripts
- Saves extracted text to `OCR_Results/`
- Logs to `ocr_gemini.log` (consider archiving if you need separate logs per mode)

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
├── 02_gemini_ocr_processor.py   # OCR processing script (printed)
├── 02_gemini_htr_processor.py   # HTR processing script (handwritten)
├── 03_omeka_content_updater.py  # Database update script
├── ocr_system_prompt.md         # OCR system prompt (printed French articles)
├── htr_system_prompt.md         # HTR system prompt (handwritten French documents)
├── README.md                    # This documentation
├── PDF/                         # Downloaded PDF files
├── OCR_Results/                 # Extracted text files
├── pdf_download.log             # Download activity log
└── ocr_gemini.log               # Processing log
```

## System Prompts (Printed vs Handwritten)

Two dedicated prompt files control extraction behavior:

1. `ocr_system_prompt.md` – Printed French newspapers / typeset PDFs (multi-column layout, reading order preservation).
2. `htr_system_prompt.md` – Handwritten French manuscripts & annotations (zone ordering, semantic line joining, de-hyphenation, French typography, self-review checklist).

Edit these Markdown files to refine behavior without modifying Python code.

### Key OCR Features (Printed)
- Multi-column layout handling & reading order preservation
- French typography spacing & diacritics
- Hyphenation correction (end-of-line) & paragraph reconstruction
- Stable output suited for research indexing

### Key HTR Features (Handwritten)
- Strict zone segmentation (headers, marginalia, notes, captions)
- Semantic line joining & controlled paragraph spacing (`\n\n` only between paragraphs)
- Aggressive de-hyphenation while preserving legitimate compounds
- Preservation of diacritics & archival fidelity
- Built-in self-review checklist to ensure formatting consistency

## Error Handling

The pipeline includes comprehensive error handling:

- **Network timeouts** with automatic retries
- **API rate limiting** with exponential backoff
- **File corruption** detection and recovery
- **Missing dependencies** with clear error messages
- **Partial processing** continuation after errors

## Logging

Each script creates detailed logs:

- `pdf_download.log` - Download progress and errors
- `ocr_gemini.log` - OCR processing details and API responses

Log files use timestamps and severity levels for easy debugging.

## Performance Considerations

### PDF Download
- Uses 2 concurrent threads by default (configurable)
- Streams large files to minimize memory usage
- Implements connection pooling for efficiency

### OCR / HTR Processing
- Processes one PDF at a time to respect API limits
- Uses temporary files for image processing (cleaned up automatically)
- Retry logic for transient failures (timeouts / 503 / network)
- Unified logging for traceability

### Memory Management
- Configured for large image processing
- Automatic cleanup of temporary files
- Optimized PIL settings for high-DPI images

## Troubleshooting

### Common Issues

1. **"Poppler not found"**
   - Verify Poppler installation path
   - Check PATH environment variable

2. **"API key not found"**
   - Verify `.env` file exists and contains valid keys
   - Check environment variable names

3. **"PDF conversion failed"**
   - PDF may be corrupted or protected
   - Try manual conversion with different settings

4. **"OCR/HTR timeout"**
   - Large or complex PDFs may exceed timeout
   - Consider splitting large documents or reducing DPI if acceptable

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

1. Test with small datasets first
2. Update / version the relevant system prompt (`ocr_system_prompt.md` or `htr_system_prompt.md`) when changing extraction logic
3. Maintain backward compatibility (avoid breaking existing scripts)
4. Add appropriate error handling & logging
5. Update documentation (README + prompt comments)

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure API credentials are valid and have proper permissions
4. Test with a small dataset before processing large collections

## License

This pipeline is designed for academic research and archival purposes under fair use principles. Choose the processing mode (OCR vs HTR) that matches your source material for optimal accuracy.
