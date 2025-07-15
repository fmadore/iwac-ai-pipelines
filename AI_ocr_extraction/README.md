# AI OCR Extraction Pipeline

This pipeline provides a complete workflow for downloading PDFs from an Omeka S digital collection, performing OCR (Optical Character Recognition) using Google's Gemini AI, and updating the database with the extracted text content.

## Overview

The pipeline consists of three main scripts that work together:

1. **`01_omeka_pdf_downloader.py`** - Downloads PDF files from Omeka S collections
2. **`02_gemini_ocr_processor.py`** - Performs AI-powered OCR on PDFs using Google Gemini
3. **`03_omeka_content_updater.py`** - Updates Omeka S items with extracted text content

## Features

- **Multi-threaded PDF downloading** with progress tracking
- **High-precision OCR** specialized for French newspaper articles
- **Robust error handling** with comprehensive logging
- **Academic-grade text extraction** with proper formatting
- **Automatic database updates** preserving existing metadata
- **Configurable AI models** (Gemini 2.5 Flash or Pro)

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

### Step 2: Perform OCR

```bash
python 02_gemini_ocr_processor.py
```

- Choose between Gemini 2.5 Flash (faster) or Pro (more accurate)
- Processes all PDFs in the `PDF/` directory
- Saves extracted text to `OCR_Results/` directory
- Creates `ocr_gemini.log` for detailed logging

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
├── 01_omeka_pdf_downloader.py  # PDF download script
├── 02_gemini_ocr_processor.py  # OCR processing script
├── 03_omeka_content_updater.py # Database update script
├── ocr_system_prompt.md        # OCR system prompt configuration
├── README.md                   # This documentation
├── PDF/                        # Downloaded PDF files
├── OCR_Results/                # Extracted text files
├── pdf_download.log            # Download activity log
└── ocr_gemini.log             # OCR processing log
```

## OCR System Prompt

The OCR system uses a specialized prompt optimized for French newspaper articles, stored in `ocr_system_prompt.md`. This prompt can be modified to adjust the OCR behavior without changing the code.

### Key OCR Features

- **French typography support** with proper spacing rules
- **Multi-column layout handling** with reading order preservation
- **Hyphenation correction** for end-of-line breaks
- **Paragraph semantic analysis** for proper text flow
- **Academic-grade accuracy** for research purposes

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

### OCR Processing
- Processes one PDF at a time to respect API limits
- Uses temporary files for image processing
- Includes retry logic for transient failures

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

4. **"OCR timeout"**
   - Large or complex PDFs may exceed timeout
   - Consider splitting large documents

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
2. Update the system prompt for OCR changes
3. Maintain backward compatibility
4. Add appropriate error handling
5. Update documentation

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure API credentials are valid and have proper permissions
4. Test with a small dataset before processing large collections

## License

This pipeline is designed for academic research and archival purposes under fair use principles.
