# IWAC AI Pipelines

AI-powered processing pipelines for the [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) (IWAC) - a digital database documenting Islam and Muslim communities across West Africa.

## Overview

This repository contains Python-based pipelines that leverage Large Language Models (primarily Google's Gemini and OpenAI's ChatGPT) to process, enhance, and analyze documents in the IWAC database. These tools were developed to handle the collection's 25+ million words of historical documentation, making it more accessible and searchable for researchers.

## Available Pipelines

### 🔍 Named Entity Recognition (`AI_NER/`)
Extracts and reconciles named entities (persons, organizations, locations, subjects) from Omeka S collections using a unified script supporting either Gemini AI or OpenAI ChatGPT. Specialized French prompt tuned for West African Islamic contexts.

### 📝 OCR Correction (`AI_ocr_correction/`)
Post-processes raw OCR text to fix common errors, improve formatting, and enhance readability using AI-powered correction.

### 📄 OCR Extraction (`AI_ocr_extraction/`)
Downloads PDFs from Omeka S, performs OCR using Gemini's vision capabilities, and updates the database with extracted text.

### 📋 Summary Generation (`AI_summary/`)
Generates concise French summaries of documents for improved searchability and quick content understanding.

### 🧩 NotebookLM Markdown Exporter (`NotebookLM/`)
Exports Omeka Item Sets into clean, NotebookLM-ready Markdown files (.md). Handles large collections by automatically splitting output into multiple parts. Default output is `.md` with an option to switch to `.txt` via an env var.

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **API Keys:**
   - Omeka S API credentials
   - Google Gemini API key
   - OpenAI API key (for ChatGPT features)
3. **For OCR extraction:** Poppler PDF utilities

### Installation

```bash
# Clone the repository
git clone https://github.com/fmadore/iwac-ai-pipelines.git
cd iwac-ai-pipelines

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API credentials
```

### Configuration

Edit `.env` file with your credentials:

```env
# Omeka S API
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key

# OpenAI API (for ChatGPT features)
OPENAI_API_KEY=your_openai_api_key

# NotebookLM exporter (optional)
# Default is "md". Set to "txt" to export plain text instead.
NOTEBOOKLM_EXPORT_EXT=md
```

## Basic Usage

Each pipeline follows a numbered sequence of scripts:

### Example: Named Entity Recognition

```bash
cd AI_NER/

# Step 1: Extract entities (unified script; choose provider interactively or with --model)
python 01_NER_AI.py --item-set-id 123            # interactive model choice
python 01_NER_AI.py --item-set-id 123 --model gemini --async --batch-size 20
python 01_NER_AI.py --item-set-id 123 --model openai --async --batch-size 20

# Step 2: Reconcile entities with authority records
python 02_NER_reconciliation_Omeka.py

# Step 3: Update Omeka S with reconciled data
python 03_Omeka_update.py
```

### Example: NotebookLM Markdown Exporter

PowerShell (Windows):

```powershell
# Export the whole IWAC collection (uses predefined Item Set IDs per country)
python .\NotebookLM\omeka_item_to_md.py all

# Export a single Item Set by ID
python .\NotebookLM\omeka_item_to_md.py 62076

# Optional: export as .txt instead of .md for this session
$env:NOTEBOOKLM_EXPORT_EXT = "txt"
python .\NotebookLM\omeka_item_to_md.py 62076
```

Output:
- Files are written to `NotebookLM/extracted_articles/` (subfolders per country when exporting “all”).
- Filenames: `<item-set-title>_<item-set-id>_articles.md` (or `_partN.md` when large).

## Pipeline Details

Each pipeline directory contains:
- Numbered Python scripts (run in sequence)
- Configuration files (prompts, settings)
- `README.md` with detailed instructions
- Output directories for results

## Key Features

- **Modular Design:** Each pipeline can be used independently
- **Batch Processing:** Handle large document collections efficiently
- **Error Handling:** Comprehensive logging and retry mechanisms
- **Progress Tracking:** Visual progress bars for long operations
- **French Language Support:** Optimized for French and West African contexts

## Use Cases

- **Archivists:** Automate OCR correction and metadata extraction
- **Researchers:** Extract entities and generate summaries for analysis
- **Digital Librarians:** Enhance collection searchability and accessibility
- **Academic Projects:** Process historical newspapers and Islamic publications

## Documentation

Detailed documentation for each pipeline is available in their respective directories:
- [Named Entity Recognition](AI_NER/README.md)
- [OCR Correction](AI_ocr_correction/README.md)
- [OCR Extraction](AI_ocr_extraction/README.md)
- [Summary Generation](AI_summary/README.md)
 - [NotebookLM Markdown Exporter](NotebookLM/)

## Contributing

Contributions are welcome! Please:
1. Test with small datasets first
2. Update documentation for any changes
3. Follow existing code patterns
4. Add error handling for new features

## License

This project is designed for academic research and archival purposes. See individual pipeline directories for specific usage notes.

## Support

For issues or questions:
- Check individual pipeline READMEs for troubleshooting
- Ensure all dependencies are correctly installed
- Verify API credentials and permissions
- Test with small batches before processing large collections

---

*These tools demonstrate how AI can help democratize access to under-resourced African archives while maintaining scholarly standards.*