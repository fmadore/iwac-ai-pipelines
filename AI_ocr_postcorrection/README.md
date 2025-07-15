# OCR Post-Correction Pipeline

This directory contains a three-step pipeline for extracting, correcting, and updating OCR text using AI-powered post-processing with Google's Gemini models.

## Overview

The OCR post-correction pipeline processes raw OCR text extracted from Omeka S items through AI-powered correction and uploads the improved text back to the database. This workflow significantly improves text quality by fixing common OCR errors, formatting issues, and structural problems.

## Pipeline Components

### 1. OCR Text Extraction (`01_extract_ocr_text.py`)
**Purpose**: Retrieves raw OCR text from Omeka S items and saves them as individual text files.

**Features**:
- Fetches items from specified Omeka S item sets
- Extracts `extracttext:extracted_text` property values
- Concurrent processing with thread pool
- Pagination support for large item sets
- Comprehensive error handling and logging

**Input**: Omeka S item set ID
**Output**: Individual `.txt` files in `TXT/` directory

### 2. AI-Powered Text Correction (`02_correct_ocr_text.py`)
**Purpose**: Applies AI-powered corrections to raw OCR text using Google Gemini models.

**Features**:
- Specialized OCR correction prompts
- Intelligent text chunking for large documents
- Structure analysis and paragraph reconstruction
- Character recognition error correction
- Spelling and grammar improvement
- Historical text preservation

**Input**: Raw OCR text files from `TXT/` directory
**Output**: Corrected text files in `Corrected_TXT/` directory

### 3. Content Database Update (`03_update_database.py`)
**Purpose**: Updates Omeka S items with corrected OCR text content.

**Features**:
- Preserves existing item metadata
- Updates or creates `bibo:content` property
- Exponential backoff retry mechanism
- Progress tracking and detailed logging
- Comprehensive error handling

**Input**: Corrected text files from `Corrected_TXT/` directory
**Output**: Updated Omeka S database items

## Directory Structure

```
AI_ocr_postcorrection/
├── 01_extract_ocr_text.py          # Step 1: Extract raw OCR text
├── 02_correct_ocr_text.py          # Step 2: AI-powered correction
├── 03_update_database.py           # Step 3: Update database
├── ocr_correction_prompt.md        # System prompt for OCR correction
├── TXT/                            # Raw OCR text files
├── Corrected_TXT/                  # AI-corrected text files
└── README.md                       # This file
```

## Prerequisites

- Python 3.8+
- Required packages (see parent directory's `requirements.txt`)
- Environment variables configured (see Setup section)

## Setup

1. **Environment Configuration**:
   Create a `.env` file in the parent directory with:
   ```env
   OMEKA_BASE_URL=https://your-omeka-instance.com/api
   OMEKA_KEY_IDENTITY=your_key_identity
   OMEKA_KEY_CREDENTIAL=your_key_credential
   GEMINI_API_KEY=your_gemini_api_key
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

## Usage

### Complete Pipeline Execution

Run the scripts in sequence:

```bash
# Step 1: Extract OCR text from Omeka S
python 01_extract_ocr_text.py

# Step 2: Apply AI corrections
python 02_correct_ocr_text.py

# Step 3: Update Omeka S with corrected text
python 03_update_database.py
```

### Individual Script Usage

#### 1. OCR Text Extraction
```bash
python 01_extract_ocr_text.py
```
- Prompts for item set ID
- Creates `TXT/` directory
- Saves raw OCR text files

#### 2. AI Text Correction
```bash
python 02_correct_ocr_text.py
```
- Processes all `.txt` files in `TXT/` directory
- Creates `Corrected_TXT/` directory
- Applies AI-powered corrections

#### 3. Database Update
```bash
python 03_update_database.py
```
- Updates Omeka S items with corrected content
- Preserves existing metadata
- Provides detailed progress reports

## Configuration

### OCR Correction Prompt
The `ocr_correction_prompt.md` file contains the specialized system instruction for Gemini. It focuses on:
- Structure analysis and paragraph reconstruction
- Character recognition error correction
- Spelling and grammar improvement
- Format cleaning and normalization
- Historical text preservation

### Processing Parameters
- **Concurrent threads**: 5 (configurable in script)
- **Items per page**: 100 (API pagination)
- **Max text length**: 200,000 characters before chunking
- **Gemini model**: `gemini-2.5-flash`
- **Temperature**: 0.2 (for consistent corrections)

## Error Handling

All scripts include comprehensive error handling:
- **Network errors**: Automatic retry with exponential backoff
- **API rate limits**: Built-in delays and retry mechanisms
- **File I/O errors**: Graceful handling with detailed logging
- **Processing failures**: Continue processing remaining items

## Logging

- **Console output**: Real-time progress with tqdm
- **File logging**: Detailed logs for debugging
- **Error reporting**: Failed items tracked and reported

## Performance Considerations

- **Concurrent processing**: Thread pools for parallel execution
- **Memory management**: Text chunking for large documents
- **API efficiency**: Batch operations and connection reuse
- **Error recovery**: Resilient processing with retry mechanisms

## Common Issues and Solutions

### 1. API Authentication Errors
- Verify `.env` file configuration
- Check API key permissions in Omeka S
- Ensure correct base URL format

### 2. Text Chunking Issues
- Large documents are automatically split
- Adjust `max_length` parameter if needed
- Monitor token usage for Gemini API

### 3. Database Update Failures
- Check property ID for `bibo:content` (default: 91)
- Verify item permissions
- Review retry mechanism logs

## Output Quality

The AI correction process improves:
- **Character accuracy**: Fixes common OCR misrecognitions
- **Text structure**: Reconstructs paragraphs and formatting
- **Readability**: Improves spacing and punctuation
- **Consistency**: Standardizes formatting across documents

## Monitoring Progress

Each script provides:
- Real-time progress bars
- Success/failure counts
- Detailed error reporting
- Processing statistics

## Advanced Usage

### Custom Correction Prompts
Modify `ocr_correction_prompt.md` to:
- Adjust correction strategies
- Add domain-specific rules
- Handle specialized text types

### Batch Processing
For large collections:
- Process in smaller batches
- Monitor API quotas
- Use background processing

### Quality Assessment
- Compare before/after samples
- Monitor correction accuracy
- Adjust parameters as needed

## Support

For issues or questions:
- Check the logs for detailed error information
- Review the main project README
- Verify environment configuration
- Test with small batches first
