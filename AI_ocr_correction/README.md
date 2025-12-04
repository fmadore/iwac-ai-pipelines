# OCR Post-Correction Pipeline

This directory contains pipelines for extracting, correcting, and updating OCR text using AI-powered post-processing with multiple LLM providers (Gemini, OpenAI, Mistral).

## Overview

The OCR post-correction pipeline processes raw OCR text through AI-powered correction and can upload the improved text back to the database. Two correction approaches are available:

1. **Plain Text Correction** (`02_correct_ocr_text.py`): Corrects raw OCR text files
2. **ALTO XML Correction** (`02_correct_alto_xml.py`): Corrects OCR in ALTO XML while preserving coordinates

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

### 2a. AI-Powered Text Correction (`02_correct_ocr_text.py`)
**Purpose**: Applies AI-powered corrections to raw OCR text using multiple LLM providers.

**Features**:
- Multi-provider support: Gemini, OpenAI, Mistral
- Rich console output with progress tracking
- Intelligent text chunking for large documents
- Structure analysis and paragraph reconstruction
- Character recognition error correction
- Historical text preservation

**Supported Models**:
- `gemini-flash`: Gemini 2.5 Flash (default, fast, thinking disabled)
- `gemini-pro`: Gemini 3 Pro (highest quality)
- `gpt-5-mini`: OpenAI GPT-5 mini (cost-optimized)
- `gpt-5.1`: OpenAI GPT-5.1 (flagship)
- `mistral-large`: Mistral Large 3
- `ministral-14b`: Ministral 3 14B (fast, cost-effective)

**Usage**:
```bash
# Interactive model selection
python 02_correct_ocr_text.py

# Use specific model
python 02_correct_ocr_text.py --model gemini-flash

# Custom directories
python 02_correct_ocr_text.py --model gpt-5-mini --input-dir ./TXT --output-dir ./Corrected_TXT
```

**Input**: Raw OCR text files from `TXT/` directory
**Output**: Corrected text files in `Corrected_TXT/` directory

### 2b. ALTO XML OCR Correction (`02_correct_alto_xml.py`) ⭐ NEW
**Purpose**: Corrects OCR text in ALTO XML files while **preserving all coordinates and layout**.

This is the recommended approach when you have ALTO XML files and need to:
- Maintain precise word-level coordinates (HPOS, VPOS, WIDTH, HEIGHT)
- Keep the XML structure intact for downstream processing
- Preserve page layout information

**How it works**:
1. Parses ALTO XML and extracts `<TextLine>` contents
2. Sends text lines to LLM for correction using **structured output**
3. Enforces strict 1:1 token alignment (same number of words in/out)
4. Updates only `CONTENT` attributes, preserving all coordinates
5. Writes corrected ALTO XML with original structure intact

**Features**:
- Structured output with Pydantic models for reliable token alignment
- Batch processing (configurable lines per API call)
- Multi-provider support (Gemini, OpenAI, Mistral)
- Automatic ALTO namespace detection (v2, v3, v4)
- Rich console output with progress tracking

**Usage**:
```bash
# Interactive model selection
python 02_correct_alto_xml.py

# Use Gemini Flash (fast, no thinking)
python 02_correct_alto_xml.py --model gemini-flash

# Custom directories and batch size
python 02_correct_alto_xml.py --model gemini-flash --input-dir ./ALTO --output-dir ./ALTO_Corrected --batch-size 30
```

**Input**: ALTO XML files from `ALTO/` directory
**Output**: Corrected ALTO XML files in `ALTO_Corrected/` directory

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
AI_ocr_correction/
├── 01_extract_ocr_text.py          # Step 1: Extract raw OCR text
├── 02_correct_ocr_text.py          # Step 2a: Plain text correction
├── 02_correct_alto_xml.py          # Step 2b: ALTO XML correction (preserves coordinates)
├── 03_update_database.py           # Step 3: Update database
├── ocr_correction_prompt.md        # Prompt for plain text correction
├── alto_correction_prompt.md       # Prompt for ALTO XML correction (token-aligned)
├── TXT/                            # Raw OCR text files
├── Corrected_TXT/                  # AI-corrected text files
├── ALTO/                           # Input ALTO XML files
├── ALTO_Corrected/                 # Corrected ALTO XML files
└── README.md                       # This file
```

## Prerequisites

- Python 3.8+
- Required packages (see parent directory's `requirements.txt`)
- Environment variables configured (see Setup section)

## Setup

1. **Environment Configuration**:
   Copy the example environment file and configure your API credentials:
   ```bash
   cp ../.env.example ../.env
   ```
   
   Edit the `.env` file in the parent directory with your actual values:
   ```env
   OMEKA_BASE_URL=https://your-omeka-instance.com/api
   OMEKA_KEY_IDENTITY=your_key_identity
   OMEKA_KEY_CREDENTIAL=your_key_credential
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key      # Optional
   MISTRAL_API_KEY=your_mistral_api_key    # Optional
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

## Usage

### Workflow A: Plain Text Correction

```bash
# Step 1: Extract OCR text from Omeka S
python 01_extract_ocr_text.py

# Step 2: Apply AI corrections to plain text
python 02_correct_ocr_text.py --model gemini-flash

# Step 3: Update Omeka S with corrected text
python 03_update_database.py
```

### Workflow B: ALTO XML Correction (Preserves Coordinates)

```bash
# Place your ALTO XML files in the ALTO/ directory

# Correct ALTO XML while preserving coordinates
python 02_correct_alto_xml.py --model gemini-flash

# Corrected files will be in ALTO_Corrected/
```

## ALTO XML Correction Details

### Token Alignment Strategy

The ALTO correction uses **structured output** to enforce strict token alignment:

```python
class CorrectedLine(BaseModel):
    line_index: int
    original_tokens: list[str]   # ["Tlie", "gouvernernent", "lias"]
    corrected_tokens: list[str]  # ["The", "gouvernement", "has"]
    # ↑ Must be same length!
```

This ensures:
- Each `<String>` element maps 1:1 to a corrected token
- Coordinates (HPOS, VPOS, WIDTH, HEIGHT) remain untouched
- XML structure is preserved exactly

### What Gets Corrected

✅ **Corrected**:
- OCR character errors: `Tlie` → `The`, `rn` → `m`
- Encoding issues: `cafÃ©` → `café`
- Obvious misspellings: `gouvernernent` → `gouvernement`

❌ **Preserved**:
- Historical spellings: `connexion`, `shew`, `to-day`
- Original language: French stays French
- Token boundaries: Never merge or split words

### Example

**Original ALTO XML**:
```xml
<TextLine>
  <String CONTENT="Tlie" HPOS="100" VPOS="50" WIDTH="40" HEIGHT="20"/>
  <String CONTENT="gouvernernent" HPOS="145" VPOS="50" WIDTH="120" HEIGHT="20"/>
  <String CONTENT="lias" HPOS="270" VPOS="50" WIDTH="35" HEIGHT="20"/>
</TextLine>
```

**After Correction**:
```xml
<TextLine>
  <String CONTENT="The" HPOS="100" VPOS="50" WIDTH="40" HEIGHT="20"/>
  <String CONTENT="gouvernement" HPOS="145" VPOS="50" WIDTH="120" HEIGHT="20"/>
  <String CONTENT="has" HPOS="270" VPOS="50" WIDTH="35" HEIGHT="20"/>
</TextLine>
```

Note: Only `CONTENT` changed; all coordinates preserved!

## Configuration

### Model Selection

| Model | Provider | Speed | Cost | Best For |
|-------|----------|-------|------|----------|
| `gemini-flash` | Google | Fast | Low | Bulk processing |
| `gemini-pro` | Google | Slow | High | Complex documents |
| `gpt-5-mini` | OpenAI | Fast | Low | General use |
| `gpt-5.1` | OpenAI | Slow | High | Maximum accuracy |
| `mistral-large` | Mistral | Medium | Medium | European languages |
| `ministral-14b` | Mistral | Fast | Low | Cost-sensitive |

### Gemini Thinking Mode

- **gemini-flash**: Thinking disabled (`thinking_budget=0`) for speed
- **gemini-pro**: Low thinking (`thinking_level="low"`) for balance

### Processing Parameters
- **Batch size**: 20 lines per API call (configurable)
- **Temperature**: 0.1-0.2 for consistent corrections
- **Max text length**: 200,000 characters before chunking (plain text)

## Error Handling

All scripts include comprehensive error handling:
- **Network errors**: Automatic retry with exponential backoff
- **Token count mismatch**: Falls back to original text
- **API rate limits**: Built-in delays and retry mechanisms
- **XML parse errors**: Graceful handling with detailed logging
- **Processing failures**: Continue processing remaining items

## Output Quality

The AI correction process improves:
- **Character accuracy**: Fixes common OCR misrecognitions
- **Text structure**: Reconstructs paragraphs and formatting (plain text)
- **Coordinate preservation**: Maintains exact layout (ALTO XML)
- **Historical authenticity**: Preserves period-appropriate language

## Support

For issues or questions:
- Check the logs for detailed error information
- Review the main project README
- Verify environment configuration
- Test with small batches first
