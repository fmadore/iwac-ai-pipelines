# AI Summary Pipeline

This pipeline automatically generates French summaries of text content from an Omeka S database using Google's Gemini 2.5 Flash model. The pipeline consists of three main steps that work together to extract, summarize, and update content.

## Pipeline Overview

The pipeline follows this workflow:

1. **Extract Content** (`01_extract_omeka_content.py`) - Retrieves OCR text from Omeka S items
2. **Generate Summaries** (`02_gemini_generate_summaries.py`) - Creates French summaries using Gemini AI
3. **Update Summaries** (`03_omeka_update_summaries.py`) - Updates Omeka S items with generated summaries

## Files Structure

```
AI_summary/
├── 01_extract_omeka_content.py    # Step 1: Extract OCR text from Omeka S
├── 02_gemini_generate_summaries.py # Step 2: Generate French summaries
├── 03_omeka_update_summaries.py   # Step 3: Update Omeka S with summaries
├── summary_prompt.md              # AI prompt template for summary generation
├── README.md                      # This file
├── TXT/                           # Directory for extracted text files (created by step 1)
└── Summaries_FR_TXT/             # Directory for generated summaries (created by step 2)
```

## Prerequisites

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Omeka S API Configuration
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_api_key_identity
OMEKA_KEY_CREDENTIAL=your_api_key_credential

# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key
# OR use Google Application Default Credentials
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
```

### Required Python Packages

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - For API communication
- `tqdm` - For progress bars
- `python-dotenv` - For environment variables
- `google-genai` - For Gemini AI integration

## Step-by-Step Usage

### Step 1: Extract Content from Omeka S

```bash
python 01_extract_omeka_content.py
```

**What it does:**
- Connects to your Omeka S instance using API credentials
- Prompts you to enter item set ID(s) (comma or space separated)
- Fetches all items from the specified item set(s)
- Extracts OCR text from the `bibo:content` field
- Saves individual text files to the `TXT/` directory
- Skips items with no content and provides detailed logging

**Output:** Text files named `{item_id}.txt` in the `TXT/` directory

### Step 2: Generate French Summaries

```bash
python 02_gemini_generate_summaries.py
```

**What it does:**
- Reads all `.txt` files from the `TXT/` directory
- Uses the prompt template from `summary_prompt.md`
- Generates French summaries using Google's Gemini 2.5 Flash model
- Saves summaries to the `Summaries_FR_TXT/` directory
- Provides progress tracking and error handling

**Requirements:**
- `summary_prompt.md` file must exist (contains the AI prompt template)
- Valid Gemini API key or Google Application Default Credentials
- Text files in the `TXT/` directory from Step 1

**Output:** Summary files named `{item_id}.txt` in the `Summaries_FR_TXT/` directory

### Step 3: Update Omeka S Items

```bash
python 03_omeka_update_summaries.py
```

**What it does:**
- Reads all summary files from the `Summaries_FR_TXT/` directory
- Updates corresponding Omeka S items with the generated summaries
- Adds summaries to the `bibo:shortDescription` field (property ID 116)
- Preserves existing item data while adding/updating summaries
- Provides progress tracking and error handling

**Output:** Updated Omeka S items with French summaries in the `bibo:shortDescription` field

## Configuration Files

### `summary_prompt.md`

This file contains the prompt template used by the Gemini AI model. The template includes:

- Instructions for generating concise French summaries
- Specific requirements for RAG (Retrieval-Augmented Generation) compatibility
- Output format specifications
- A `{text}` placeholder that gets replaced with the actual content

**Important:** This file is required for Step 2. The script will fail if it's missing (no hardcoded fallback).

## Error Handling and Logging

All scripts include comprehensive error handling and logging:

- **Concurrent processing** with configurable thread pools
- **Retry mechanisms** for API calls
- **Progress tracking** with `tqdm` progress bars
- **Detailed logging** of operations and errors
- **Graceful handling** of missing files, empty content, and API failures

## Authentication Methods

The pipeline supports two authentication methods for Google Gemini:

1. **API Key Method:**
   - Set `GEMINI_API_KEY` in your `.env` file
   - Simpler setup, good for development

2. **Application Default Credentials (ADC):**
   - Set `GOOGLE_APPLICATION_CREDENTIALS` pointing to your service account JSON
   - More secure, recommended for production

The script will try ADC first, then fall back to API key if ADC fails.

## Troubleshooting

### Common Issues

1. **Missing environment variables:**
   - Ensure all required variables are set in your `.env` file
   - Check that the `.env` file is in the correct location

2. **Gemini API authentication failures:**
   - Verify your API key is valid and has appropriate permissions
   - Check if you have sufficient quota/credits

3. **Omeka S connection issues:**
   - Verify your Omeka S base URL and API credentials
   - Ensure the API is enabled on your Omeka S instance

4. **Missing prompt template:**
   - Ensure `summary_prompt.md` exists in the same directory as the scripts
   - The script will fail without this file (no fallback)

### Performance Tuning

- Adjust `MAX_WORKERS` in `01_extract_omeka_content.py` for concurrent processing
- Modify `ITEMS_PER_PAGE` for API pagination size
- Configure Gemini model parameters (temperature, etc.) in `02_gemini_generate_summaries.py`

## Customization

### Modifying the Summary Prompt

Edit `summary_prompt.md` to change:
- Summary language (currently French)
- Summary length and style
- Specific instructions for your use case
- Output format requirements

### Changing the AI Model

In `02_gemini_generate_summaries.py`, modify the `MODEL_NAME` constant to use a different Gemini model:

```python
MODEL_NAME = 'gemini-2.0-flash-001'  # or another available model
```

### Adjusting Omeka S Properties

In `03_omeka_update_summaries.py`, modify the property ID and label if using different Omeka S properties:

```python
"property_id": 116,  # Change to your property ID
"property_label": "shortDescription",  # Change to your property label
```

## Security Considerations

- Never commit `.env` files to version control
- Use Application Default Credentials for production deployments
- Regularly rotate API keys and credentials
- Monitor API usage and costs

## Support

For issues related to:
- **Omeka S API:** Check your Omeka S documentation and API configuration
- **Google Gemini API:** Refer to the Google AI documentation
- **Pipeline-specific issues:** Check the logs for detailed error messages
