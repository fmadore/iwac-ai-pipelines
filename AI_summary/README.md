# AI Summary Pipeline (Gemini, OpenAI, or Mistral)

This pipeline automatically generates French summaries of text content from an Omeka S database using:

1. **Google Gemini** (gemini-3-flash-preview) - Fast and cost-effective
2. **OpenAI** (gpt-5-mini) - Cost-optimized with reasoning capabilities
3. **Mistral** (ministral-14b) - Fast and affordable ($0.2/M tokens)

The user selects the provider interactively at runtime. The workflow extracts full text, generates concise French summaries (RAG-friendly), then updates Omeka S with the results.

## Pipeline Overview

The pipeline follows this workflow:

1. **Extract Content** (`01_extract_omeka_content.py`) - Retrieves OCR text from Omeka S items
2. **Generate Summaries** (`02_AI_generate_summaries.py`) - Creates French summaries (OpenAI or Gemini)
3. **Update Summaries** (`03_omeka_update_summaries.py`) - Updates Omeka S items with generated summaries

## Files Structure

```
AI_summary/
├── 01_extract_omeka_content.py    # Step 1: Extract OCR text from Omeka S
├── 02_AI_generate_summaries.py     # Step 2: Generate French summaries (Gemini or OpenAI)
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

# OpenAI API (required if selecting OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# Google Gemini API (required if selecting Gemini models)
GEMINI_API_KEY=your_gemini_api_key
# OR use Application Default Credentials
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Mistral API (required if selecting Mistral models)
MISTRAL_API_KEY=your_mistral_api_key
```

### Required Python Packages

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - For API communication
- `tqdm` - For progress bars
- `rich` - For beautiful console output
- `python-dotenv` - For environment variables
- `google-genai` - For Gemini AI integration
- `openai` - For OpenAI Responses API integration
- `mistralai` - For Mistral AI integration

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

### Step 2: Generate French Summaries (Select AI Provider at Runtime)

```bash
python 02_AI_generate_summaries.py
```

You will be prompted with the available cost-effective models:
- **GPT-5 mini** (OpenAI) - Fast with reasoning capabilities
- **Gemini 2.5 Flash** (Google) - Fast and cost-effective
- **Ministral 14B** (Mistral) - Affordable at $0.2/M tokens

Select the desired option and the script will:

* Display a styled configuration panel with all settings
* Read all `.txt` files from `TXT/`
* Load the shared prompt template `summary_prompt.md`
* Call the chosen provider's API
* Write summaries to `Summaries_FR_TXT/`
* Show progress with styled progress bars
* Display a final results panel with success/error counts

Provider-specific requirements:
* OpenAI: `OPENAI_API_KEY`
* Gemini: `GEMINI_API_KEY` OR `GOOGLE_APPLICATION_CREDENTIALS`
* Mistral: `MISTRAL_API_KEY`

Output: One summary file per input (`{item_id}.txt`) in `Summaries_FR_TXT/`.

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

This file contains the prompt template used by both AI providers. The template includes:

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

| Provider | Primary Auth | Alternative | Notes |
|----------|--------------|-------------|-------|
| OpenAI   | `OPENAI_API_KEY` | — | Required for GPT-5 mini |
| Gemini   | `GOOGLE_APPLICATION_CREDENTIALS` (ADC) | `GEMINI_API_KEY` | Script attempts ADC first, then API key |
| Mistral  | `MISTRAL_API_KEY` | — | Required for Ministral 14B |

If credentials for multiple providers are present, you can switch freely per run.

## Troubleshooting

### Common Issues

1. **Missing environment variables:**
   - Ensure all required variables are set in your `.env` file
   - Check that the `.env` file is in the correct location

2. **AI provider authentication failures:**
   - OpenAI: confirm `OPENAI_API_KEY` value and account quota
   - Gemini: verify ADC path or `GEMINI_API_KEY` and quota
   - Mistral: verify `MISTRAL_API_KEY` value and account quota

3. **Omeka S connection issues:**
   - Verify your Omeka S base URL and API credentials
   - Ensure the API is enabled on your Omeka S instance

4. **Missing prompt template:**
   - Ensure `summary_prompt.md` exists in the same directory as the scripts
   - The script will fail without this file (no fallback)

### Performance Tuning

- Adjust `MAX_WORKERS` in `01_extract_omeka_content.py` for concurrent processing
- Modify `ITEMS_PER_PAGE` for API pagination size
- Adjust temperature or model constants inside `02_AI_generate_summaries.py` (`GEMINI_MODEL`, `OPENAI_MODEL`)

## Customization

### Modifying the Summary Prompt

Edit `summary_prompt.md` to change:
- Summary language (currently French)
- Summary length and style
- Specific instructions for your use case
- Output format requirements

### Changing the AI Model

The script uses the shared `common/llm_provider.py` module for model selection. Available cost-effective models for summaries:

| Key | Provider | Model | Description |
|-----|----------|-------|-------------|
| `gpt-5-mini` | OpenAI | gpt-5-mini | Cost-optimized with reasoning |
| `gemini-flash` | Google | gemini-3-flash-preview | Fast and cost-effective |
| `ministral-14b` | Mistral | ministral-14b-2512 | Affordable ($0.2/M tokens) |

To add more models or change defaults, edit `common/llm_provider.py` and update the `allowed_keys` in the script.

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
- **Google Gemini / OpenAI APIs:** Refer to respective provider documentation
- **Pipeline-specific issues:** Check the logs for detailed error messages
