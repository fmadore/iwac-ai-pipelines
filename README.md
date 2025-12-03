# IWAC AI Pipelines

**Making African Islamic archives searchable and accessible through AI**

[![Islam West Africa Collection](https://img.shields.io/badge/Collection-IWAC-blue)](https://islam.zmo.de/s/westafrica/)

## The Challenge: Unlocking Hidden African Archives

African archives face a paradox: while the continent holds rich historical documentation, these materials often remain invisible to global scholarship. Colonial-era extraction moved many sources to European institutions, while materials that stayed in Africa frequently lack the infrastructure for digitization, cataloging, and online access.

The **[Islam West Africa Collection](https://islam.zmo.de/s/westafrica/)** (IWAC) was created by **[Frédérick Madore](https://www.frederickmadore.com/)** to address this gap. Hosted at the Leibniz-Zentrum Moderner Orient (ZMO) in Berlin, IWAC is a **free, open-access digital database** documenting Islam and Muslim communities in **Benin, Burkina Faso, Côte d'Ivoire, Niger, and Togo**.

### What's in the Collection?

- **14,500+ digitized items**
- **26+ million words** of searchable text
- Newspapers, magazines, photographs, audio recordings, manuscripts
- Primarily in **French**, with materials also in **Arabic, Hausa**, and local languages
- Sources from the postcolonial period

### The Problem We're Solving

Processing this volume of material manually would take years and significant funding—resources rarely available for African studies projects. Traditional archival work for a collection this size might require:

- A team of trained catalogers working for years
- Traditional OCR that struggles with complex newspaper layouts and multilingual text
- Subject matter experts for entity identification

**This repository contains the AI tools that made it possible for one person to process the entire IWAC collection.**

## How AI Helps

These pipelines use Large Language Models from **Google Gemini**, **OpenAI**, and **Mistral** to automate tasks that would otherwise require extensive human labor:

| Task | Traditional Approach | AI Pipeline |
|------|---------------------|-------------|
| **OCR Extraction** | Manual transcription | Native PDF processing with Gemini vision or Mistral Document AI |
| **OCR Correction** | Manual proofreading | Automated correction with context awareness |
| **Named Entity Recognition** | Expert catalogers | AI extraction + authority reconciliation |
| **Summarization** | Reading each document | Automated French summaries |
| **Handwritten Text** | Specialist paleographers | Vision AI transcription (French, Arabic, multilingual) |
| **Audio/Video Transcription** | Manual transcription | Speech-to-text with translation and visual descriptions |

The result: tasks that might take months can be completed in days, freeing researchers to focus on interpretation rather than data entry.

## Available Pipelines

### 🎙️ Audio Transcription (`AI_audio_summary/`)
Transcribes interviews, sermons, and oral histories using Google Gemini. Supports multiple audio and video formats including **Hausa**. Features automatic splitting for long recordings, speaker detection, and video-to-audio conversion.

### 🎬 Video Processing (`AI_video_summary/`)
Processes video files for summarization or full transcription with periodic visual descriptions. Uses Gemini's multimodal capabilities to analyze both audio and visual content. Handles files of any size through automatic upload management.

### ✍️ Handwritten Text Recognition (`AI_htr_extraction/`)
Reads handwritten documents—colonial correspondence, manuscript annotations, personal letters—in **French, Arabic, or mixed languages**. Uses specialized prompts for each language with proper typography handling. Essential for materials that standard OCR cannot process.

### 🔍 Named Entity Recognition (`AI_NER/`)
Extracts people, places, organizations, and topics from documents using **Gemini, OpenAI, or Mistral**. Reconciles entities against authority databases and updates Omeka S with structured metadata. Features native structured outputs for guaranteed valid JSON.

### 📝 OCR Correction (`AI_ocr_correction/`)
Fixes errors in machine-generated text from scanned documents. Uses the shared LLM provider supporting **Gemini, OpenAI, and Mistral**. Particularly effective for historical newspapers where print quality varies.

### 📄 OCR Extraction (`AI_ocr_extraction/`)
Extracts text from PDF scans using **Gemini's vision capabilities** or **Mistral Document AI**. Handles complex layouts common in newspapers and magazines. Includes intelligent copyright handling with automatic RECITATION error recovery.

### 📋 Summary Generation (`AI_summary/`)
Creates concise French summaries for each document using **Gemini, OpenAI, or Mistral**. RAG-friendly output enables quick assessment of relevance without reading full texts.

### 📰 Magazine Article Extraction (`AI_summary_issue/`)
Identifies and indexes individual articles within digitized magazine issues using **Gemini** or **Mistral**. Two-step pipeline: page-by-page extraction followed by cross-page consolidation. Crucial for periodicals where each issue contains dozens of separate pieces.

### 🧩 NotebookLM Exporter (`NotebookLM/`)
Exports collection data for use with Google's NotebookLM, enabling AI-powered research conversations with the archive. Supports whole collection, single item set, or subject-based exports with automatic file splitting.

## Getting Started

### Prerequisites

- **Python 3.8+**
- **API Keys** (at least one):
  - [Google Gemini API](https://aistudio.google.com/app/api-keys) — Required for most pipelines (free tier available)
  - [OpenAI API](https://platform.openai.com/api-keys) — Alternative for text pipelines
  - [Mistral API](https://console.mistral.ai/api-keys) — Alternative for text pipelines + dedicated OCR endpoint
- **Omeka S credentials** (if connecting to an Omeka database)

### Installation

```bash
# Clone the repository
git clone https://github.com/fmadore/iwac-ai-pipelines.git
cd iwac-ai-pipelines

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Create a `.env` file with your credentials:

```env
# AI Providers (at least one required)
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key

# Omeka S connection (for database integration)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential
```

### Supported Models

| Provider | Model Key | Description |
|----------|-----------|-------------|
| OpenAI | `gpt-5-mini` | Cost-optimized, fast |
| OpenAI | `gpt-5.1` | Flagship model |
| Gemini | `gemini-flash` | Gemini 2.5 Flash — fast, cost-effective |
| Gemini | `gemini-pro` | Gemini 3.0 Pro — highest quality |
| Mistral | `mistral-large` | Mistral Large 3 — 41B active params |
| Mistral | `ministral-14b` | Ministral 3 14B — fast, affordable |

Most pipelines let you select the model interactively or via `--model` flag.

### Quick Example: Transcribe Audio

```bash
cd AI_audio_summary/

# Place audio/video files in the Audio/ folder
# Run the script (interactive prompts guide you)
python 02_AI_transcribe_audio.py

# Or specify model directly
python 02_AI_transcribe_audio.py --model gemini-2.5-flash

# Output appears in Transcriptions/
```

### Quick Example: Extract Entities

```bash
cd AI_NER/

# Process an Omeka item set (interactive model selection)
python 01_NER_AI.py --item-set-id 123

# Or specify model directly
python 01_NER_AI.py --item-set-id 123 --model gemini-flash

# Reconcile against authority databases
python 02_NER_reconciliation_Omeka.py

# Update the database
python 03_Omeka_update.py
```

### Quick Example: Generate Summaries

```bash
cd AI_summary/

# Extract content from Omeka
python 01_extract_omeka_content.py

# Generate summaries (choose Gemini, OpenAI, or Mistral)
python 02_AI_generate_summaries.py

# Update Omeka with summaries
python 03_omeka_update_summaries.py
```

## Who Is This For?

- **Archivists** working with under-resourced collections
- **Digital humanities researchers** processing multilingual sources
- **African studies scholars** seeking to make sources accessible
- **Librarians** enhancing discoverability of historical materials
- **Anyone** working with large document collections

## Adapting for Your Project

While these tools were built for IWAC, they can be adapted for other archival projects:

1. **Prompts are customizable** — Edit the `.md` prompt files to match your collection's context
2. **Modular design** — Use individual pipelines independently
3. **Multiple AI providers** — Switch between Gemini, OpenAI, and Mistral based on cost/quality/speed needs
4. **Language flexibility** — Prompts can be modified for other languages
5. **Shared LLM provider** — Text pipelines use `common/llm_provider.py` for unified model selection

## Technical Notes

- **Shared LLM provider** — Text pipelines route through `common/llm_provider.py` for consistent model selection and configuration
- **Multimodal exceptions** — Audio, video, HTR, and OCR pipelines use provider APIs directly for file uploads and vision features
- Each pipeline directory contains numbered scripts (run in sequence)
- **Rich console output** — Beautiful progress bars, tables, and status indicators using the `rich` library
- Comprehensive error handling and retry logic for API calls
- Progress tracking for batch operations
- Detailed logging for troubleshooting

## Learn More

- **[Islam West Africa Collection](https://islam.zmo.de/s/westafrica/)** — Explore the archive
- **[IWAC on Hugging Face](https://huggingface.co/datasets/fmadore/islam-west-africa-collection)** — Download the dataset
- **[Leibniz-Zentrum Moderner Orient](https://www.zmo.de/en)** — Host institution
- **[Frédérick Madore](https://www.frederickmadore.com/)** — Project creator
- **[LLM Provider Documentation](common/README.md)** — Configuration guide for model selection
- Individual pipeline documentation in each folder's README

## Contributing

Contributions welcome! Please:
1. Test with small datasets first
2. Update documentation for changes
3. Follow existing code patterns

## Citation

If you use these tools in your research, please cite the IWAC collection and consider acknowledging this repository.

## License

Designed for academic research and archival purposes. See individual pipeline directories for specific notes.

---

*These tools demonstrate how AI can help democratize access to under-resourced archives. By reducing the labor required for processing, we can make more of Africa's documentary heritage visible and searchable for researchers worldwide.*