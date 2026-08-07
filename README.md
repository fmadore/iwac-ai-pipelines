# IWAC AI Pipelines

Python workflows for processing the [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) (IWAC) using Large Language Models.

[![Islam West Africa Collection](https://img.shields.io/badge/Collection-IWAC-blue)](https://islam.zmo.de/s/westafrica/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21804210.svg)](https://doi.org/10.5281/zenodo.21804210)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Context

The [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) is an open-access digital database documenting Islam and Muslim communities in Benin, Burkina Faso, Côte d'Ivoire, Niger, Nigeria, and Togo since the 1960s. Created by [Frédérick Madore](https://www.frederickmadore.com/) and hosted at the Leibniz-Zentrum Moderner Orient (ZMO) in Berlin, the collection contains over 14,500 items and 28 million words of text.

At this scale, traditional manual processing—metadata tagging, OCR correction, entity identification—is no longer viable. These pipelines use LLMs from Google Gemini, OpenAI, and Mistral, plus open-weights models (Qwen, DeepSeek) reached through OpenRouter, to automate labor-intensive tasks that would otherwise leave much of the corpus inaccessible.

## What These Tools Do

| Pipeline | Purpose |
|----------|---------|
| **OCR Extraction** | Extract text from PDF scans using Gemini vision or Mistral Document AI |
| **OCR Correction** | Fix errors in machine-generated text, including ALTO XML with coordinate preservation |
| **Named Entity Recognition** | Extract people, places, organizations with authority reconciliation |
| **Summarization** | Generate bilingual French/English summaries for document discovery |
| **Audio Transcription** | Transcribe interviews and oral histories using Gemini or Voxtral (with speaker diarization) |
| **Video Processing** | Summarize or transcribe video with visual descriptions |
| **Handwritten Text Recognition** | Read manuscripts in French, Arabic, or mixed languages |
| **Magazine Article Extraction** | Index individual articles within digitized periodicals |
| **Sentiment Analysis** | Evaluate centrality, subjectivity, and polarity of Islam/Muslim representations with a four-model panel |
| **Reference Indexing** | Assign controlled subject and spatial keywords to scholarly references using Claude, with authority reconciliation |

## Limitations and Caveats

These tools are research aids, not replacements for scholarly judgment. Users should be aware of several constraints:

**Algorithmic opacity.** LLMs operate as black boxes. We cannot fully trace their decision pathways, which challenges the transparency expected in historical scholarship. This project documents prompts, model versions, and processing parameters, but the models' internal reasoning remains opaque.

**Western-centric bias.** Models trained predominantly on Western data may misrepresent African contexts, linguistic nuances, and naming conventions. The NER pipeline includes fuzzy matching and human review stages to catch errors, but some will inevitably pass through.

**Hallucinations.** Unlike traditional OCR, which signals failure through garbled text, AI-generated errors appear as fluent prose. The cognitive burden shifts from fixing visible mistakes to detecting hidden ones. Original documents are preserved alongside AI outputs for verification.

**Non-determinism.** Running the same text through a model twice may yield different results, complicating reproducibility.

**The human in the loop.** Effective use of these tools requires domain expertise. Prior familiarity with the source material is necessary to audit AI outputs and distinguish genuine insight from plausible-sounding error.

## Installation

Requires **Python >= 3.11** (3.13+ recommended for the audio pipelines, which rely on `audioop-lts` to replace the `audioop` module removed from the standard library).

```bash
git clone https://github.com/fmadore/iwac-ai-pipelines.git
cd iwac-ai-pipelines
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

For running the tests and linter, install the optional dev extras instead:

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file:

```env
# AI Providers (at least one required)
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Omeka S connection (for database integration)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential
```

## Usage

Each pipeline directory contains numbered scripts to run in sequence:

```bash
cd AI_ocr_extraction/
python 01_omeka_pdf_downloader.py   # Download source PDFs
python 02_gemini_ocr_processor.py   # Extract text
python 03_omeka_content_updater.py  # Update database
```

Most scripts support both interactive mode and command-line flags:

```bash
python 01_NER_AI.py --item-set-id 123 --model gemini-flash
```

### Supported Models

| Provider | Key | Notes |
|----------|-----|-------|
| OpenAI | `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` | Text pipelines only. GPT-5.6 tiers: Luna (cheapest), Terra (balanced), Sol (flagship). Legacy `gpt-5-mini` / `gpt-5.1` keys still resolve to Luna / Sol. |
| Gemini | `gemini-flash`, `gemini-flash-lite`, `gemini-pro` | Text and multimodal |
| Gemma  | `gemma-4` | Google Gemma 4 31B open-weights flagship, served via the Gemini API (shares `GEMINI_API_KEY`); text + image only, no audio. Supports only `MINIMAL` or `HIGH` thinking levels. Currently wired into NER and OCR extraction. |
| Mistral | `mistral-large`, `ministral-14b` | Text pipelines; dedicated OCR and audio transcription endpoints |
| OpenRouter | `deepseek-v4-flash-0731` (default), Qwen and legacy/quality options | DeepSeek V4 Flash 0731 is the default for every text-generation stage, including sentiment and magazine consolidation. It is text-only: PDF/image/audio/video extraction still uses the modality-specific Gemini, Mistral, or Voxtral APIs. Requests are routed only to backends that do not retain data. |

`deepseek-v4-flash-0731` is pinned to the dated OpenRouter slug
`deepseek/deepseek-v4-flash-0731`; the generic aliases `deepseek` and
`deepseek-flash` resolve to it. The former `deepseek-v4-flash` key remains
available only to reproduce runs made with the superseded preview.

## Adapting for Other Projects

These tools were built for IWAC but can be modified for other collections:

- **Prompts** are stored as `.md` files in each pipeline directory and can be edited for different contexts, languages, or document types
- **Pipelines** are modular and can be used independently
- **Shared utilities** (`common/`) centralize Omeka S API access (`omeka_client.py`), the model catalog (`llm_registry.py`), provider adapters (`llm_provider.py`), durable checkpoints (`checkpoint.py`), the page-by-page Gemini PDF loop (`gemini_page_processor.py`), and idempotent text/resource-link writes (`omeka_text_updater.py`, `omeka_link_updater.py`) — so a pipeline is mostly its prompts and its choice of model

The approach assumes you have digitized materials and need to make them searchable. It is designed for institutions and researchers managing substantial digital collections with limited resources.

## Documentation

- [Shared Utilities](common/README.md) — OmekaClient, LLM provider configuration, the Gemini page processor and the Omeka text updater
- [Magazine Article Extraction](AI_summary_issue/README.md) — Article indexing from digitized periodicals (Gemini, Mistral, or Claude agent)
- [Reference Indexing](AI_reference_indexing/README.md) — Subject and spatial keyword assignment for scholarly references
- [IWAC on Hugging Face](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) — Full dataset
- Individual pipeline directories contain their own documentation

## Related Resources

- [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) — The digital collection
- [AI-NER-Validator](https://github.com/fmadore/AI-NER-Validator) — Web app for reviewing and validating NER results
- [Leibniz-Zentrum Moderner Orient](https://www.zmo.de/en) — Host institution

## Citation

If you use these tools in your research, please cite the software:

Madore, Frédérick. *IWAC AI Pipelines*. 2026. https://doi.org/10.5281/zenodo.21804210

That DOI resolves to the latest release; each version also has its own. GitHub's
"Cite this repository" button generates the same reference from `CITATION.cff`.

For the accompanying article:

Madore, Frédérick. "When AI Meets the Archive: Transforming the Islam West Africa Collection with Large Language Models." forthcoming.

## License

MIT. See individual pipeline directories for additional notes.

---

These workflows represent one approach to managing digital abundance in under-resourced archival contexts. They do not solve the fundamental challenges of algorithmic opacity or Western-centric bias in AI systems, but they offer a documented, auditable method for processing materials that would otherwise remain inaccessible.
