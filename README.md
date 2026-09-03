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
| **Audio Transcription** | Transcribe interviews and oral histories using Gemini 3.5 Transcribe (word timestamps + diarization), Gemini, or Voxtral |
| **Video Processing** | Summarize or transcribe video with visual descriptions |
| **YouTube Transcription** | Transcribe YouTube-hosted items from their URLs — no download — detecting the spoken languages and correcting the catalogue record from them |
| **Handwritten Text Recognition** | Read manuscripts in French, Arabic, or mixed languages |
| **Magazine Article Extraction** | Index individual articles within digitized periodicals |
| **Sentiment Analysis** | Evaluate centrality, subjectivity, and polarity of Islam/Muslim representations with a five-model panel |
| **Reference Indexing** | Assign controlled subject and spatial keywords to scholarly references, with authority reconciliation and per-link model provenance |

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

# Optional: a model you serve yourself (see serving/README.md)
SELFHOSTED_LLM_BASE_URL=http://localhost:8000/v1
SELFHOSTED_LLM_API_KEY=sk-...
```

## Usage

Each pipeline directory contains numbered scripts to run in sequence:

```bash
cd AI_ocr_extraction/
python 01_omeka_pdf_downloader.py   # Download source PDFs into PDF/
python 02_gemini_ocr_processor.py   # Extract text into OCR_Results/
python 03_omeka_content_updater.py  # Update database
```

Each pipeline directory ships its input folder empty (`PDF/`, `Audio/`,
`video/`, `TXT/`, `ALTO/` hold a `.gitkeep`), so source material can be
dropped in place or fetched by the `01` step; output, cache and backup folders
are created on first run. All of them are git-ignored.

Every step that writes to Omeka goes through the same gate: `--dry-run` reports
what would change, the pre-write payload of every item is dumped to a
`backups/` or `output/` folder first (the only route back from a bulk write),
and a live run asks for confirmation unless `--yes` is passed. Whatever a
pipeline writes carries a value annotation naming the model that produced it —
`iwac:ocrModel`, `iwac:summaryModel`, `iwac:transcriptionModel`, and
`iwac:nerModel` on every subject or place link.

Most scripts support both interactive mode and command-line flags:

```bash
python 01_NER_AI.py --item-set-id 123 --model gemini-3.7-flash
```

### Supported Models

| Provider | Key | Notes |
|----------|-----|-------|
| OpenAI | `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` | Text pipelines only. GPT-5.6 tiers: Luna (cheapest), Terra (balanced), Sol (flagship). Legacy `gpt-5-mini` / `gpt-5.1` keys still resolve to Luna / Sol. |
| Gemini | `gemini-3.7-flash`, `gemini-flash-lite`, `gemini-pro` | Text and multimodal |
| Gemini (speech) | `gemini-3.5-transcribe` | Dedicated speech-to-text via the Interactions API, not the model registry: word-level timestamps and speaker diarization, 82 locales including Hausa but none of Mooré, Dioula, Ewé, Kabyè or Dendi. Takes no prompt. Used by `AI_audio_summary/02c`. |
| Gemma  | `gemma-4` | Google Gemma 4 31B open-weights flagship, served via the Gemini API (shares `GEMINI_API_KEY`); text + image only, no audio. Supports only `MINIMAL` or `HIGH` thinking levels. Currently wired into NER and OCR extraction. |
| Mistral | `mistral-large`, `ministral-14b` | Text pipelines; dedicated OCR and audio transcription endpoints |
| OpenRouter | `deepseek-v4-flash-0731` (default), Qwen and legacy/quality options | DeepSeek V4 Flash 0731 is the shared text default (`DEFAULT_TEXT_MODEL_KEY`), used by NER, OCR correction and magazine consolidation. Summarization is the one exception and defaults to `gpt-5.6-luna` for throughput. It is text-only: PDF/image/audio/video extraction still uses the modality-specific Gemini, Mistral, or Voxtral APIs. Requests are routed only to backends that do not retain data. |
| Self-hosted | `qwen3.8-27b-selfhosted` | Any OpenAI-compatible endpoint you run yourself — vLLM on a GPU cluster, or llama.cpp / LM Studio / TGI locally. Text pipelines only. The address comes from `SELFHOSTED_LLM_BASE_URL`, so no model here is tied to one machine; a model on this route is simply reported as unavailable when the variable is unset. See [`serving/`](serving/README.md). |

`deepseek-v4-flash-0731` is pinned to the dated OpenRouter slug
`deepseek/deepseek-v4-flash-0731`; the generic aliases `deepseek` and
`deepseek-flash` resolve to it. Every DeepSeek Flash run goes to that release:
the earlier `deepseek-v4-flash` preview sits in no model tier, so no pipeline
offers it and no `--model` accepts it. Its registry entry is kept for the
archive alone, so the annotations it already wrote stay attributable.

## Adapting for Other Projects

These tools were built for IWAC but can be modified for other collections:

- **Prompts** are stored as `.md` files in each pipeline directory and can be edited for different contexts, languages, or document types
- **Pipelines** are modular and can be used independently
- **Shared utilities** (`common/`) centralize Omeka S API access (`omeka_client.py`), the model catalog (`llm_registry.py`), provider adapters (`llm_provider.py`), durable checkpoints (`checkpoint.py`), the page-by-page Gemini PDF loop (`gemini_page_processor.py`), and idempotent text/resource-link writes (`omeka_text_updater.py`, `omeka_link_updater.py`) — so a pipeline is mostly its prompts and its choice of model
- **Serving** (`serving/`) is written to be site-agnostic: every cluster-specific value is an environment override, so pointing it at your own hardware — or at any OpenAI-compatible endpoint you already run — means editing one file of defaults, not the pipelines

The approach assumes you have digitized materials and need to make them searchable. It is designed for institutions and researchers managing substantial digital collections with limited resources.

## Documentation

- [Shared Utilities](common/README.md) — OmekaClient, LLM provider configuration, the Gemini page processor and the Omeka text updater
- [Serving Your Own Models](serving/README.md) — running an open-weights model on your own GPU (Slurm + vLLM), the SSH tunnel, and the probe that checks a route's reasoning levels are real
- [Magazine Article Extraction](AI_summary_issue/README.md) — Article indexing from digitized periodicals (Gemini or Mistral)
- [Audio Transcription](AI_audio_summary/README.md) — three transcribers compared, which languages Gemini 3.5 Transcribe actually covers, and why splitting is mandatory once timestamps are on
- [YouTube Transcription](AI_youtube_transcription/README.md) — URL-based transcription with language detection, the measured token budget, and the public-video-only limit
- [Reference Indexing](AI_reference_indexing/README.md) — Subject and spatial keyword assignment for scholarly references
- [Publication Extraction](AI_publication_extraction/README.md) — Structured OCR for journal articles, chapters, books and theses: footnotes and bibliography separated from the body, oversized scans split automatically
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
What changed between releases, and the operational history behind the model
registry and the sentiment panel, is in [CHANGELOG.md](CHANGELOG.md).

For the accompanying article:

Madore, Frédérick. "When AI Meets the Archive: Transforming the Islam West Africa Collection with Large Language Models." forthcoming.

## License

MIT. See individual pipeline directories for additional notes.

---

These workflows represent one approach to managing digital abundance in under-resourced archival contexts. They do not solve the fundamental challenges of algorithmic opacity or Western-centric bias in AI systems, but they offer a documented, auditable method for processing materials that would otherwise remain inaccessible.
