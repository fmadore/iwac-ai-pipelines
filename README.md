# IWAC AI Pipelines

Python workflows for processing the [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) (IWAC) using Large Language Models.

[![Islam West Africa Collection](https://img.shields.io/badge/Collection-IWAC-blue)](https://islam.zmo.de/s/westafrica/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21804210.svg)](https://doi.org/10.5281/zenodo.21804210)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!TIP]
> **Not comfortable with Python and a terminal? Use [Kalamu](https://github.com/fmadore/Kalamu).**
> Kalamu is a free desktop app for Windows and macOS that brings these pipelines to a
> graphical interface: install it, paste an API key, pick a file or a folder, press **Start**.
> It ports their prompts, provider handling and model choices, and checks every result
> against its source side by side, with corrections kept apart from the AI's text.
>
> <a href="https://apps.microsoft.com/detail/9MXT5555BZ1H?referrer=appbadge&mode=direct"><img src="https://get.microsoft.com/images/en-us%20dark.svg" alt="Get it from Microsoft" height="40"></a>
> &nbsp;or the [releases page](https://github.com/fmadore/Kalamu/releases) for macOS and Windows installers.

| Pipeline here | In Kalamu |
|---------------|-----------|
| OCR extraction, HTR, publication extraction | Extract text from PDFs and images, printed or handwritten, in the cloud or offline |
| Audio, video and YouTube transcription | Transcribe with speakers and timestamps, in the cloud or offline with Whisper; YouTube from its link |
| Summarization, NER, sentiment analysis | Analyse texts with ready-made or user-defined fields |
| Magazine article extraction | Detailed table of contents of an issue, each article an item of its own |
| OCR correction | Correct existing text page by page, every change shown against the original |

Kalamu works on files on your computer, organised in projects; it does not read from or
write to an Omeka S database. The `01` download and `03` write-back steps, authority
reconciliation and authority curation remain specific to this repository.

## Context

The [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) is an open-access digital database documenting Islam and Muslim communities in Benin, Burkina Faso, Côte d'Ivoire, Niger, Nigeria, and Togo since the 1960s. It was created by [Frédérick Madore](https://www.frederickmadore.com/) and is hosted at the Leibniz-Zentrum Moderner Orient (ZMO) in Berlin. The collection grows continuously; research reports should state a dated source snapshot and distinguish documents from authority records when reporting counts.

At this scale, traditional manual processing—metadata tagging, OCR correction, entity identification—is no longer viable. These pipelines use LLMs from Google, OpenAI, Mistral and Anthropic, plus open-weights models (Qwen, DeepSeek, Gemma) reached through OpenRouter or served from your own GPU, to automate labor-intensive tasks that would otherwise leave much of the corpus inaccessible.

## What These Tools Do

| Pipeline | Directory | Purpose |
|----------|-----------|---------|
| **OCR Extraction** | [`AI_ocr_extraction/`](AI_ocr_extraction/README.md) | Extract text from PDF scans using Gemini or Claude vision, or Mistral Document AI |
| **OCR Correction** | [`AI_ocr_correction/`](AI_ocr_correction/README.md) | Fix errors in machine-generated text, including ALTO XML with coordinate preservation |
| **Handwritten Text Recognition** | [`AI_htr_extraction/`](AI_htr_extraction/README.md) | Read manuscripts in French, Arabic, or mixed languages |
| **Publication Extraction** | [`AI_publication_extraction/`](AI_publication_extraction/README.md) | Structured OCR for journal articles, chapters, books and theses: footnotes and bibliography separated from the body, citations extracted |
| **Summarization** | [`AI_summary/`](AI_summary/README.md) | Generate bilingual French/English summaries for document discovery |
| **Named Entity Recognition** | [`AI_NER/`](AI_NER/README.md) | Extract people, places, organizations with authority reconciliation |
| **Reference Indexing** | [`AI_reference_indexing/`](AI_reference_indexing/README.md) | Assign controlled subject and spatial keywords to scholarly references, with authority reconciliation and per-link model provenance |
| **Magazine Article Extraction** | [`AI_summary_issue/`](AI_summary_issue/README.md) | Index individual articles within digitized periodicals |
| **Audio Transcription** | [`AI_audio_summary/`](AI_audio_summary/README.md) | Transcribe interviews and oral histories using Gemini 3.5 Transcribe (word timestamps + diarization), Gemini, or Voxtral |
| **Video Processing** | [`AI_video_summary/`](AI_video_summary/README.md) | Summarize or transcribe video with visual descriptions |
| **YouTube Transcription** | [`AI_youtube_transcription/`](AI_youtube_transcription/README.md) | Transcribe YouTube-hosted items from their URLs — no download — detecting the spoken languages and correcting the catalogue record from them |
| **Sentiment Analysis** | [`AI_sentiment_analysis/`](AI_sentiment_analysis/README.md) | Evaluate centrality, subjectivity, and polarity of Islam/Muslim representations with a six-model panel |
| **Authority Curation** | [`AI_authority_curation/`](AI_authority_curation/README.md) | Draft descriptions, propose person roles and find duplicates among authority records; agent-assisted, every write approved by the curator |
| **NotebookLM Export** | [`NotebookLM/`](NotebookLM/README.md) | Export articles as Markdown for Google NotebookLM. Read-only, no model call |

Each directory's README documents its scripts, flags, prompts and measured
limits. What each pipeline reads, writes, resumes and refuses is summarised in
one table in [docs/PIPELINE_CONTRACTS.md](docs/PIPELINE_CONTRACTS.md).

## Limitations and Caveats

These tools are research aids, not replacements for scholarly judgment. Users should be aware of several constraints:

**Algorithmic opacity.** LLMs operate as black boxes. We cannot fully trace their decision pathways, which challenges the transparency expected in historical scholarship. This project documents prompts, model versions, and processing parameters, but the models' internal reasoning remains opaque.

**Western-centric bias.** Models trained predominantly on Western data may misrepresent African contexts, linguistic nuances, and naming conventions. The NER pipeline includes fuzzy matching and human review stages to catch errors, but some will inevitably pass through.

**Hallucinations.** Unlike traditional OCR, which signals failure through garbled text, AI-generated errors appear as fluent prose. The cognitive burden shifts from fixing visible mistakes to detecting hidden ones. Original documents are preserved alongside AI outputs for verification.

**Non-determinism.** Running the same text through a model twice may yield different results, complicating reproducibility.

**The human in the loop.** Effective use of these tools requires domain expertise. Prior familiarity with the source material is necessary to audit AI outputs and distinguish genuine insight from plausible-sounding error.

## Installation

Requires **Python >= 3.11**. On 3.13 and later, `audioop-lts` (installed automatically) replaces the `audioop` module that the audio pipelines' `pydub` dependency needs and the standard library no longer ships.

On Windows, set `PYTHONUTF8=1` or invoke `python -X utf8` when redirecting output to a file. Audio conversion and splitting require a separately installed FFmpeg; see the audio pipeline README.

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
pytest tests/
ruff check .
```

To see a pipeline's handoff without credentials, a model call or an Omeka
instance, run the [offline example](examples/README.md). It builds synthetic
inputs, commits and validates an output artifact, and previews a write against a
fake archive:

```bash
python -X utf8 examples/offline_workflow.py --output-dir .cache/example
```

## Configuration

Create a `.env` file at the repository root. [`.env.example`](.env.example)
lists every variable with where to obtain it and which pipelines read it:

```env
# AI Providers (at least one required)
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
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

`01` reads from Omeka S, `02` calls the model, `03` writes back. Every script
answers `--help` and returns its outcome as an exit status, so a run can be
chained or scheduled.

### Repository layout

```text
AI_<name>/      One pipeline: numbered scripts, its prompts (.md) and a README
NotebookLM/     Read-only Markdown export
common/         Shared library: Omeka client, model registry and provider
                adapters, checkpoints, artifact sidecars, the write guard
serving/        Slurm + vLLM setup for an open-weights model on your own GPU
examples/       Offline example that needs no credentials
tests/          Mocked behavioural tests; no test calls a provider or Omeka
docs/           Pipeline contracts, publication guide, pending work,
                validated environment snapshots
scripts/        Release checks (metadata and links, environment capture)
```

Prompts are plain Markdown files beside the script that sends them, so the
instruction behind any output can be read without reading code. Model
identifiers, reasoning levels and sampling defaults live in one place,
[`common/llm_registry.py`](common/llm_registry.py).

Each pipeline directory ships its input folder empty (`PDF/`, `Audio/`,
`video/`, `TXT/`, `ALTO/` hold a `.gitkeep`), so source material can be
dropped in place or fetched by the `01` step; output, cache and backup folders
are created on first run. All of them are git-ignored.

Every step that writes to Omeka goes through the same gate: `--dry-run` reports
what would change, the pre-write payload of every item is dumped to a
`backups/` or `output/` folder first (the only route back from a bulk write),
and a live run asks for confirmation unless `--yes` is passed. Text and link
updates persist each backup before its PATCH. New authority creation uses a
durable operation journal and term-to-ID mapping. Model-generated discovery
text and keyword links carry annotations naming their producer —
`iwac:ocrModel`, `iwac:summaryModel`, `iwac:transcriptionModel`, and
`iwac:nerModel` on every subject or place link. OCR correction deliberately
preserves the original OCR annotation; it records correction provenance in its
local artifact manifest. Citation and detected-language writes have their own
contracts, described in the pipeline documentation.

OCR, summaries, publication text, corrected text and magazine indexes now have
hash-validated artifact sidecars. Incomplete OCR is salvaged in `partial/` and
cannot be uploaded through the normal path. Existing output from older releases
must be regenerated or explicitly imported with `--legacy-import` after review;
model-annotating legacy imports also require `--model`. See the
[pipeline contracts](docs/PIPELINE_CONTRACTS.md) and
[publication/reproducibility guide](docs/PUBLICATION.md).

Most scripts support both interactive mode and command-line flags:

```bash
python 01_NER_AI.py --item-set-id 123 --model gemini-3.8-flash
```

### Supported Models

| Provider | Key | Notes |
|----------|-----|-------|
| OpenAI | `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol` | Text pipelines only. GPT-6 Luna (default for summaries and citation extraction); GPT-6.1 Terra and Sol in OCR correction only, until they have authority items. The GPT-5.6 entries stay in the registry for the sentiment panel, which runs `gpt-5.6-luna`, and for older annotations; no pipeline offers them through `--model`. Legacy `gpt-5-mini` / `gpt-5.1` keys resolve to GPT-6 Luna / GPT-6 Sol. |
| Gemini | `gemini-3.8-flash`, `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon` | Text and multimodal. Gemini 4 Carbon (strongest) and Hydrogen (fast tier) replace Pro in OCR, HTR, audio, video, issue extraction and OCR correction; with 3.8 Flash they are the pinned keys that write steps can annotate (3.7 Flash is now answered by 3.8 and offered nowhere). Pro is offered nowhere: `gemini-3.1-pro` stays registered only for the annotations its runs left. Gemini 4 Argon is offered only in OCR correction. YouTube transcription also offers `gemini-3.5-flash-lite`; audio transcription also offers the rolling `gemini-flash-lite-latest` for runs that do not go to Omeka. |
| Gemini (speech) | `gemini-3.5-transcribe` | Dedicated speech-to-text via the Interactions API, not the model registry: word-level timestamps and speaker diarization, 82 locales including Hausa but none of Mooré, Dioula, Ewé, Kabyè or Dendi. Takes no prompt. Used by `AI_audio_summary/02c`. |
| Gemma | `gemma-4-openrouter` | Google Gemma 4 31B open weights, reached through OpenRouter. Offered by NER and reference indexing, and a sentiment panel member. Two reasoning levels, `minimal` and `high`. The registry's Gemini-API route (`gemma-4`) is in no pipeline menu: Google serves Gemma there only on a free tier whose content it uses. |
| Mistral | `mistral-small` (text), `mistral-large-4`, `ministral-14b` (OCR correction only) | Small 4 reasons at `none` or `high` only. Large 4, and the GLM 5.3 and DeepSeek V4.1 Flash models Mistral hosts (keys ending `-mistral`), are offered in OCR correction only until they have been compared on real material; Ministral 14B has no authority item, so only OCR correction, which stamps no model, offers it. Pinned `mistral-ocr-4-1` for OCR; Voxtral Small Transcribe 3 (default), Mini Transcribe 3 and `voxtral-mini-2602` for audio. |
| OpenRouter | `deepseek-v4-flash-0731` (default), Qwen and legacy/quality options | DeepSeek V4 Flash 0731 is the shared text default (`DEFAULT_TEXT_MODEL_KEY`), used by NER, OCR correction and magazine consolidation. Summarization and publication citation extraction default to `gpt-6-luna` for throughput. It is text-only: PDF/image/audio/video extraction still uses the modality-specific Gemini, Mistral, or Voxtral APIs. Requests go only to providers that, by OpenRouter's account, do not collect user data (`data_collection: "deny"`); `OPENROUTER_ZDR=1` narrows that to endpoints with a formal zero-data-retention policy. |
| Anthropic | `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2` | Haiku 5.5 is offered by NER and reference indexing and is a sentiment panel member. Sonnet 5.5 is the one Claude model in the document tier: a second reading for OCR and HTR, and the `sonnet` profile of magazine extraction. Opus 5.5 and Fable 5.2 are offered in OCR correction only. The system prompt is cached on every request. |
| Self-hosted | `qwen3.8-27b-selfhosted` | Any OpenAI-compatible endpoint you run yourself — vLLM on a GPU cluster, or llama.cpp / LM Studio / TGI locally. Text pipelines only. The address comes from `SELFHOSTED_LLM_BASE_URL`, so no model here is tied to one machine; a model on this route is simply reported as unavailable when the variable is unset. See [`serving/`](serving/README.md). |

`deepseek-v4-flash-0731` is pinned to the dated OpenRouter slug
`deepseek/deepseek-v4-flash-0731`; the generic aliases `deepseek` and
`deepseek-flash` resolve to it. Every DeepSeek Flash run goes to that release:
the earlier `deepseek-v4-flash` preview sits in no model tier, so no pipeline
offers it and no `--model` accepts it. Its registry entry is kept for the
archive alone, so an old pilot payload that names its slug still resolves; the
sentiment values it wrote were deleted from Omeka on 2026-08-07.

## Adapting for Other Projects

These tools were built for IWAC but can be modified for other collections:

- **Prompts** are stored as `.md` files in each pipeline directory and can be edited for different contexts, languages, or document types
- **Pipelines** are modular and can be used independently
- **Shared utilities** (`common/`) centralize Omeka S API access (`omeka_client.py`), the model catalog (`llm_registry.py`), provider adapters (`llm_provider.py`), durable checkpoints (`checkpoint.py`), the page-by-page Gemini PDF loop (`gemini_page_processor.py`), and idempotent text/resource-link writes (`omeka_text_updater.py`, `omeka_link_updater.py`) — so a pipeline is mostly its prompts and its choice of model
- **Serving** (`serving/`) is written to be site-agnostic: every cluster-specific value is an environment override, so pointing it at your own hardware — or at any OpenAI-compatible endpoint you already run — means editing one file of defaults, not the pipelines

The approach assumes you have digitized materials and need to make them searchable. It is designed for institutions and researchers managing substantial digital collections with limited resources.

## Documentation

Each pipeline's README is linked from the [table above](#what-these-tools-do).
Across pipelines:

- [Pipeline contracts](docs/PIPELINE_CONTRACTS.md) — per pipeline: input and output, default model, what resumes, what counts as incomplete, what is written to Omeka
- [Publication and reproducibility](docs/PUBLICATION.md) — release validation, artifact sidecars and legacy output, recovery after an interrupted write, the run record to keep for each run a paper reports
- [Shared utilities](common/README.md) — the Omeka client, the model registry and provider adapters, structured output, checkpoints, the write guard
- [Serving your own models](serving/README.md) — running an open-weights model on your own GPU (Slurm + vLLM), the SSH tunnel, and the probe that checks a route's reasoning levels are real
- [Sentiment panel record](AI_sentiment_analysis/PANEL_RECORD.md) — dated measurements of each panel member's corpus pass and the decisions they led to
- [Changelog](CHANGELOG.md) — releases, and the operational history behind the model registry, the model authority items and the sentiment panel
- [Pending work](docs/PENDING.md) — what is unfinished and what closes each item: models awaiting a live probe or an authority item, dated deadlines, measurements owed
- [IWAC on Hugging Face](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) — Public projection: full text is retained only where `OCR_is_public` permits it. The private `-full` mirror contains the complete text; these are different access scopes.

## Reproducibility

The repository changes continuously, and so do the models behind it: providers
retire and silently replace model versions, and the same prompt can return a
different answer twice. What can be reproduced is therefore the procedure, and
what is recorded is which model produced each stored value.

- **Code.** Cite the version DOI of the release a run used, not the concept
  DOI. Each release is archived on Zenodo; `docs/validation/` holds the package
  versions it was tested with.
- **Prompts and settings.** Prompts are the `.md` files in each pipeline
  directory. A run's model, reasoning level and prompt hash are stored in the
  checkpoint and the `.artifact.json` sidecar beside its output.
- **Provenance in the archive.** Model-written text and links carry an
  annotation naming the model (`iwac:ocrModel`, `iwac:summaryModel`,
  `iwac:transcriptionModel`, `iwac:nerModel`), so a reader of the archive can
  tell machine output from catalogued data and trace it to a model.
- **Tests.** The suite mocks every provider and Omeka. It demonstrates that the
  pipelines resume, refuse and write as documented; it says nothing about OCR
  fidelity, summary faithfulness or the validity of a sentiment reading, which
  need a human-reviewed sample.

## Related Resources

- [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) — The digital collection
- [Kalamu](https://github.com/fmadore/Kalamu) — Desktop app (Windows, macOS) that runs these workflows through a graphical interface
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

## Acknowledgements

Every model here is reached through a provider's API except Qwen3.8-27B, which
was served with this repository's [`serving/`](serving/README.md) setup for the
reasoning-depth probe of 16 August 2026 and the full-corpus sentiment pass of
17–24 August 2026.

> Calculations were performed using the festus-cluster of the Bayreuth Centre
> for High Performance Computing (https://www.bzhpc.uni-bayreuth.de), funded by
> the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) -
> 523317330.

## License

[MIT](LICENSE).

---

These workflows represent one approach to managing digital abundance in under-resourced archival contexts. They do not solve the fundamental challenges of algorithmic opacity or Western-centric bias in AI systems, but they offer a documented, auditable method for processing materials that would otherwise remain inaccessible.
