# Copilot Instructions for iwac-ai-pipelines

These notes tell GitHub Copilot how to help inside this repo. Apply them to every suggestion and code edit.

## Core expectation: shared LLM provider

- **For text-only pipelines, route AI calls through `common/llm_provider.py`.** Import `build_llm_client`, `get_model_option`, `LLMConfig`, and `summary_from_option`; never instantiate `openai.OpenAI()`, `google.genai.Client()`, or `mistralai.Mistral()` directly in text-processing scripts.
- Let `get_model_option()` drive model selection (interactive prompt or `--model` flag). Use `allowed_keys` parameter to restrict choices per pipeline.
- Registry keys: `gpt-5-mini`, `gpt-5.1`, `gemini-flash`, `gemini-pro`, `mistral-large`, `ministral-14b`. Common aliases: `openai` → `gpt-5-mini`, `gemini` → `gemini-flash`, `mistral` → `mistral-large`, `ministral` → `ministral-14b`.
- When logging or naming output files, reuse `summary_from_option(model_option)` and `model_option.key` so runs clearly identify the provider.

## Exception: Multimodal pipelines (audio, vision, HTR, OCR)

- **Audio transcription (`AI_audio_summary/`)** and **HTR (`AI_htr_extraction/`)** use `google.genai.Client()` directly because they require multimodal features (audio bytes, PDF uploads) not supported by the shared text-only wrapper.
- **Gemini OCR (`AI_ocr_extraction/02_gemini_ocr_processor.py`)** uses `google.genai.Client()` for native PDF processing with custom system prompts.
- **Mistral OCR (`AI_ocr_extraction/02_mistral_ocr_processor.py`)** uses `mistralai.Mistral()` directly because it requires the dedicated `client.ocr.process()` endpoint (not the chat API). This is a specialized Document AI service with its own model (`mistral-ocr-latest`).
- These scripts handle their own model selection and API configuration.
- When adding new multimodal pipelines, it's acceptable to use provider clients directly for file uploads, vision, audio, or dedicated OCR endpoints.
- Still load prompts from `.md` files where applicable and follow other conventions.

## CLI + configuration conventions

- Pipelines should expose a `--model` option with `choices=[...]` to limit available models for that specific use case.
- Use `LLMConfig` to specify reasoning effort, verbosity, thinking budget, and temperature appropriate for the task.
- Remove standalone `--temperature` flags; configure via `LLMConfig` instead.
- Per-script config loaders should validate Omeka/API env vars only. Let `common/llm_provider.py` raise errors about missing `OPENAI_API_KEY` / `GEMINI_API_KEY` / `MISTRAL_API_KEY`.

## LLM Configuration best practices

- **Create task-specific configs**: Use `LLMConfig()` to define parameters suited to your pipeline's needs.
- **OpenAI parameters**: `reasoning_effort` ("low"/"medium"/"high"), `text_verbosity` ("low"/"medium"/"high").
- **Gemini parameters**: `temperature` (0.0-1.0), `thinking_budget` (0=off for Flash, None=default, >0=custom).
- **Mistral parameters**: `temperature` (0.0-1.0) - straightforward configuration.
- **Cost-conscious defaults**: Use "medium" reasoning and moderate thinking budgets (e.g., 500) unless high quality is critical.
- **Log your config**: Always log the effective configuration for reproducibility.

## Prompt + output handling

- Keep prompt templates in the pipeline directory (e.g., `ner_system_prompt.md`, `summary_prompt.md`). Load them once and feed into `llm_client.generate(system_prompt, user_prompt)`.
- Before invoking the model, short-circuit empty or whitespace-only inputs.
- When writing per-model artifacts, embed `model_option.key.replace('-', '_')` in the filename (e.g., `_processed_openai.csv`).
- Per-request config overrides: Pass `config=LLMConfig(...)` to `generate()` for one-off adjustments.

## Structured outputs (JSON schema enforcement)

- **Use `generate_structured()` for data extraction tasks** (NER, classification, form filling) where you need a specific JSON structure.
- Define output schemas as Pydantic `BaseModel` classes with `Field(description=...)` annotations.
- OpenAI, Gemini, and Mistral APIs all support native structured outputs that guarantee valid JSON matching your schema.
- Use `generate()` for free-form text outputs (summaries, translations, creative writing).
- Example:
  ```python
  from pydantic import BaseModel, Field
  from typing import List
  
  class NERResult(BaseModel):
      persons: List[str] = Field(description="Person names")
      locations: List[str] = Field(description="Place names")
  
  result = llm_client.generate_structured(system_prompt, user_prompt, NERResult)
  print(result.persons)  # Typed access, no JSON parsing needed
  ```

## Adding or updating models

1. Modify `MODEL_REGISTRY` and `MODEL_ALIASES` inside `common/llm_provider.py`.
2. Set appropriate `default_reasoning_effort`, `default_text_verbosity`, `default_thinking_mode`, and `default_thinking_budget` for the model.
3. Update relevant README files and pipeline `--model` choices lists.
4. Avoid duplicating provider-specific code elsewhere—changes belong in the shared helper only.

## Anti-patterns (do not do this)

- Re-implementing OpenAI/Gemini/Mistral client setup inside text-only scripts (use `common/llm_provider.py`).
- Hard-coding model names outside `common/llm_provider.py` (besides documentation, command examples, or multimodal scripts).
- Bypassing `summary_from_option` when logging model choice (for text pipelines).
- Calling `llm_client.generate` without guarding against empty content.

## Checklist for new text-only AI scripts

- [ ] Imports `common.llm_provider` helpers (`build_llm_client`, `LLMConfig`, `get_model_option`, `summary_from_option`).
- [ ] Provides `--model` flag with appropriate `choices=[...]` for the pipeline, or uses `allowed_keys` in `get_model_option()`.
- [ ] Creates task-appropriate `LLMConfig` with reasoning/thinking parameters.
- [ ] Logs `summary_from_option(model_option)` and config parameters before processing.
- [ ] Loads prompts from sibling `.md` files.
- [ ] Skips empty text before requesting an LLM.

## Checklist for new multimodal scripts (audio/vision/HTR/OCR)

- [ ] Uses appropriate provider client directly:
  - Gemini: `google.genai.Client()` with `GEMINI_API_KEY`
  - Mistral OCR: `mistralai.Mistral()` with `MISTRAL_API_KEY` and `client.ocr.process()`
- [ ] Offers model selection where applicable (e.g., Gemini Flash vs Pro).
- [ ] Loads prompts from sibling `.md` files (if the API supports system prompts).
- [ ] Handles API errors with appropriate retry logic.
- [ ] Logs the selected model before processing.

Following these rules keeps every pipeline aligned; when APIs change, a single edit to `common/llm_provider.py` fixes text pipelines across the repo.