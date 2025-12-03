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

## Gemini API best practices (multimodal scripts)

When using `google.genai.Client()` directly in multimodal scripts:

- **Use `system_instruction` in `GenerateContentConfig`**: Pass system prompts via the `system_instruction` parameter, not concatenated with user content.
  ```python
  from google.genai import types
  
  config = types.GenerateContentConfig(
      system_instruction=system_prompt,  # Modern pattern
      temperature=0.2,
      max_output_tokens=8192,
  )
  response = client.models.generate_content(
      model=model_name,
      contents=[pdf_part, user_prompt],  # Only user content here
      config=config
  )
  ```
- **Thinking configuration by model**:
  - Gemini 3 Pro: Use `thinking_level` ("low" or "high") - cannot be disabled
  - Gemini 2.5 Flash/Pro: Use `thinking_budget` (0=disabled, >0=token budget)
  ```python
  # Gemini 3 Pro
  config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
  
  # Gemini 2.5 Flash (disable thinking)
  config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
  ```
- **PDF processing**: Use `types.Part.from_bytes()` for inline PDF processing.
- **Structured outputs**: Pass Pydantic `BaseModel` classes directly to `response_schema` parameter.

## Prompt + output handling

- Keep prompt templates in the pipeline directory (e.g., `ner_system_prompt.md`, `summary_prompt.md`). Load them once and feed into `llm_client.generate(system_prompt, user_prompt)`.
- Before invoking the model, short-circuit empty or whitespace-only inputs.
- When writing per-model artifacts, embed `model_option.key.replace('-', '_')` in the filename (e.g., `_processed_openai.csv`).
- Per-request config overrides: Pass `config=LLMConfig(...)` to `generate()` for one-off adjustments.

## Console output with rich

Use the `rich` library for professional console output in all pipelines:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

console = Console()
```

- **Welcome banners**: Use `Panel()` with title and border styling for script introductions.
- **Configuration display**: Use `Table()` to show model selection, file lists, and settings.
- **Progress tracking**: Use `Progress()` with appropriate columns for long-running operations.
- **Status indicators**: Use colored text for success (`[green]✓[/]`), errors (`[red]✗[/]`), and info (`[cyan]...[/]`).
- **Section dividers**: Use `console.rule()` for clean visual separation.
- **Spinners**: Use `console.status()` for indeterminate operations.

Example pattern:
```python
# Welcome panel
console.print(Panel("Pipeline description", title="🚀 Pipeline Started", border_style="cyan"))

# Configuration table
model_table = Table(title="🤖 Model Configuration", box=box.ROUNDED)
model_table.add_column("Setting", style="dim")
model_table.add_column("Value", style="green")
model_table.add_row("Model", model_name)
console.print(model_table)

# Progress bar
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
              BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
    task = progress.add_task("Processing...", total=len(items))
    for item in items:
        # process item
        progress.update(task, advance=1)

# Summary
console.print(f"[green]✓[/] Completed: {success_count} items")
console.print(f"[red]✗[/] Failed: {error_count} items")
```

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
- [ ] Uses `rich` library for console output (panels, tables, progress bars).

## Checklist for new multimodal scripts (audio/vision/HTR/OCR)

- [ ] Uses appropriate provider client directly:
  - Gemini: `google.genai.Client()` with `GEMINI_API_KEY`
  - Mistral OCR: `mistralai.Mistral()` with `MISTRAL_API_KEY` and `client.ocr.process()`
- [ ] **Gemini API**: Uses `system_instruction` in `GenerateContentConfig` (not concatenated prompts).
- [ ] **Gemini API**: Uses correct thinking config (`thinking_level` for Gemini 3, `thinking_budget` for 2.5).
- [ ] Offers model selection where applicable (e.g., Gemini Flash vs Pro).
- [ ] Loads prompts from sibling `.md` files (if the API supports system prompts).
- [ ] Handles API errors with appropriate retry logic.
- [ ] Logs the selected model before processing.
- [ ] Uses `rich` library for console output (panels, tables, progress bars).

Following these rules keeps every pipeline aligned; when APIs change, a single edit to `common/llm_provider.py` fixes text pipelines across the repo.