# Copilot Instructions for iwac-ai-pipelines

These notes tell GitHub Copilot how to help inside this repo. Apply them to every suggestion and code edit.

## Core expectation: shared LLM provider

- **Always route AI calls through `common/llm_provider.py`.** Import `build_llm_client`, `get_model_option`, `LLMConfig`, and `summary_from_option`; never instantiate `openai.OpenAI()` or `google.genai.Client()` directly in scripts.
- Let `get_model_option()` drive model selection (interactive prompt or `--model` flag). Use `allowed_keys` parameter to restrict choices per pipeline.
- Registry contains: `openai` (GPT-5.1 mini), `openai-5.1` (GPT-5.1 full), `gemini-flash`, `gemini-pro`.
- When logging or naming output files, reuse `summary_from_option(model_option)` and `model_option.key` so runs clearly identify the provider.

## CLI + configuration conventions

- Pipelines should expose a `--model` option with `choices=[...]` to limit available models for that specific use case.
- Use `LLMConfig` to specify reasoning effort, verbosity, thinking budget, and temperature appropriate for the task.
- Remove standalone `--temperature` flags; configure via `LLMConfig` instead.
- Per-script config loaders should validate Omeka/API env vars only. Let `common/llm_provider.py` raise errors about missing `OPENAI_API_KEY` / `GEMINI_API_KEY`.

## LLM Configuration best practices

- **Create task-specific configs**: Use `LLMConfig()` to define parameters suited to your pipeline's needs.
- **OpenAI parameters**: `reasoning_effort` ("low"/"medium"/"high"), `text_verbosity` ("low"/"medium"/"high").
- **Gemini parameters**: `temperature` (0.0-1.0), `thinking_budget` (0=off for Flash, None=default, >0=custom).
- **Cost-conscious defaults**: Use "medium" reasoning and moderate thinking budgets (e.g., 500) unless high quality is critical.
- **Log your config**: Always log the effective configuration for reproducibility.

## Prompt + output handling

- Keep prompt templates in the pipeline directory (e.g., `ner_system_prompt.md`, `summary_prompt.md`). Load them once and feed into `llm_client.generate(system_prompt, user_prompt)`.
- Before invoking the model, short-circuit empty or whitespace-only inputs.
- When writing per-model artifacts, embed `model_option.key.replace('-', '_')` in the filename (e.g., `_processed_openai.csv`).
- Per-request config overrides: Pass `config=LLMConfig(...)` to `generate()` for one-off adjustments.

## Adding or updating models

1. Modify `MODEL_REGISTRY` and `MODEL_ALIASES` inside `common/llm_provider.py`.
2. Set appropriate `default_reasoning_effort`, `default_text_verbosity`, `default_thinking_mode`, and `default_thinking_budget` for the model.
3. Update relevant README files and pipeline `--model` choices lists.
4. Avoid duplicating provider-specific code elsewhere—changes belong in the shared helper only.

## Anti-patterns (do not do this)

- Re-implementing OpenAI/Gemini client setup inside individual scripts.
- Hard-coding model names outside `common/llm_provider.py` (besides documentation or command examples).
- Bypassing `summary_from_option` when logging model choice.
- Calling `llm_client.generate` without guarding against empty content.

## Checklist for new AI scripts

- [ ] Imports `common.llm_provider` helpers (`build_llm_client`, `LLMConfig`, `get_model_option`, `summary_from_option`).
- [ ] Provides `--model` flag with appropriate `choices=[...]` for the pipeline, or uses `allowed_keys` in `get_model_option()`.
- [ ] Creates task-appropriate `LLMConfig` with reasoning/thinking parameters.
- [ ] Logs `summary_from_option(model_option)` and config parameters before processing.
- [ ] Loads prompts from sibling `.md` files.
- [ ] Skips empty text before requesting an LLM.

Following these rules keeps every pipeline aligned; when APIs change, a single edit to `common/llm_provider.py` fixes the whole repo.