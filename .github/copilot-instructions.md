# Copilot Instructions for iwac-ai-pipelines

These notes tell GitHub Copilot how to help inside this repo. Apply them to every suggestion and code edit.

## Core expectation: shared LLM provider

- **Always route AI calls through `common/llm_provider.py`.** Import `build_llm_client`, `get_model_option`, and `summary_from_option`; never instantiate `openai.OpenAI()` or `google.genai.Client()` directly in scripts.
- Let `get_model_option()` drive model selection (interactive prompt or `--model` flag). Supported keys today: `openai` (GPT-5.1 mini), `openai-5.1`, `gemini-flash`, `gemini-pro`.
- When logging or naming output files, reuse `summary_from_option(model_option)` and `model_option.key` so runs clearly identify the provider.

## CLI + configuration conventions

- Pipelines should expose a `--model` option that simply forwards user input to `get_model_option`. If omitted, rely on the interactive selection.
- Accept a `--temperature` flag only if the script benefits from it; remind users that OpenAI paths ignore this parameter while Gemini honors it.
- Per-script config loaders should validate Omeka/API env vars only. Let `common/llm_provider.py` raise errors about missing `OPENAI_API_KEY` / `GEMINI_API_KEY`.

## Prompt + output handling

- Keep prompt templates in the pipeline directory (e.g., `ner_system_prompt.md`, `summary_prompt.md`). Load them once and feed into `llm_client.generate(system_prompt, user_prompt)`.
- Before invoking the model, short-circuit empty or whitespace-only inputs.
- When writing per-model artifacts, embed `model_option.key.replace('-', '_')` in the filename (e.g., `_processed_openai.csv`).

## Adding or updating models

1. Modify `MODEL_REGISTRY` and `MODEL_ALIASES` inside `common/llm_provider.py`.
2. Mention the new key in relevant README files so humans know which `--model` flag unlocks it.
3. Avoid duplicating provider-specific code elsewhere—changes belong in the shared helper only.

## Anti-patterns (do not do this)

- Re-implementing OpenAI/Gemini client setup inside individual scripts.
- Hard-coding model names outside `common/llm_provider.py` (besides documentation or command examples).
- Bypassing `summary_from_option` when logging model choice.
- Calling `llm_client.generate` without guarding against empty content.

## Checklist for new AI scripts

- [ ] Imports `common.llm_provider` helpers and uses `build_llm_client`.
- [ ] Provides `--model` flag or clearly documents interactive selection.
- [ ] Logs `summary_from_option(model_option)` before processing.
- [ ] Loads prompts from sibling `.md` files.
- [ ] Skips empty text before requesting an LLM.

Following these rules keeps every pipeline aligned; when APIs change, a single edit to `common/llm_provider.py` fixes the whole repo.