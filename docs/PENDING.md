# Pending work

What is known to be unfinished, and what closes each item. Remove an entry when
it is done and record what happened in [CHANGELOG.md](../CHANGELOG.md); add one
whenever a change stops short of where it should end.

Last reviewed 2026-09-27.

## New models: verify, register, then promote

Gemini 3.8 Flash and GPT-6 Sol are in `MODEL_REGISTRY` but offered only where
no model is stamped into Omeka (`TEXT_FULL_MODELS`, used by OCR correction, and
the video summary). The steps below are in order: each depends on the one before.

1. **Probe Gemini 3.8 Flash live.** Its `supported_thinking_levels`
   (`low`/`medium`/`high`) come from Google's documentation, not from the API,
   which the registry rule requires. Needs `GEMINI_API_KEY`: one
   `generate_content` call per level, including `minimal` (expect a 400), plus
   one OCR page with `media_resolution` and one `response_schema` call. Then
   change the registry comment to "verified live" with the date. Check at the
   same time which release `gemini-flash-latest` now serves.
2. **Call GPT-6 Sol live.** Confirm the Responses API accepts
   `text.verbosity`, the `reasoning.effort` values the pipelines send
   (`low`, `medium`) and `responses.parse` structured output, and that the
   registry's $2/$0.20/$10 rate matches the invoice. Needs `OPENAI_API_KEY`.
3. **Create the Omeka authority items.** One each for Gemini 3.8 Flash and
   GPT-6 Sol: class 244, template 3, item set 267, `dcterms:type` →
   "Notice d'autorité". Add both to `AI_MODEL_ITEMS` in `common/iwac_config.py`.
   Needs Omeka admin access.
4. **Benchmark before promoting.** Run Gemini 3.8 Flash beside 3.7 on the same
   OCR, HTR and summary samples. Google says 3.8 spends more tokens by design on
   complex tasks, so compare cost per item as well as quality.
5. **Promote.** Only after 3 and 4: add `gemini-3.8-flash` to
   `GEMINI_DOCUMENT_MODELS`, `TEXT_ECONOMY_MODELS` and `TEXT_EXTENDED_MODELS`,
   point the `gemini`/`flash` aliases at it, and decide whether
   `AI_audio_summary/02` and `AI_youtube_transcription/02` offer it (both of their
   write steps stamp a model). Decide which tiers GPT-6 Sol joins.
   `tests/test_summary_annotation.py` fails if a stamping tier gets a model
   with no authority item.

## Dates

- **2026-10-23** — GPT-5 and GPT-5.1 snapshots shut down. Their keys already
  resolve to `gpt-6-luna` / `gpt-6-sol`. After that date, check whether any
  script or saved command still names them, then drop `LEGACY_CLI_MODEL_KEYS`.
- **2027-01-01** — Gemini 3.8 Flash goes from $0.75/$3.75 to $1.50/$7.50 per
  1M tokens. Re-estimate the cost of any bulk run that would cross that date,
  and update the registry description.

## Measurements owed

- **GPT-6 Luna throughput.** `AI_summary` pins it, but the published
  comparison (2.7 h versus DeepSeek 0731 at 31.5 h) was measured on GPT-5.6
  Luna. Re-run it and update `AI_summary/README.md` and CLAUDE.md.
- **Qwen3.8 polarité cross-check.** Only 1.9% of its 12,098 answers are
  negative. Compare them with the other four members on the same articles before
  a write-up leans on Qwen for that dimension
  (`AI_sentiment_analysis/PANEL_RECORD.md`).
- **Gemma 4 31B corpus cost.** The ~$8–12 figure is projected from three calls.
  Measure the 2026-08-14/15 pass against the OpenRouter credits endpoint and
  replace the row in `PANEL_RECORD.md`.
- **DeepSeek V4 Pro reasoning levels.** The registry accepts five
  (`minimal` … `xhigh`); `common/README.md` says it accepts only `high` and
  `xhigh`. Probe the OpenRouter route with `serving/probe_reasoning.py` and
  correct whichever is wrong.

## Code changes that need a decision

- **Credentials sent to any URL.** `OmekaClient.get_resource(url)` adds the
  Omeka key pair to whatever URL it is given. It should refuse a host other
  than `OMEKA_BASE_URL`'s. CLAUDE.md asks for approval before touching
  `omeka_client.py`.
- **Gemini structured output.** google-genai now documents
  `response_json_schema` (full JSON Schema) beside `response_schema`, which is
  converted to Google's OpenAPI subset. Switching is only worth it once Gemma 4
  on the Gemini route has been checked against it.
- **Page processor retries.** A transient error on the inline request
  (429/500/503) is not retried inline: the page goes straight to a Files API
  upload, which does retry. Retrying inline first would save an upload per
  overloaded page.
- **Mistral Large 3 and Ministral 14B authority items.** Both left the
  summary, NER and reference-indexing menus on 2026-10-06 because nothing in
  Omeka names them. Create the items and add them to `AI_MODEL_ITEMS` if either
  should return; the guard test then allows it.
- **Voxtral provenance.** `voxtral-mini-2602` has no authority item, so its
  transcripts can only be uploaded with `--no-model-annotation`. Create one if
  Voxtral output is to carry `iwac:transcriptionModel`.
- **NotebookLM exporter argv.** `NotebookLM/omeka_items_to_md.py` parses argv
  by hand: `--help` is reported as an unrecognised argument and the script drops
  into its interactive menu. It only reads Omeka, so this is a usability gap
  rather than a write-safety one; `argparse` would fix it.
