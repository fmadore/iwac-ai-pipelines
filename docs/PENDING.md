# Pending work

What is known to be unfinished, and what closes each item. Remove an entry when
it is done and record what happened in [CHANGELOG.md](../CHANGELOG.md); add one
whenever a change stops short of where it should end.

Last reviewed 2026-09-27.

## GPT-6 Sol: verify, register, then promote

In `MODEL_REGISTRY` but offered only in `TEXT_FULL_MODELS` (OCR correction,
which stamps nothing). In order:

1. **Call it live.** Confirm the Responses API accepts `text.verbosity`, the
   `reasoning.effort` values the pipelines send (`low`, `medium`) and
   `responses.parse` structured output, and that the registry's $2/$0.20/$10
   rate matches the invoice. Needs `OPENAI_API_KEY`.
2. **Create its Omeka authority item** (class 244, template 3, item set 267,
   `dcterms:type` → "Notice d'autorité", listed under 78030) and add it to
   `AI_MODEL_ITEMS`. Needs Omeka admin access.
3. **Decide which tiers it joins.** `tests/test_summary_annotation.py` fails if
   a stamping tier gets a model with no authority item.

## Annotations that name Gemini 3.7 Flash but were written by 3.8

Google answers `gemini-3.7-flash` with Gemini 3.8 Flash; found 2026-10-23,
start date unknown. No local output stamped with 3.7 was made after 3.8 shipped
(2026-09-02) except an OCR run that was never uploaded, but outputs made on
another machine or since deleted cannot be ruled out, and Omeka does not index
value annotations, so the affected items cannot be queried. Closes when Google's
deprecation date is known and every `iwac:*Model` → 111774 written after it has
been listed (from the HF mirror's provenance columns or a full scan) and
repointed to 195175 with the curator's approval.

## Claude Haiku 5.5 in the sentiment panel

In `PANEL` since 2026-10-22, promoted without a pilot. Its properties are in
IWAC vocabulary v1.6 (uploaded) and it annotated 16,617 of 16,618 eligible
articles on 2026-10-23. Left:

1. **Cost.** Read the 2026-10-23 spend from the Anthropic console and put the
   full-pass figure, thinking tokens included, in `PANEL_RECORD.md`'s cost
   table.
2. **Measure what the pilot would have.** `02_pilot_new_panel.py --repeats 3
   --sample-size 50` and `03_pilot_report.py` for its agreement and
   self-consistency, before a write-up leans on it.
3. **Bring the other five up to date.** The article class grew from 12,349 to
   16,715 items since their August passes, so about 4,300 articles carry
   Haiku's reading alone. Run each member with `--models` over class 36; the
   already-annotated guard skips what they have.

## Authority curation: before and after the first live pass

1. **Upload IWAC vocabulary v1.5.** It adds `iwac:descriptionModel`;
   `00_setup_properties.py --verify` passed on 2026-10-17 (one addition,
   nothing omitted holds values). Admin → Vocabularies → IWAC Ontology →
   Update. Until then `AI_authority_curation/03_apply.py` refuses annotated
   descriptions.
2. **Review the organisations pilot.** Twelve descriptions were drafted on
   2026-10-17 (`AI_authority_curation/output/organisations_20261017T182725`)
   and 35 pairs judged (`…181850`, one merge ready: AMOCI into AMSCI). Approve
   or edit them, then run more organisations before persons.
3. **Sort "Notices d'autorités à traiter".** 3,005 stubs in set 12826, most
   linked first, in runs of about 60 (`--kind unsorted --limit 60`, repeated).
   Decide what happens to the stubs no document links to
   (`unlinked_stubs.csv`): file them or delete them. The 7 already in set 266
   only need to leave the set: they are `ready` rows of the first run's
   `plan_sort.csv`. Two stubs can duplicate each other; the second is caught
   as a duplicate once the first has been filed.
4. **Wikidata for the rest.** After the organisations pilot, run
   `--task wikidata` over organisations (203 without a Wikidata item),
   subjects (156), events (131), places (189), then persons, where name-only
   matches are riskiest.
5. **Scholarly authors.** The 546 people named in references' author and
   editor lists need descriptions from professional sources, and ORCID and
   IdRef identifiers. That means the `wikicite` resolvers and an identifier
   write step, neither of which exists here yet. See the skill's last section.

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
- **Upload ledger for the other write steps.** `AI_summary/03` no longer
  overwrites a summary a curator corrected in Omeka (`UploadLedger` in
  `common/omeka_text_updater.py`). The OCR, OCR-correction, publication, audio
  and YouTube `03` steps write `bibo:content` without it, so rerunning one over
  an old output folder still replaces hand-corrected full text. Adopt the same
  `check` / `on_result` pair in each; the decision is whether full text that
  predates the ledger should be held back by default, as summaries are.
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
