# AI Sentiment Analysis Pipeline

Annotates how French- and English-language West African press articles treat
Islam and Muslims, with a panel of five language models working as independent
annotators. Each model's answers are stored in Omeka under properties named for
that model, so every value says which model produced it.

Dated measurements behind the choices below (throughput, cost, the Qwen3.8 and
Gemma probes, the prompt's history) are in [PANEL_RECORD.md](PANEL_RECORD.md).

## Scripts

| Script | Purpose | Writes to Omeka |
|---|---|---|
| `00_setup_properties.py` | Generate the ontology properties the panel needs; pre-flight the vocabulary upload | no |
| `01_sentiment_analysis.py` | Production run: annotate items and store the results | **yes** |
| `02_pilot_new_panel.py` | Trial a candidate model on a sample of already-annotated articles | no |
| `03_pilot_report.py` | Agreement and self-consistency report for a pilot | no |
| `04_import_offline_run.py` | Seed the cache from a run on your own GPU, so `01` writes those answers instead of re-asking | no |
| `sentiment_core.py` | The panel, schema, prompt loading, vocabulary and model calls | no |
| `sentiment_cache.py` | Resumable per-(item, model) result cache | no |

`sentiment_core` is shared so that a pilot and a production run are the same
instrument: same Pydantic schema, same `sentiment_prompt.md`, same `PANEL`, same
call path through `common/llm_provider.py`.

## What is annotated

Three dimensions, each with a one-sentence justification in French.

**Centralité** — how central Islam or Muslims are to the article: *Très
central*, *Central*, *Secondaire*, *Marginal*, *Non abordé*.

**Subjectivité** — how far the article commits itself on the subject,
independently of whether the treatment is favourable: *Très objectif*, *Plutôt
objectif*, *Mixte*, *Plutôt subjectif*, *Très subjectif*.

**Polarité** — the article's stance towards Islam or Muslims: *Très positif*,
*Positif*, *Neutre*, *Négatif*, *Très négatif*, *Non applicable*.

When centralité is *Non abordé*, subjectivité is null and polarité is *Non
applicable*. The schema's validator also rejects the converse: a null
subjectivité beside any other centralité.

Generation 2 asks for subjectivité as a **label**; generation 1 asked for an
integer 1–5. Subjectivité was the one dimension requested as a number and by a
distance the least reliable (pairwise κ 0.093–0.470 in the pilot panel, against
0.248–0.478 for polarité and up to 0.725 for centralité), and numeric scales are
a documented cause of that ([arXiv:2406.11980](https://arxiv.org/abs/2406.11980)).
`SUBJECTIVITE_ORDER` keeps the ranking for ordinal analysis. The schema field is
still called `subjectivite_score` because it is the live Hugging Face column
name and Omeka property suffix.

## The panel

Defined once in `sentiment_core.PANEL`; the property terms, the ontology, the
cache keys and the pilot are all derived from it.

| Member | Model, route | Omeka properties | Authority item | Hugging Face prefix | Coverage |
|---|---|---|---|---|---|
| Gemma 4 31B | `google/gemma-4-31b-it`, OpenRouter | `iwac:gemma431bIt*` | 111663 | `gemma_4_31b_it_` | 12,298 |
| GPT-5.6 Luna | `gpt-5.6-luna`, OpenAI | `iwac:gpt56Luna*` | 79610 | `gpt_5_6_luna_` | 12,298 |
| Mistral Small 4 | `mistral-small-2603`, Mistral | `iwac:mistralSmall2603*` | 79614 | `mistral_small_2603_` | 12,298 |
| DeepSeek V4 Flash 0731 | `deepseek/deepseek-v4-flash-0731`, OpenRouter | `iwac:deepseekV4Flash0731*` | 83261 | `deepseek_v4_flash_0731_` | 12,298 |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B`, self-hosted vLLM | `iwac:qwen3827b*` | 111933 | `qwen3_8_27b_` | 12,098 |

Coverage is the number of articles carrying the member's values on the live
archive, read on 6 October 2026. The four API members cover every eligible
article. Qwen3.8 covers 12,098 of the 12,251 articles in its input and stays
there by decision: the 153 it never annotated concentrate on articles where
Islam is marginal, so **its missing subjectivité is not missing at random**
([details](PANEL_RECORD.md#corpus-pass)).

### Why these five

- **Every member is its vendor's high-volume tier**, so the panel is a set of
  readings of the construct rather than a quality ladder. The Google slot held
  Gemini 3.6 Flash until 2026-07-31, at five to seventeen times the others'
  price, which let any disagreement be read as "the expensive model knows
  better".
- **No two members share a lab or pretraining family.** Gemma replaced Gemini
  rather than joining it, because two Google models would buy correlated
  annotator error and inflate agreement.
- **The route is part of the provenance.** Gemma is reached through OpenRouter
  because Google serves it on the Gemini API only on a free tier whose content
  Google uses. Qwen3.8 is the self-hosted run; the same weights on OpenRouter
  are a separate registry key and a pilot candidate only.
- **Four of the five are open weights** (Gemma, Mistral Small 4, DeepSeek,
  Qwen), so their annotations can in principle be re-run from archived weights.
  Gemma and DeepSeek reached the panel through OpenRouter at whatever
  quantization the chosen backend served, which qualifies that claim.

Their active parameter counts span about a factor of five (6.5B to 31B), so an
agreement figure partly measures model size; a write-up should say so.

### Reasoning depth

The panel asks every member for a middle reasoning setting, and only two have
one. The others are sent `high`, explicitly, in `PANEL_REASONING_OVERRIDES` or
by the Mistral client's documented rounding, so that a run records a decision
rather than a fallback.

| Member | Levels the API accepts | Sent |
|---|---|---|
| GPT-5.6 Luna | none / low / medium / high / xhigh / max | `medium` |
| Qwen3.8 27B | low / medium / xhigh | `medium` |
| Gemma 4 31B | minimal / high | `high` |
| DeepSeek V4 Flash 0731 | low / high / max | `high` |
| Mistral Small 4 | none / high | `high` |

So **two of five sit at a genuine middle setting and three are rounded up**,
which is a real limit on comparability. Through OpenRouter Gemma's reasoning is
on/off rather than graduated, so read its depth as requested, never as
measured. Temperature is not standardised: it is vendor-owned in
`MODEL_REGISTRY` (1.0 for DeepSeek and Qwen, 0.3 for Mistral Small 4, unset for
Gemma and Luna). Without a self-consistency figure from `02 --repeats`, a
low-agreement high-temperature model cannot be told apart from noise.

## Provenance

**Generation 2 (from 2026-07-31) is what Omeka holds.** Each member has six
properties named for the model, never its vendor, because Omeka does not index
value annotations: the property name is the only provenance a query can reach.
That is also why there are six model-keyed properties per member rather than six
multi-valued ones. *"polarité = Négatif according to DeepSeek"* has to be
answerable by query. Generation-2 values carry no value annotation; the
`iwac:sentimentModel` annotation written before 2026-07-31 was dropped as an
unreachable copy of what the property name says.

**Generation 1 (January–February 2026) was deleted from Omeka on 2026-08-07**,
after its values were confirmed on the Hugging Face full mirror, which is now its
only copy (frozen with `omeka_prefix=None` in the uploader's panel; unfreezing
it deletes the campaign). Nothing in this pipeline reads it. Its vendor-keyed
properties map to:

| Hugging Face prefix | Model | Run configuration |
|---|---|---|
| `gemini_3_flash_preview_` | `gemini-3-flash-preview` | temperature `0.2`, `response_schema` |
| `gpt_5_mini_` | `gpt-5-mini` | no temperature sent, `response_format` |
| `ministral_14b_2512_` | `ministral-14b-2512` | temperature `0.2`, **`max_tokens=512`** |

No generation-1 model ran with a reasoning or thinking parameter; Ministral alone
capped output at 512 tokens, so its justifications could be truncated; GPT-5 mini
ran at the API default temperature while the other two were pinned to 0.2. The
mapping was recovered from commit `07fb007`, because generation 1 recorded no
model at all. Generation 1 also ran the original prompt, so a cross-generation
difference confounds model change with prompt change, and a comparison reads the
Hub for one side and Omeka for the other.

**Properties that hold nothing.** The Gemini 3.5 Flash-Lite slot
(`iwac:gemini35FlashLite*`, 2026-07-31 to 08-14) and Qwen3.5 122B-A10B
(`iwac:qwen35A10b*`, dropped 2026-08-05) never annotated an article. The April
DeepSeek preview's `iwac:deepseekV4Flash*` values were deleted on 2026-08-07 and
never exported; `deepseek_v4_flash_0731_` on the Hub is a different run.

> **Rule:** a new model gets a new property set named for the model. Never reuse
> a vendor slot; `test_panel_does_not_reuse_an_abandoned_property` enforces it.

## Usage

### Production run

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --limit 50 --dry-run
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36
```

The first is a trial: 50 items, annotated and cached, nothing written. The
second is the whole article class, the usual target.

| Flag | Effect |
|---|---|
| `--resource-class-id [ID]` | A whole resource class (bare flag: 36, newspaper articles) |
| `--item-set-id` | One or more item sets, comma-separated |
| `--item-ids` | Exactly these items, comma-separated or `@file`; replaces the listing |
| `--models` | Run part of the panel, e.g. `deepseek_v4_flash_0731` |
| `--limit N` | Stop after N items needing work |
| `--concurrency N` | Items annotated in parallel (default 6) |
| `--model-timeout S` | Total seconds per model across its three attempts (default 120; use 300 for DeepSeek and Gemma) |
| `--dry-run` | Analyse and cache, PATCH nothing |
| `--skip-update` | Analyse and cache only; do not touch Omeka at all |
| `--force-reanalyze` | Ignore the cache **and** the already-annotated guard |
| `--rewrite` | Re-PATCH items that already carry values, from cached answers only; no model is called |
| `--from-cache` | Write cached answers only and build no client, for a member annotated offline |
| `--yes` | Skip the confirmation prompt |
| `--backup-dir` | Where each item's pre-write JSON is appended before its PATCH (default `backups/`); the only route back from a corpus pass |
| `--no-backup` | Do not dump pre-write payloads. Not recommended |
| `--verbose` | Log each model failure as it happens |

**One member at a time** is the normal mode, not a degraded one. Each member owns
its six properties, so running them one after another builds the same result as
running them together, with a smaller blast radius per run and a real read on
one model's cost and failure rate. A scoped run reads and writes only the
members named; a member already annotated in Omeka is skipped even with a cold
cache.

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 \
    --models deepseek_v4_flash_0731 --model-timeout 300
```

`--concurrency` multiplies with the per-item fan-out: one member keeps that many
requests in flight, the whole panel five times as many. Measured wall clocks
range from 2.7 h (Luna) to 31.5 h (DeepSeek) per corpus pass at the default
([table](PANEL_RECORD.md#throughput)).

**Running out of credit** (a 402, or a 429 naming a daily or billing cap) stops
the run on the first occurrence, prints what the provider said, and exits 2.
The cache flushes per record, so topping up and re-running the same command
resumes.

### Language gate

Only articles whose `dcterms:language` is *Français* or *Anglais* are annotated:
12,298 of 12,349 articles on 2026-08-15. The Ewé, Kabiyè and Dendi articles and
those with no language value are counted separately, because a French-prompted
model returns a confident, unusable score on them that is indistinguishable from
a real annotation once stored. `dcterms:language` is a link to an authority item,
so the label is read from its `display_title`. The ceiling moves with the
corpus: a gap in every member at once is the corpus moving, a gap in one member
is a failed run.

### Resuming

A corpus pass takes hours to days, so **the safe response to any failure is to
run the same command again.**

1. **Omeka is checked first.** An item already carrying values for every
   selected member is skipped without another call. This works across machines,
   because the state lives in the archive.
2. **Results are cached per (item, model)** in `cache/sentiment_v2.jsonl`, an
   append-only file flushed after every record, so each model is asked only for
   what it has not answered.
3. **Only successes are cached.** An errored call is not written, so the next
   run retries it.

Each cache record carries the model id, the reasoning depth sent and the prompt
fingerprint, and a record whose provenance no longer matches the live panel is
not reused. `--force-reanalyze` appends rather than rewrites, so earlier answers
stay in the file as an audit trail while the newest wins on load.

### Running a member on your own GPU

`serving/` runs an open-weights model on your own hardware (Slurm and vLLM, or
any OpenAI-compatible endpoint), and the registry reaches it like a hosted one.
It is how Qwen3.8 was annotated, for two reasons: a corpus pass costs queue time
rather than tokens, and a request goes to one server whose reasoning depth you
can measure with `serving/probe_reasoning.py` instead of a router's backends.

The cluster never holds credentials. Sample the corpus on the machine that has
them, ship a JSON file of article ids, text, prompt and prompt fingerprint, and
let `serving/annotate_job.sbatch` annotate it against `localhost`. Results append
per article and resume after a walltime kill, and records with another prompt
fingerprint are ignored. Each article is annotated independently: system prompt
plus one article, no history and no batching. Concurrency therefore cannot
change any article's answer beyond GPU batch-level numerical noise, which
temperature 1.0 dwarfs.

Getting the answers into Omeka is a **write pass, not a re-run**: a second pass
at temperature 1.0 would publish different labels from the ones measured.

```bash
python serving/merge_shards.py --shards 'AI_sentiment_analysis/cache/qwen38_full/full-s*.jsonl' \
    --output AI_sentiment_analysis/cache/qwen38_full/qwen38_merged.jsonl
python AI_sentiment_analysis/04_import_offline_run.py \
    --input cache/qwen38_full/qwen38_merged.jsonl --model qwen3_8_27b
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 \
    --models qwen3_8_27b --from-cache --dry-run
```

The importer refuses any record whose model id, reasoning level or prompt
fingerprint disagrees with the live panel. Setup, tunnels and partition choice
are in [`serving/README.md`](../serving/README.md).

## Piloting a candidate

```bash
python AI_sentiment_analysis/02_pilot_new_panel.py --sample-size 200 --seed 42
python AI_sentiment_analysis/02_pilot_new_panel.py --models qwen3_8_27b_openrouter --sample-size 50
python AI_sentiment_analysis/02_pilot_new_panel.py --repeats 3 --sample-size 50
python AI_sentiment_analysis/03_pilot_report.py
```

`02` samples already-annotated articles, runs the live panel plus everything in
`PILOT_CANDIDATES` on them, and writes `cache/pilot/pilot_<timestamp>.json` with
a manifest of model ids, seed and repeat count. Nothing reaches Omeka, and a
candidate could not write there anyway, since `01` iterates `PANEL` alone. A
model whose credentials are missing is skipped and named. An interrupted pilot
resumes only when it is re-run with the same `--output` path, because the
default name carries a timestamp.

The only candidate now is Qwen3.8 27B's OpenRouter twin, which measures the
route against the self-hosted member; it must never be promoted beside it.

`03` reports, per dimension, each model's agreement with the majority of the
*other* members (leave-one-out), pairwise Cohen's κ, and with `--repeats` > 1
how often each model reproduces its own answer. The 2026-07-29 pilot measured
DeepSeek's polarité self-consistency at 0.52 against 0.70–0.80 for the rest.

## Adding a model to the panel

A candidate is piloted first, from `PILOT_CANDIDATES`, which `01` cannot see.
Promotion is then:

1. Add the model to `MODEL_REGISTRY` in `common/llm_registry.py`.
2. Create its authority item in Omeka (class 244, template 3, item set 267,
   `dcterms:type` → "Notice d'autorité") and add it to `AI_MODEL_ITEMS` in
   `common/iwac_config.py`.
3. Move its `PanelMember` from `PILOT_CANDIDATES` into `PANEL`.
4. Regenerate the ontology, add it to `iwac-vocabulary.ttl`, and pre-flight the
   upload:

```bash
python AI_sentiment_analysis/00_setup_properties.py --emit-ttl
python AI_sentiment_analysis/00_setup_properties.py --verify
```

Upload through **Admin → Vocabularies → IWAC Ontology → Update**. That update
applies additions only: it has never removed an installed property the file
omits (vocabulary 10 went 74 → 80 → 86 without losing any of the 48 empty
declarations the file leaves out). `--verify` still counts the values under
every installed property the file omits, which is what would catch a member
dropped from `PANEL` while it held values; if it reports any, stop. Never add
properties with `PATCH /api/vocabularies/10`: that route removes every property
absent from the request. `POST /api/properties` fails on Omeka S 4.2.x
("A vocabulary must be set"), so the UI is the only route.

Property IDs are resolved from Omeka at startup, never hardcoded, because Omeka
assigns them when the vocabulary is updated. Never count annotators from the
installed property list: it includes the empty declarations above.

## The prompt

`sentiment_prompt.md` is the instrument. Every cache record and pilot manifest
carries its **fingerprint** (`sentiment_core.prompt_fingerprint()`, a short
sha256 of the text sent), because prompt wording moves label distributions in
ways a diff does not predict. Every member of the current panel ran
`#d14ace9ac192`. The 2026-07-31 rewrite and later corrections are listed in
[PANEL_RECORD.md](PANEL_RECORD.md#the-prompt).

The prompt states that *Non abordé* implies a null subjectivité, but not the
converse; that rule lives only in the schema validator. It is the rule Qwen3.8's
153 unannotated articles fail, and a write-up of that member should say so.

## Omeka properties and vocabulary

Six properties per member:

| Suffix | Type | Holds |
|---|---|---|
| `Centralite` | resource:item | link into the centralité vocabulary |
| `CentraliteJustification` | literal (`@language: fr`) | one sentence |
| `Polarite` | resource:item | link into the polarité vocabulary |
| `PolariteJustification` | literal (`@language: fr`) | one or two sentences |
| `SubjectiviteScore` | resource:item | link into the subjectivité vocabulary |
| `SubjectiviteJustification` | literal (`@language: fr`) | one or two sentences |

The three scales are links to controlled-vocabulary items, not literals; resolve
them through these IDs (`CENTRALITE_ITEM_IDS`, `POLARITE_ITEM_IDS`,
`SUBJECTIVITE_ITEM_IDS` in `sentiment_core.py`):

| Centralité | Item | Polarité | Item | Subjectivité | Item |
|---|---|---|---|---|---|
| Très central | 78048 | Très positif | 78031 | Très objectif | 78043 |
| Central | 78049 | Positif | 78038 | Plutôt objectif | 78044 |
| Secondaire | 78050 | Neutre | 78039 | Mixte | 78045 |
| Marginal | 78051 | Négatif | 78040 | Plutôt subjectif | 78046 |
| Non abordé | 78052 | Très négatif | 78041 | Très subjectif | 78047 |
| | | Non applicable | 78042 | | |

## Environment variables

```bash
# Omeka S API (required)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# One per hosted member you run
OPENAI_API_KEY=...        # GPT-5.6 Luna
MISTRAL_API_KEY=...       # Mistral Small 4
OPENROUTER_API_KEY=...    # Gemma 4 31B and DeepSeek V4 Flash 0731

# Only to annotate with Qwen3.8 on your own server (see serving/README.md).
# Writing its imported answers with --from-cache needs neither.
SELFHOSTED_LLM_BASE_URL=http://localhost:8000/v1
SELFHOSTED_LLM_API_KEY=sk-...
```

A member whose credentials are missing is skipped and named, never silently
dropped from the panel.

## Publication workflow

See the [pipeline contract](../docs/PIPELINE_CONTRACTS.md) for inputs, outputs,
resume identity and Omeka write semantics, and the
[publication guide](../docs/PUBLICATION.md) for release validation.
