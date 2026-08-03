# AI Sentiment Analysis Pipeline

Analyse sentiment towards Islam and Muslims in French-language West African
media articles, using several AI models concurrently as an annotator panel.

## Scripts

| Script | Purpose | Writes to Omeka |
|---|---|---|
| `00_setup_properties.py` | Generate the ontology properties a panel needs; pre-flight the vocabulary upload | no |
| `01_sentiment_analysis.py` | Production run: annotate items and store results | **yes** |
| `02_pilot_new_panel.py` | Trial a candidate panel on a sample of already-annotated articles | no |
| `03_pilot_report.py` | Agreement report for a pilot run | no |
| `sentiment_core.py` | Shared panel, schema, prompt, vocabulary and analysis calls | no |
| `sentiment_cache.py` | Resumable per-(item, model) result cache | no |

`sentiment_core` exists so the pilot and production runs are comparable: both
use the same Pydantic schema, the same `sentiment_prompt.md`, the same `PANEL`,
and the same call path through `common/llm_provider.py`. A pilot whose prompt or
panel has drifted from production measures nothing.

## Analysis dimensions

### Centralité (Centrality)
How central Islam/Muslims are to the article:

- **Très central** — Islam/Muslims are the main subject
- **Central** — important theme shared with others
- **Secondaire** — mentioned significantly but secondary
- **Marginal** — brief or anecdotal mention
- **Non abordé** — no mention

### Subjectivité (Subjectivity)
How much the article commits itself on the subject, independent of whether the
treatment is favourable:

- **Très objectif** — pure facts, no opinions
- **Plutôt objectif** — mostly factual with subtle opinions
- **Mixte** — balanced mix of facts and opinions
- **Plutôt subjectif** — clear opinions supported by facts
- **Très subjectif** — strongly biased, editorial style

**Generation 2 asks for the label; generation 1 asked for the integer 1–5.**
Not cosmetic: subjectivité was the one dimension requested as a number and is by
a distance the least reliable — pairwise κ within the pilot panel ran
**0.093–0.470** against 0.248–0.478 for polarité and up to 0.725 for centralité,
and one model reproduced its own answer only 47% of the time. Numeric scales are
a documented cause of exactly that ([arXiv:2406.11980](https://arxiv.org/abs/2406.11980)
finds numeric scores reduce compliance and accuracy across every model tested),
and Omeka always stored a link to a labelled item anyway, so nothing downstream
needed the integer. `SUBJECTIVITE_ORDER` keeps the 1–5 ranking available for
ordinal maths, and generation-1 links are read back as labels so both
generations compare on one scale.

The schema field is still called `subjectivite_score` — it is the live HF column
name and Omeka property suffix — but it holds a label.

### Polarité (Polarity)
Sentiment towards Islam/Muslims:

- **Très positif** — extremely favourable portrayal
- **Positif** — favourable, optimistic
- **Neutre** — balanced or factual without emotion
- **Négatif** — critical, pessimistic
- **Très négatif** — extremely unfavourable, alarmist
- **Non applicable** — subject not addressed

## Model provenance — read this before comparing generations

Two generations of annotations coexist on the same items. They live in separate
properties and a run never touches the other generation's values.

### Generation 2 (from 2026-07-31) — what `01` writes now

Properties are named for the **model**. That name is the provenance — and it
is the only thing that can be, because Omeka does not index value annotations:

| Property prefix | Model | Authority item | HF column prefix |
|---|---|---|---|
| `iwac:gemini35FlashLite*` | `gemini-3.5-flash-lite` | 79617 | `gemini_3_5_flash_lite_` |
| `iwac:gpt56Luna*` | `gpt-5.6-luna` | 79610 | `gpt_5_6_luna_` |
| `iwac:mistralSmall2603*` | `mistral-small-2603` | 79614 | `mistral_small_2603_` |
| `iwac:qwen35A10b*` | `qwen/qwen3.5-122b-a10b` | 79616 | `qwen3_5_122b_a10b_` |
| `iwac:deepseekV4Flash0731*` | `deepseek/deepseek-v4-flash-0731` | 83261 | `deepseek_v4_flash_0731_` |

The superseded preview remains under `iwac:deepseekV4Flash*` with authority
item 79613. Those properties are historical and are never repointed to 0731;
mixing two checkpoints in one property set would destroy model provenance.

An `iwac:sentimentModel` value annotation was written alongside until
2026-07-31 and has been **dropped**. Verified live: a query for
`iwac:sentimentModel = 79613` returned **0 items** while **498** carried exactly
that annotation, and `GET /api/value_annotations` is a 500. It was an
unreachable second copy of what the property name already says, written six
times per model per item.

That same finding is why the panel keeps thirty model-keyed properties rather
than six multi-valued ones. The tidier design — one `iwac:polarite` holding a
value per model — would need no vocabulary change to add a model, but it puts
the only thing distinguishing those values in the unsearchable layer, so
*"polarité = Négatif according to DeepSeek"* stops being answerable by query.
Display modules can group either layout; retrieval cannot.

The panel is defined once in `sentiment_core.PANEL` and everything else is
derived from it — the property terms, the ontology, the cache keys, the pilot.
Property **IDs** are resolved from Omeka at startup rather than hardcoded,
because Omeka assigns them when the vocabulary is updated and a stale ID would
write sentiment into the wrong property.

### Generation 1 (January–February 2026) — read-only

The Omeka properties are keyed by **vendor** (`iwac:gemini*`, `iwac:chatgpt*`,
`iwac:mistral*`) and carry **no `iwac:*Model` value annotation**. The property
name therefore does not identify the model that produced a value.

The stored corpus was annotated in **January–February 2026** (campaign window
verified from `o:modified` on the annotated items), by:

| Omeka property prefix | Model that produced it | Hugging Face column prefix |
|---|---|---|
| `iwac:gemini*` | `gemini-3-flash-preview` (Gemini 3 Flash) | `gemini_3_flash_preview_` |
| `iwac:chatgpt*` | `gpt-5-mini` (GPT-5 mini) | `gpt_5_mini_` |
| `iwac:mistral*` | `ministral-14b-2512` (Ministral 3 14B) | `ministral_14b_2512_` |

#### Exact generation-1 run configuration

Recovered from commit `07fb007` (2026-01-27), which was the live code for the
whole campaign — the pipeline then called each SDK directly, before the
`llm_provider` refactor (`fbc2645`, 2026-02-10). The three models did **not**
share a configuration:

| Model | Temperature | Reasoning / thinking | Other |
|---|---|---|---|
| `gemini-3-flash-preview` | `0.2` | **none set** | `response_mime_type=application/json`, `response_schema`, system_instruction |
| `gpt-5-mini` | **none sent** | **none set** | `chat.completions.parse` with `response_format` |
| `ministral-14b-2512` | `0.2` | **none set** | `response_format`, **`max_tokens=512`** |

Three details that are easy to miss and matter for reproducing a value:

- **No model ran with any reasoning or thinking parameter.** `thinking_level`
  only entered this repo on 2026-02-16 (`5748358`), after the campaign; the
  Gemini call above sets temperature and schema only. Any recollection of the
  v1 run using `thinking_level="low"` is mistaken — `low` was the *Pro* default
  introduced later, and Flash's later default was `MINIMAL`.
- **Ministral capped output at `max_tokens=512`**, the only model with a cap.
  Long justifications could be truncated for that model and not the others.
- **GPT-5 mini sent no temperature at all**, so it ran at the API default while
  the other two were pinned to 0.2.

> **Rule:** a new model gets a new property set named for the model. Never
> reuse a vendor slot — that is what made generation 1 ambiguous.

## Adding a model to the panel

1. Add it to `MODEL_REGISTRY` in `common/llm_registry.py`.
2. Create its authority item in Omeka (class 244, template 3, item set 267,
   `dcterms:type` → "Notice d'autorité") and add it to `AI_MODEL_ITEMS` in
   `common/iwac_config.py`.
3. Add a `PanelMember` to `PANEL` in `sentiment_core.py`.
4. Regenerate the ontology and update the vocabulary:

```bash
python AI_sentiment_analysis/00_setup_properties.py --emit-ttl
```

Paste the block into `iwac-vocabulary.ttl`, then pre-flight the upload:

```bash
python AI_sentiment_analysis/00_setup_properties.py --verify
```

This proves the `.ttl` is a **superset** of what is installed. It matters:
Omeka's vocabulary update deletes any installed property the uploaded file
omits, taking every value stored under it across the archive with it. Upload via
**Admin → Vocabularies → IWAC Ontology → Update**; Omeka shows the same diff
before committing, and if it lists anything to delete, stop.

The properties cannot be created through the REST API on this instance:
`PropertyAdapter::hydrate()` in Omeka S 4.2.x never reads `o:vocabulary`, so
`POST /api/properties` always fails validation with "A vocabulary must be set".
The hydration exists on `develop`, so a future upgrade will unlock it.

## Usage

### Production run

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123
```

The whole article corpus, which is the usual generation-2 target:

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36
```

A trial first is worth it — 50 items, no writes:

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --limit 50 --dry-run
```

| Flag | Effect |
|---|---|
| `--item-set-id` | One or more item sets, comma-separated |
| `--resource-class-id` | A whole resource class (bare flag defaults to 36, articles) |
| `--models` | Run part of the panel — see below |
| `--concurrency N` | Items annotated in parallel (default 6; `1` = the old serial loop) |
| `--limit N` | Stop after N items — for a trial run |
| `--dry-run` | Analyse and cache, but PATCH nothing |
| `--skip-update` | Analyse and cache only; never contacts Omeka for writes |
| `--force-reanalyze` | Ignore the cache **and** the already-annotated guard |
| `--rewrite` | Re-PATCH items that already have values, reusing cached answers (no model calls) |
| `--yes` | Skip the confirmation prompt (for unattended runs) |
| `--verbose` | Log each model failure as it happens |

### One model at a time

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --models qwen3_5_122b_a10b
```

A first-class mode, not a degraded one. Each member owns six properties, so
running them one after another builds exactly the same result as running all
five together — with a far smaller blast radius per run, and a real read on one
model's cost and failure rate before committing to the next.

A scoped run **reads and writes only the models named**. Values already on the
item from another member are neither re-requested nor rewritten, and a member
already annotated in Omeka is skipped even on a machine with a cold cache.

### Throughput — pick `--concurrency` per model

The run is almost entirely *waiting on someone else's API*, so the serial loop
this pipeline used until 2026-07-31 spent the corpus's worth of latency doing
nothing. Items now go through a pool of `--concurrency` workers.

Median call latency, measured 2026-07-31 on real articles (2.1–3.7k chars) at
the panel's reasoning setting, 5 concurrent requests, zero rejections:

| Model | Median | Notes |
|---|---|---|
| Gemini 3.5 Flash-Lite | **3.8 s** | 1.1 s at `LOW`, 1.4 s at `MINIMAL` |
| GPT-5.6 Luna | **5.8 s** | 4.3 s at `low` |
| Mistral Small 4 | **5.8 s** | 2.1 s at `none` |
| DeepSeek V4 Flash 0731 | *not benchmarked yet* | New official release; 8 OpenRouter endpoints at adoption |
| Qwen3.5 122B-A10B | **104 s** | 4 usable endpoints — see below |

Every provider transport now has a finite deadline. `--model-timeout` is the
total budget across the pipeline's three attempts (120 seconds by default);
the runner subtracts retry backoff and assigns the remainder to the individual
SDK calls, so a timed-out future cannot leave an unbounded HTTP thread behind.

**The three first-party models finish the corpus in a couple of hours** at the
default concurrency. Nothing about the prompt or the reasoning level was ever
the bottleneck: Qwen is *faster* at `medium` (104 s) than at `low` (116 s),
which is not a reasoning cost, it is queueing.

**Qwen is the outlier and it is a serving problem, not a model problem.**
OpenRouter lists only 5 endpoints for it — one of which does not support
structured outputs, so `require_parameters` leaves four — against 22 for
DeepSeek. The good news is that the latency is queueing rather than a
throughput wall, so it divides cleanly by concurrency (verified: 8 concurrent
calls complete in the time of 1). Give it a much larger pool:

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --models qwen3_5_122b_a10b --concurrency 24
```

`--concurrency` multiplies with the per-item model fan-out. Running one member
at a time — the normal mode — keeps requests in flight equal to the flag;
running all five multiplies it by five.

### When the money runs out

A provider returning **402** (or a 429 that names a daily/billing cap) stops the
run on the first occurrence, prints what the provider said, and exits 2. It is
not counted as a model failure.

This is worth its own path because the alternative already happened: an
OpenRouter balance ran dry around article 11,500 of a DeepSeek pass, and because
nothing recognised 402 as terminal, `analyze_with_model` retried each remaining
call three times with backoff. The run walked the rest of the corpus, produced
**823 identical failures**, and reported them as if the model had misbehaved —
then a retry reproduced exactly 823 again, because a dead account is
deterministic. Diagnosing it needed a direct API call; the summary table gave no
hint.

Nothing is lost when it halts: the cache flushes per record, so topping up and
re-running the same command resumes.

### Language

Only articles whose `dcterms:language` is **Français** or **Anglais** are
annotated. On the article corpus that is 12,305 of 12,356; the 45 Ewé, Kabiyè and
Dendi articles and the 6 with no language value are skipped and counted
separately in the summary.

`dcterms:language` is a **link to an authority item**, not a literal, so the
label is read off `display_title` — there is no ISO code. The reason for the gate
is that a French-prompted model does not fail visibly on an Ewé article; it
returns a confident, unusable score that is indistinguishable from a real
annotation once stored. Same reasoning that got the 2026-07 `ocr_quality` column
reverted before it shipped.

The run reports an estimated duration before asking to proceed. At roughly 15 s
per item — five models in parallel, the slowest deciding the item — the full
12,356-article corpus is on the order of two days. It is built to be
interrupted; see below.

### Piloting a candidate panel

```bash
python AI_sentiment_analysis/02_pilot_new_panel.py --sample-size 200 --seed 42
```

Samples already-annotated articles, runs the candidate panel on them, and
writes `cache/pilot/pilot_<timestamp>.json` containing both the new
annotations and the generation-1 ones for the same articles. Nothing is written
to Omeka.

To measure self-consistency rather than just agreement:

```bash
python AI_sentiment_analysis/02_pilot_new_panel.py --repeats 3 --sample-size 50
```

Then report:

```bash
python AI_sentiment_analysis/03_pilot_report.py
```

The report gives, per dimension: agreement with the generation-1 consensus
(with the generation-1 models scored against their own consensus as a
baseline), pairwise Cohen's kappa within the candidate panel, and — when the
pilot used `--repeats` > 1 — how often each model reproduces its own answer.

That last one matters because sampling temperature is vendor-owned and varies
across the panel: DeepSeek V4 runs at 1.0, Qwen3.5 at 0.7, Mistral Small 4 at
0.3, Gemini unset. Without a self-consistency figure, a low agreement score for
a high-temperature model cannot be told apart from noise. The 2026-07-29 pilot
measured DeepSeek at **0.52** polarité self-consistency against 0.70–0.80 for
the rest, so this is not a hypothetical concern.

## The prompt

`sentiment_prompt.md` was rewritten on 2026-07-31, the first change since it was
committed. **Generation 1 and all three 2026-07 pilots ran the original text**, so
a v1↔v2 difference now confounds model change with prompt change — say so rather
than attributing it to the models.

What changed and why:

- **Removed the checklist and self-verification instructions.** They asked for
  output that had nowhere to go: the call is `generate_structured()` against a
  six-field schema, so a "checklist of conceptual steps" could only be ignored
  or crammed into a justification. They were also redundant now the whole panel
  runs with reasoning enabled — explicit chain-of-thought is
  [counterproductive on reasoning models](https://karozieminski.substack.com/p/ai-prompting-techniques-reasoning-models-2026).
- **Subjectivité became a label** (see above).
- **Added worked examples** — since removed on 2026-08-03, after an A/B measured
  them anchoring the label distribution toward the labels the worked set
  over-represented. Prose boundary rules replaced them.
- **Disambiguated polarité.** It measures *the article's* stance, not a quoted
  source's: reporting a hostile statement with attribution and contrepoint is
  Neutre; endorsing or amplifying it is Négatif. Also that factual reporting of
  an attack is Neutre unless responsibility is extended to Muslims generally.
- **Added centralité boundary rules** for the two highest-frequency ambiguous
  cases: a Muslim actor in a secular story (Non abordé unless religion is
  thematised), and cooperation with Arab states or Islamic organisations —
  Libye 383 articles, "saoudite" 1,559, Koweït 368, OCI 247, Iran 212, ISESCO 48,
  so this is several percent of the corpus, not an edge case. Such cooperation
  is at least **Marginal** even when the surface topic is a loan or a hospital.
- **Added an OCR-noise instruction.** Nothing previously told the model how to
  behave on a garbled article. Corrected on 2026-08-03: the first version said
  truncated words and stray characters were *frequent*, which is not true of
  this corpus — extraction is mostly vision-model based and the text is
  generally clean. Overstating the noise hands the model a reason to reach for
  the illegibility escape hatch, so the instruction now covers the rare case
  without claiming it is common.

Every cache record and pilot manifest carries a **prompt fingerprint**
(`sentiment_core.prompt_fingerprint`, a short sha256 of the text actually sent).
Prompt wording moves label distributions in ways a diff does not predict
([arXiv:2406.11980](https://arxiv.org/abs/2406.11980)), so a stored value whose
prompt is unknown cannot be compared with one whose prompt is known. A hash
rather than a hand-maintained version string, because the latter is exactly what
gets forgotten in the edit that mattered.

The `--without-examples` flag and `load_system_prompt(include_examples=…)` were
removed with the worked-example section; keeping a flag whose only arm no longer
exists would fail at load time rather than mean anything.

## The generation-2 panel

Defined once in `sentiment_core.PANEL`; `01`, `02` and `00` all read it from
there.

| Column prefix | Model id | Omeka properties | Params (active/total) | $/1M in–out |
|---|---|---|---|---|
| `gemini_3_5_flash_lite` | `gemini-3.5-flash-lite` | `iwac:gemini35FlashLite*` | closed | $0.30 / $2.50 |
| `gpt_5_6_luna` | `gpt-5.6-luna` | `iwac:gpt56Luna*` | closed | $1.00 / $6.00 |
| `mistral_small_2603` | `mistral-small-2603` | `iwac:mistralSmall2603*` | **6.5B / 119B** | $0.15 / $0.60 |
| `qwen3_5_122b_a10b` | `qwen/qwen3.5-122b-a10b` | `iwac:qwen35A10b*` | **10B / 122B** | $0.26 / $2.08 |
| `deepseek_v4_flash_0731` | `deepseek/deepseek-v4-flash-0731` | `iwac:deepseekV4Flash0731*` | **13B / 284B** | from $0.09 / $0.18 |

Property prefixes are the camelCase fold of the column prefix, so the Omeka→HF
mapping is mechanical. Qwen is the one exception — the literal fold would be
`qwen35122bA10b`, so the parameter count is dropped and the `A10B` active-params
tag kept. Every property records its exact model id in `rdfs:comment` regardless.

### Why these five

Every member is its vendor's **high-volume tier**, which is what makes the panel
a panel rather than a quality ladder. The slot Gemini occupies was
`gemini-3.6-flash` until 2026-07-31; at $1.50/$7.50 it cost five to seventeen
times the rest, so an inter-model disagreement could always be read as "the
expensive model knows better" rather than as two readings of the construct.
Flash-Lite is Google's actual counterpart to Luna and Mistral Small.

**Three of the five are open weights and can be re-run locally** — which for an
archive is the difference between an annotation you can cite and one you can
only have taken on trust:

| Model | Licence | Hugging Face |
|---|---|---|
| Mistral Small 4 | Apache-2.0 | [`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| Qwen3.5 122B-A10B | Apache-2.0 | [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) |
| DeepSeek V4 Flash 0731 | MIT | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |

Mistral Small 4 is the one served by its *vendor's* API rather than OpenRouter,
which is why an earlier version of this table called it closed. It is not: the
API model `mistral-small-2603` and the Apache-2.0 release carry the same
119B/6.5B-active MoE shape (128 experts, 4 active), the same 256k context and
the same `2603` release code. The card does not state weight-identity in so many
words, so treat it as the same release rather than as a proof.

Their three active-parameter counts — 6.5B, 10B, 13B — sit inside a factor of
two, so an agreement figure between them is not quietly measuring model size.

### Reasoning depth — comparable, but not identical

The panel is standardised on a middle reasoning setting. Vendors split on the
parameter name, so `LLMConfig` carries both and each client reads its own.
Verified against the live APIs on 2026-07-29:

| Model | Parameter | Accepted values | Panel setting |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | `thinking_level` | MINIMAL / LOW / **MEDIUM** / HIGH | `MEDIUM` |
| GPT-5.6 Luna | `reasoning.effort` | none / low / **medium** / high / xhigh / max | `medium` |
| Qwen3.5 122B-A10B | `reasoning.effort` (OpenRouter-normalised, ~50% budget) | minimal…xhigh | `medium` |
| DeepSeek V4 Flash 0731 | `reasoning.effort` | low / high / max | `high` (no medium level) |
| **Mistral Small 4** | `reasoning_effort` | **`none` or `high` only** | `high` |

**Mistral is the exception and it cannot be fixed by configuration.** Its API
rejects `low` and `medium` with a 400; there is no middle setting to ask for.
`MistralClient` rounds a `medium` request up to `high` so Mistral stays in the
reasoning regime with the rest of the panel rather than dropping to
non-reasoning, but it is doing more reasoning than the other four. State this
in any write-up that compares the panel members.

Mistral also changes its response shape once reasoning is on: `message.content`
becomes a `thinking` + `text` chunk list instead of a string, which the SDK's
`chat.parse()` cannot read. The client detects this and routes reasoning
requests through `chat.complete()`, validating the text chunk against the same
schema.

Temperature is **not** standardised and should not be — it stays vendor-owned
per `MODEL_REGISTRY`, because Google and Alibaba both document a lowered
temperature as a cause of looping.

## Environment variables

Required in `.env`:

```bash
# Omeka S API (required)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# AI APIs (at least one required)
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key

# Open-weights models via OpenRouter (Qwen, DeepSeek) — optional
OPENROUTER_API_KEY=your_openrouter_api_key
```

Scripts skip any model whose credentials are missing and say which, rather than
silently shrinking the panel.

## Omeka S property mappings

Six properties per model, in this order:

| Suffix | Type | Holds |
|---|---|---|
| `Centralite` | resource:item | link into the centralité vocabulary |
| `CentraliteJustification` | literal (`@language: fr`) | one sentence |
| `Polarite` | resource:item | link into the polarité vocabulary |
| `PolariteJustification` | literal (`@language: fr`) | one or two sentences |
| `SubjectiviteScore` | resource:item | link into the 1–5 vocabulary |
| `SubjectiviteJustification` | literal (`@language: fr`) | one or two sentences |

Note that `SubjectiviteScore` is a **link to a controlled-vocabulary item**, not
a numeric literal — readers must resolve it through the item IDs below. The same
is true of `Centralite` and `Polarite`, which generation 1 nonetheless declared
`owl:DatatypeProperty` in the ontology; the generation-2 declarations are
`owl:ObjectProperty` and match what is actually written.

Generation-1 property IDs are 319–336 (`iwac:gemini*` 319–324, `iwac:chatgpt*`
325–330, `iwac:mistral*` 331–336) in the order above. Generation-2 IDs are
assigned when the vocabulary is updated and are **not** hardcoded anywhere —
`01` resolves all 31 terms in a single request at startup via
`common.iwac_config.resolve_property_ids`, and fails loudly naming any that are
missing rather than writing a partial annotation set.

Every generation-2 value additionally carries:

```json
"@annotation": {"iwac:sentimentModel": [{"type": "resource:item",
                                         "value_resource_id": 79611}]}
```

## Controlled vocabulary item IDs

### Centralité
| Value | Item ID |
|---|---|
| Très central | 78048 |
| Central | 78049 |
| Secondaire | 78050 |
| Marginal | 78051 |
| Non abordé | 78052 |

### Polarité
| Value | Item ID |
|---|---|
| Très positif | 78031 |
| Positif | 78038 |
| Neutre | 78039 |
| Négatif | 78040 |
| Très négatif | 78041 |
| Non applicable | 78042 |

### Subjectivité
| Score | Label | Item ID |
|---|---|---|
| 1 | Très objectif | 78043 |
| 2 | Plutôt objectif | 78044 |
| 3 | Mixte | 78045 |
| 4 | Plutôt subjectif | 78046 |
| 5 | Très subjectif | 78047 |

These live in `sentiment_core.py` (`CENTRALITE_ITEM_IDS`, `POLARITE_ITEM_IDS`,
`SUBJECTIVITE_ITEM_IDS`, and the `ITEM_ID_TO_SUBJECTIVITE` reverse map).

## Resuming an interrupted run

A full-corpus run takes days, so **the safe response to any failure is to run
the same command again.** Three mechanisms make that cheap, at three different
granularities:

1. **Omeka is checked first.** An item already carrying values for every panel
   member is skipped without any further API call. This is what makes a resume
   fast even with an empty cache — and it works across machines, because the
   state lives in the archive rather than on disk.
2. **Results are cached per (item, model)**, in `cache/sentiment_v2.jsonl`. On
   resume each model is asked only for what it has not already answered; with
   five models, re-running the whole item because one timed out would waste
   four calls per retry.
3. **Only successes are cached.** An errored call is deliberately not written,
   so the next run retries it. A cache that recorded its own failures would
   converge on a corpus of error placeholders.

The cache is append-only JSONL, flushed after every record. That is a deliberate
replacement for the generation-1 cache, which was one JSON object rewritten in
full after every item — quadratic over 12,356 items, and a crash *during*
`json.dump` truncated the very file that existed to make the run resumable. Here
a killed process costs one unreadable final line, which loading skips and
reports.

Each record carries the model id and effective reasoning depth alongside the
result, so a cache file stays interpretable after the code that produced it has
moved on. The generation-1 cache stored neither, which is part of why
attributing the stored corpus needed a dig through git history.

`--force-reanalyze` bypasses both the cache and the Omeka guard. It appends
rather than rewriting, so earlier answers stay in the file as an audit trail
while the newest wins on load.

The generation-1 cache (`cache/sentiment_cache.json`) is left in place and is no
longer read; its keys are vendor slots holding results from three models that
are not on the current panel.

Pilot output goes to `cache/pilot/` and carries a manifest recording the exact
model ids, sample seed and repeat count for the run.

## Architecture

- `common/omeka_client.py` — authenticated Omeka S API access with retry logic
- `common/llm_provider.py` — unified LLM interface with native structured outputs
- `common/console_utils.py` — the shared progress bar and table furniture

A single generic `analyze_with_model()` handles every provider; models run
concurrently via `ThreadPoolExecutor`. No script sets `temperature` — it is
vendor-owned and lives once in `MODEL_REGISTRY`.

## Dependencies

```bash
pip install -e ".[dev]"
```
