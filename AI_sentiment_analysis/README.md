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

Omeka now carries generation 2 alone. Generation 1 was deleted on 2026-08-07
and survives on Hugging Face, so a cross-generation comparison reads the Hub for
one side and Omeka for the other — never Omeka for both.

### Generation 2 (from 2026-07-31) — what `01` writes now

Properties are named for the **model**. That name is the provenance — and it
is the only thing that can be, because Omeka does not index value annotations:

| Property prefix | Model | Authority item | HF column prefix |
|---|---|---|---|
| `iwac:gemma431bIt*` | `google/gemma-4-31b-it` | 111663 | `gemma_4_31b_it_` |
| `iwac:gpt56Luna*` | `gpt-5.6-luna` | 79610 | `gpt_5_6_luna_` |
| `iwac:mistralSmall2603*` | `mistral-small-2603` | 79614 | `mistral_small_2603_` |
| `iwac:deepseekV4Flash0731*` | `deepseek/deepseek-v4-flash-0731` | 83261 | `deepseek_v4_flash_0731_` |

**The Google slot became Gemma 4 31B on 2026-08-14**, and it annotated the corpus
the same day: 18.8 h, then a short retry pass, ending **complete at 12,298
articles — exact parity with the other three members**. It was
`gemini-3.5-flash-lite` from 2026-07-31, but Flash-Lite never annotated
anything: `iwac:gemini35FlashLiteCentralite` was verified at **0 items** on the
live archive on the day of the swap, and `00 --verify` re-confirmed it as empty.
So this filled an empty slot rather than mixing two models into one column —
**generation 2 is unchanged**, and there is no mixed-column question to resolve
and no re-run to pay for.

The upload that added Gemma's six properties did **not** remove Flash-Lite's, or
any of the other 42 empty declarations: vocabulary 10 went 74 → **80**, not
74 → 32. Whatever its diff preview lists, the update flow adds on this instance
and does not delete, so `00 --verify` proves a deletion *would* be safe rather
than predicting one will happen. Harmless — all 48 are at 0 items and
`resolve_property_ids` asks for the 25 terms the panel needs — but do not count
annotators from the installed property list.

The April preview's `iwac:deepseekV4Flash*` values (11,482 items) were deleted
on 2026-08-07. They were never exported to Hugging Face, so that reading is
gone; `deepseek_v4_flash_0731_` on the Hub is a different run, not a newer copy
of the same one.

**Qwen3.5 122B-A10B was dropped from the panel on 2026-08-05** without ever
annotating an article, so `iwac:qwen35A10b*` holds zero values and is not part of
the ontology `00 --verify` expects. The reason was serving, not quality:
OpenRouter listed 5 endpoints for it against DeepSeek's 22, one of them without
structured-output support, so `require_parameters` left four. The resulting
queueing put its median call at **104 s** against 4–6 s for the rest of the panel
— a corpus pass in days rather than hours. Do not re-add it without checking
endpoint availability first; the wall was never the prompt or the reasoning level
(it was marginally *faster* at `medium` than at `low`).

An `iwac:sentimentModel` value annotation was written alongside until
2026-07-31 and has been **dropped**. Verified live: a query for
`iwac:sentimentModel = 79613` returned **0 items** while **498** carried exactly
that annotation, and `GET /api/value_annotations` is a 500. It was an
unreachable second copy of what the property name already says, written six
times per model per item.

That same finding is why the panel keeps six model-keyed properties per member —
twenty-four across the four — rather than six multi-valued ones. The tidier design — one `iwac:polarite` holding a
value per model — would need no vocabulary change to add a model, but it puts
the only thing distinguishing those values in the unsearchable layer, so
*"polarité = Négatif according to DeepSeek"* stops being answerable by query.
Display modules can group either layout; retrieval cannot.

The panel is defined once in `sentiment_core.PANEL` and everything else is
derived from it — the property terms, the ontology, the cache keys, the pilot.
Property **IDs** are resolved from Omeka at startup rather than hardcoded,
because Omeka assigns them when the vocabulary is updated and a stale ID would
write sentiment into the wrong property.

### Generation 1 (January–February 2026) — deleted from Omeka, kept on the Hub

The vendor-keyed properties (`iwac:gemini*`, `iwac:chatgpt*`, `iwac:mistral*`,
12,286 items each) were **deleted from Omeka on 2026-08-07**, after the values
were confirmed present on the Hugging Face full mirror. Nothing in this pipeline
reads them any more. The archive is:

| Hugging Face column prefix | Model that produced it | Run configuration |
|---|---|---|
| `gemini_3_flash_preview_` | `gemini-3-flash-preview` | temperature `0.2`, `response_schema` |
| `gpt_5_mini_` | `gpt-5-mini` | no temperature sent, `response_format` |
| `ministral_14b_2512_` | `ministral-14b-2512` | temperature `0.2`, **`max_tokens=512`** |

Those columns are frozen on the Hub with `omeka_prefix=None` in the uploader's
panel, so `hub_merge` preserves what the uploader no longer emits. **That freeze
is now the only copy** — unfreezing it, or dropping the entries, deletes the
campaign.

Three things matter when reading those columns, none of them recoverable from
the data:

- **No model ran with any reasoning or thinking parameter** — `thinking_level`
  postdates the campaign. A recollection of v1 running at `thinking_level="low"`
  is mistaken.
- **Ministral alone capped output at 512 tokens**, so its long justifications
  could be truncated where the others' were not.
- **GPT-5 mini sent no temperature**, running at the API default while the other
  two were pinned to 0.2.

Generation 1 also carried no model annotation, so the property name named a
*vendor* and nothing recorded which model ran; the mapping above was recovered
from commit `07fb007`.

> **Rule:** a new model gets a new property set named for the model. Never
> reuse a vendor slot — that is what made generation 1 ambiguous, and
> `test_panel_does_not_reuse_an_abandoned_property` now pins it.

## Adding a model to the panel

A candidate is piloted **before** any of this, from `PILOT_CANDIDATES` in
`sentiment_core.py` — a staging list `02` runs and `01` cannot see. Membership of
`PANEL` is what makes a model writable, so a candidate parked in the staging list
is one that cannot reach Omeka however the scripts are invoked, and no properties
or authority items need to exist while it is being judged. Steps 1 and 3 below
are what promotion means; step 2 is the record that has to exist first.

1. Add it to `MODEL_REGISTRY` in `common/llm_registry.py`.
2. Create its authority item in Omeka (class 244, template 3, item set 267,
   `dcterms:type` → "Notice d'autorité") and add it to `AI_MODEL_ITEMS` in
   `common/iwac_config.py`.
3. Move its `PanelMember` from `PILOT_CANDIDATES` into `PANEL` in
   `sentiment_core.py`.
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
python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --models deepseek_v4_flash_0731
```

A first-class mode, not a degraded one. Each member owns six properties, so
running them one after another builds exactly the same result as running all
four together — with a far smaller blast radius per run, and a real read on one
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
| GPT-5.6 Luna | **5.8 s** | 4.3 s at `low` |
| Mistral Small 4 | **5.8 s** | 2.1 s at `none` |
| DeepSeek V4 Flash 0731 | **~55 s** | Measured end to end (below), not on the 07-31 bench |
| Gemma 4 31B | **~72 s** | 3 calls, one article, 2026-08-14; 13 s with no effort sent |

Gemma is the slowest member and it is not close: 51–142 s per call against
DeepSeek's 17–42 s in the same sample. Two things drive it — reasoning is the
larger part (13 s without it), and OpenRouter's routing puts the work on
whichever third-party backend is free, which is why the spread is so wide. **This
is the risk the Qwen3.5 rejection was about, and Gemma clears the bar Qwen
failed**: 16 of its 19 OpenRouter endpoints carry structured outputs, against
Qwen's four, so the queueing that put Qwen at a 104 s median does not apply.
Expect a corpus pass in DeepSeek's league or somewhat worse — **budget 40–85 h at
`--concurrency 6`** — and measure it rather than trusting that range.

Every provider transport now has a finite deadline. `--model-timeout` is the
total budget across the pipeline's three attempts (120 seconds by default);
the runner subtracts retry backoff and assigns the remainder to the individual
SDK calls, so a timed-out future cannot leave an unbounded HTTP thread behind.

**Full-corpus throughput, timed from the `ts` on all 12,305 cache records at
`--concurrency 6`** (2026-08, so these are real runs rather than a latency bench):

| Model | Wall clock | Items/hour |
|---|---|---|
| GPT-5.6 Luna | **2.7 h** | 4,511 |
| Mistral Small 4 | **3.7 h** | 3,318 |
| DeepSeek V4 Flash 0731 | **31.5 h** | 391 |
| Gemma 4 31B | **18.8 h** | 653 |

DeepSeek is ~12× slower than Luna, and nothing like the retired preview's 9.7 s
median: 0731 has no middle reasoning level, so the panel rounds it up to `high`.
Budget a full day for it and hours for the others.

Gemma's first corpus pass ran **2026-08-14/15**: 18.8 h for 12,240 annotated
articles, 653 items/hour — slower than DeepSeek but the same order, and well
inside the 40–85 h that was budgeted from its ~72 s median call. Per-call latency
under-predicts a pass here, because OpenRouter re-picks a backend per call and
the pool absorbs the slow ones: an 18-item trial ran at 9 items/min, the opening
minutes at closer to 4, and the pass averaged 11.

It ended with **58 model-call failures** out of 12,298 eligible articles (0.5%),
all transient connection drops, plus a handful of `RemoteDisconnected` retries
against Omeka that urllib3 absorbed (PATCH failures: 0). Failures are never
cached — only valid results are — so re-running the same command retried exactly
those: **61 items, 0 failures, 0 re-requested**, and the member came out complete.
That resume path is the reason a 0.5% failure rate costs minutes rather than
another pass.

**Pass `--model-timeout 300` for any 0731 or Gemma run.** The 120 s default
allots 37.3 s per attempt while both models take ~55–72 s per item, so normal
variance crosses a line drawn too tight: a DeepSeek corpus pass produced 91
model-call failures of which 88 succeeded on a plain retry. Only 3 were genuine,
and they cleared immediately at the larger budget. Gemma's slowest probe call was
142 s, so 300 is the floor for it rather than a comfortable margin.

`--concurrency` multiplies with the per-item model fan-out. Running one member
at a time — the normal mode — keeps requests in flight equal to the flag;
running all four multiplies it by four.

### Running a member on your own GPU

A panel member does not have to be someone else's API. `serving/` sets up an
open-weights model on your own hardware — a Slurm cluster with vLLM, or any
OpenAI-compatible endpoint — and the registry reaches it exactly like a hosted
one. Two reasons it matters here specifically, in the order they actually did:

**Cost, for a corpus annotated repeatedly.** Every panel change re-runs ~12,300
articles. Qwen3.8-27B is $0.45/$3.20 per 1M on OpenRouter, roughly twice the
band that has already disqualified candidates. On university hardware the
marginal cost of a pass is queue time.

**Reasoning depth you can actually verify.** This is the bigger one. A router
fans each request across third-party backends that disagree about what an effort
means, which is how Gemma's documented levels collapsed to on/off — `medium` and
`high` indistinguishable in latency and reasoning length. On your own server the
request goes to one process whose logs you can read. `serving/probe_reasoning.py`
measures it; run it before trusting a route's middle setting.

Measured on Festus (University of Bayreuth) **2026-08-16**, Qwen3.8-27B bf16 on
2× L40, one francophone article, two calls per level:

| Level | Median | Completion tokens | Reasoning chars |
|---|---|---|---|
| `low` | 28.5 s | 714 | 2,370 |
| `medium` | 37.0 s | 947 | 3,663 |
| `xhigh` | 89.1 s | 2,299 | 8,249 |

**The ladder is real**, and this is the first candidate since GPT-5.6 Luna where
the panel's requested `medium` is a rung the model has rather than one it gets
rounded up to. Weights load in ~150 s; guided decoding held, so no response
needed unfencing.

**The open question is not reasoning depth but validity.** A full-corpus pass
ran 2026-08-17/18 — all 12,251 eligible articles, `medium`, prompt
`#d14ace9ac192`, offline on Festus across three shards (one L40S, two H100) —
followed by three retry rounds that ended 2026-08-24.

| | |
|---|---|
| Annotated, first pass | 10,975 / 12,251 (89.6%) |
| Annotated, after 3 retries | **12,098 / 12,251 (98.75%)** |
| Never annotated | 153 (1.25%), each attempted exactly 4 times |
| Throughput | 361 articles/h on 2× L40S, 540–576 on one H100 |
| Wall clock | ~7 h per 4,084-article shard on H100; retries 23–36 min per round |

The fault is always the same: a null `subjectivite_score` beside a non-null
centralité, which the schema's cross-field validator rejects. Guided decoding
constrains shape, never logic. The first-pass rate held at **10.6% / 9.9% /
10.8%** across three independent slices and two hardware configurations, so it
is a property of the model on this task, not of the slice or the GPU.

**Retrying converges, but not to zero.** Each round recovered slightly under
half of what was left, and the per-round failure *rate* climbed as the
population concentrated:

| Round | Attempted | Failed | Failure rate |
|---|---:|---:|---:|
| First pass | 12,251 | 1,276 | 10.4% |
| Retry 1 | 1,276 | 539 | 42.3% |
| Retry 2 | 539 | 278 | 51.6% |
| Retry 3 | 278 | 153 | 55.0% |

That climb is the finding. If the residual were transient — a sampling accident,
as the 200-article pilot suggested when one retry pass took 12.5% to 4.0% — the
rate would stay flat and the population would drain. Instead the survivors get
harder to annotate each round, which is what a hard core looks like: 145 of the
153 are articles the model answers the same invalid way every time it is asked.

**And the residual is not spread evenly across the corpus — it concentrates on
low centrality.** Failure rate by the centralité the model was trying to assign:

| Centralité | Annotated | Never annotated | Failure rate |
|---|---:|---:|---:|
| `Marginal` | 1,440 | 83 | **5.45%** |
| `Secondaire` | 1,049 | 11 | 1.04% |
| `Très central` | 7,576 | 42 | 0.55% |
| `Central` | 1,746 | 9 | 0.51% |
| `Non abordé` | 287 | 0 | **0.00%** |

A tenfold enrichment on `Marginal`, and a clean zero on `Non abordé`. Read
together those two rows say the same thing: the model wants to decline
subjectivité whenever Islam is *peripheral* to an article, and the schema only
licenses declining when Islam is **absent** from it. Where the rule permits a
null it uses one correctly on every single item; where it does not, it tries
anyway and is rejected. This is a disagreement about the instrument, not a
formatting failure — the model is drawing the "nothing to judge" line one notch
further up the centralité scale than the prompt does.

Distributions over the 12,098 valid annotations:

```
centralité    Très central 7,576 · Central 1,746 · Marginal 1,440 · Secondaire 1,049 · Non abordé 287
polarité      Neutre 6,298 · Positif 5,107 · Non applicable 288 · Négatif 224 · Très positif 176 · Très négatif 5
subjectivité  Plutôt objectif 7,431 · Plutôt subjectif 1,815 · Très objectif 1,471 · Très subjectif 797 · Mixte 297 · null 287
```

Two things to weigh before promotion. **Polarité is barely negative** — 229 of
12,098 (1.9%) across a press corpus spanning decades of contested public
argument — and that needs checking against what the live panel assigned on the
same articles before Qwen is trusted on the dimension. In its favour, the null
subjectivité count (287) matches `Non abordé` (287) exactly: the cross-field
rule obeyed perfectly where it *is* obeyed.

**The 153 stay unannotated, and that is the decision.** A fourth retry round
would be ~30 min of H100 time for perhaps 70 recovered items, and the trend says
what it would leave behind. Dropping those items to `low` is ruled out below;
relaxing the validator would let the model's own reading of "peripheral" enter
the data as a null nobody chose. The gap is recorded instead:
`serving/merge_shards.py` writes a failure log beside the merged JSONL listing
every item, its attempt count and the fault it hit on each attempt, so the
shortfall in this member's coverage is documented rather than mistaken later for
a failed run to be repaired.

**Do not "fix" the residual by dropping to `low`.** Run on the 8 articles that
stayed stuck through two passes at `medium`, `low` returned valid output for 6 —
but all six got *identical* `subjectivite_score` (`Très objectif`) and
`polarite` (`Neutre`), labels occurring 8% and 49% of the time at `medium`.
Centralité still varied, so it is the two contested fields flattening rather
than a constant record. On six articles that is suggestive, not conclusive, but
the shape is the failure this project has met before: an unusable answer
indistinguishable from a real one once stored, which is why the 2026-07
`ocr_quality` column was reverted. A visible residual beats invisible defaults,
and mixing depths would forfeit the comparability that made a genuine `medium`
rung this candidate's main argument.

Note also that `xhigh` exceeded the 300 s request timeout on one call in two on
L40s, and that per-call latency at `medium` under `--concurrency 6` has a long
tail: median 80 s, p90 281 s, max 483 s.

**Unattended runs.** A queued job starts when the scheduler says so, and a
tunnel cannot be held open for a 3 a.m. slot. The obvious fix — putting `.env`
on the cluster — means Omeka and provider keys on shared university storage.
`serving/annotate_job.sbatch` avoids the trade by splitting the work where the
secrets are: sample the corpus **on the machine that has the keys**, ship a plain
JSON file of article ids and text, and let the job annotate it against
`localhost` and write JSONL you collect afterwards. The cluster holds no
credentials and contacts nothing but its own server. Results append per article
and completed ids are skipped on restart, so a job killed at its walltime
resumes rather than starting over — and records carrying a different prompt
fingerprint are ignored, so resuming across a prompt edit cannot silently blend
two instruments into one file.

The payload carries the prompt and its fingerprint, so an offline pass is
provably the same instrument as an online one. The 2026-08-16 pilot payload is
`#d14ace9ac192` — what `gpt56Luna`, `deepseekV4Flash0731` and `gemma431bIt` all
ran.

Set `--concurrency` on the offline annotator as you would on `01`: a self-hosted
server is not a rate limit, and serial annotation leaves most of the GPU idle.
At one article at a time the measured 47 s/article would put a full corpus near
160 h; vLLM keeps many sequences resident, which is what brings that back into
range.

**Every article is annotated independently, and concurrency is only sound
because of that.** Each call sends the system prompt and one article — no
conversation history, no running context, no batching of several articles into
one request — and nothing mutable is shared between workers. Reordering,
resuming, or re-running a subset cannot change any other article's answer, which
is what lets per-item agreement statistics mean anything. The one honest caveat
is numerical rather than semantic: a GPU server batches whatever requests happen
to be in flight, and floating-point reduction order varies with batch shape, so
a given article can sample slightly differently depending on what ran beside it.
That is the same class of variation as temperature — which is 1.0 here, and
dominates it — not one article influencing another. Re-annotation was never
reproducible on this panel anyway: repairing 1,485 DeepSeek items returned a
different centralité for 19 of them.

Setup, tunnels, partition choice and what stays private are in
[`serving/README.md`](../serving/README.md).

### Cost — measure it, never infer it from the tier name

Measured full-corpus figures (12,305 articles):

| Model | Full pass | How it was obtained |
|---|---|---|
| DeepSeek V4 Flash 0731 | **$10.95** | measured against the OpenRouter credits endpoint |
| Gemma 4 31B | **~$8–12** projected | 3 calls; 3,940 in / ~1,100 out at $0.09–0.15 / $0.34–0.40. The 2026-08-14/15 pass has now run — measure it against the credits endpoint and replace this row |
| Gemini 3.5 Flash-Lite | **~$47** projected | retired from the panel 2026-08-14; 8 articles, `thinking_level=MEDIUM` |

Gemma's projection is what moved it into the Google slot: it lands in the same
band as the rest of the panel rather than at 4× it, which is the difference
between a panel and a quality ladder. **It is a projection from three calls on one
article — re-measure it against the OpenRouter credits endpoint before quoting
it**, as was done for DeepSeek. The endpoint rates were read live from
`https://openrouter.ai/api/v1/models/google/gemma-4-31b-it/endpoints` on
2026-08-14; the cheapest carrying structured outputs was $0.09/$0.34, the routing
policy will not always pick it, and Gemma emits ~1,100 output tokens per call of
which most is reasoning.

Two traps, both of which have caught this repo:

**Reasoning tokens dominate, and they invert the rate-card ranking.** Gemini
3.5 Flash-Lite averaged 2,852 input / 159 answer / **1,037 thinking** tokens —
so 87% of billed output is thinking, and the output rate lands on it. Assuming
thinking is unbilled gives ~$19 and is wrong by 2.5×. This is why the cheaper
looking tier costs 4× the DeepSeek pass.

**The price in `MODEL_REGISTRY` is a description, not a source of truth.** On
2026-08-06 the `gpt-5.6-luna` entry read `$1/$6 per 1M` against a real
`$0.20/$0.02/$1.20`; a corpus estimate built on it came out 5× high, and
ignoring cached input made it 7× high overall. Re-check
[OpenAI's pricing page](https://developers.openai.com/api/docs/pricing) (or the
provider's) before quoting a figure.

**Prompt caching is not a rounding error.** These pipelines send one long system
prompt with every request, so it is the shared prefix of every call: on the
summary pipeline, 55% of input tokens were served from cache at 10% of the rate,
with 39 of 40 concurrent calls hitting it. Read `cached_tokens`, do not assume
zero.

To measure tokens for any member, replicate the call — `generate_structured()`
returns only the parsed object, and no provider surfaces usage through it:

```python
client = build_llm_client(option, LLMConfig(**panel_reasoning(key)))
cfg = client._build_generation_config(client._get_effective_config(None))
cfg |= {"system_instruction": ..., "response_mime_type": ..., "response_schema": ...}
resp = client._client.models.generate_content(...)      # Gemini
resp.usage_metadata.thoughts_token_count
# OpenAI: responses.parse(...).usage.output_tokens_details.reasoning_tokens
#         and .usage.input_tokens_details.cached_tokens
```

Sample items spread across the corpus. The first page of the article class is
one newspaper's consecutive issues and is not representative.

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
annotated. **The eligible ceiling moves with the corpus and is currently 12,298
of 12,349** (verified live 2026-08-15) — it was 12,305 of 12,356 through the
2026-08 campaigns, so figures of that size elsewhere in this file describe the
corpus as it stood then, not a shortfall. Seven articles left the class between
the two readings and every member's count dropped together, which is the
signature to look for: a gap in one member alone is a failed run, a gap in all
four is the corpus moving. The 45 Ewé, Kabiyè and
Dendi articles and the 6 with no language value are skipped and counted
separately in the summary.

`dcterms:language` is a **link to an authority item**, not a literal, so the
label is read off `display_title` — there is no ISO code. The reason for the gate
is that a French-prompted model does not fail visibly on an Ewé article; it
returns a confident, unusable score that is indistinguishable from a real
annotation once stored. Same reasoning that got the 2026-07 `ocr_quality` column
reverted before it shipped.

The run reports an estimated duration before asking to proceed. With four models
in parallel the slowest decides the item, and since the Google slot became Gemma
that is Gemma at ~72 s rather than DeepSeek at ~55 s, so a whole-panel pass over
the 12,356-article corpus runs to two days or more. It is built to be
interrupted; see below.

### Piloting a candidate panel

```bash
python AI_sentiment_analysis/02_pilot_new_panel.py --sample-size 200 --seed 42
```

Samples already-annotated articles, runs the live panel plus everything in
`PILOT_CANDIDATES` on them, and writes `cache/pilot/pilot_<timestamp>.json`.
Nothing is written to Omeka — and a candidate could not write there anyway,
since `01` iterates `PANEL` alone.

Running both halves in one pass is what makes the report readable: agreement is
only meaningful against the annotators already in use. To trial one candidate
without paying for the rest, name it:

```bash
python AI_sentiment_analysis/02_pilot_new_panel.py --models qwen3_8_27b --sample-size 50
```

A model whose credentials are missing is listed as skipped and the pilot
continues, so a candidate on a self-hosted endpoint costs nothing when the
tunnel is closed.

**Where the panel stands (2026-08-16).** All four live members are complete at
12,298 — `gpt56Luna`, `mistralSmall2603`, `deepseekV4Flash0731` and
`gemma431bIt`. Qwen3.8 27B is the only sentiment work outstanding, and it is
still a *candidate*: staged in `PILOT_CANDIDATES`, unwritable by `01`, pending a
pilot and an add-or-replace decision (issue #12). Nothing about it has touched
Omeka, and no properties have been created for it.

**Currently staged:** Qwen3.8 27B, twice — `qwen3_8_27b` on a self-hosted vLLM
endpoint and `qwen3_8_27b_openrouter` on OpenRouter. Same weights, two routes,
one sample. See [issue #12](https://github.com/fmadore/iwac-ai-pipelines/issues/12)
for what the pilot has to establish, and [`serving/`](../serving/README.md) for
the endpoint. Before judging either on agreement, run
`python serving/probe_reasoning.py`: the panel asks every member for `medium`,
Qwen3.8 is the first candidate since GPT-5.6 Luna whose ladder actually has that
rung, and whether it survives each route is the thing worth knowing first.

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
across the panel: DeepSeek V4 runs at 1.0, Mistral Small 4 at 0.3, Gemini and
Luna unset. Without a self-consistency figure, a low agreement score for
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
| `gemma_4_31b_it` | `google/gemma-4-31b-it` | `iwac:gemma431bIt*` | **31B dense** | from $0.09 / $0.34 |
| `gpt_5_6_luna` | `gpt-5.6-luna` | `iwac:gpt56Luna*` | closed | $1.00 / $6.00 |
| `mistral_small_2603` | `mistral-small-2603` | `iwac:mistralSmall2603*` | **6.5B / 119B** | $0.15 / $0.60 |
| `deepseek_v4_flash_0731` | `deepseek/deepseek-v4-flash-0731` | `iwac:deepseekV4Flash0731*` | **13B / 284B** | from $0.09 / $0.18 |

Property prefixes are the camelCase fold of the column prefix, so the Omeka→HF
mapping is mechanical. Every property also records its exact model id in
`rdfs:comment`.

### Why these four

Every member is its vendor's **high-volume tier**, which is what makes the panel
a panel rather than a quality ladder. The slot Google occupies was
`gemini-3.6-flash` until 2026-07-31; at $1.50/$7.50 it cost five to seventeen
times the rest, so an inter-model disagreement could always be read as "the
expensive model knows better" rather than as two readings of the construct.
Flash-Lite replaced it and was still the panel's cost outlier at $0.30/$2.50, four
times the DeepSeek pass; Gemma 4 31B, at $0.09/$0.34 from the cheapest endpoint
carrying structured outputs, is the first occupant of that slot priced like the
rest of the panel.

**Gemma replaces the Google slot rather than joining as a fifth voice**, and that
is deliberate. Gemma and Gemini come out of the same lab and the same
pretraining-pipeline family, so running both would buy correlated annotator error
— the *preference leakage* effect documented in the LLM-as-judge literature —
which inflates agreement for reasons unrelated to the construct, while making the
panel 2/5 Google.

**Three of the four are open weights and can be re-run locally** — which for an
archive is the difference between an annotation you can cite and one you can
only have taken on trust:

| Model | Licence | Hugging Face |
|---|---|---|
| Gemma 4 31B | Apache-2.0 | [`google/gemma-4-31B`](https://huggingface.co/google/gemma-4-31B) |
| Mistral Small 4 | Apache-2.0 | [`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| DeepSeek V4 Flash 0731 | MIT | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |

Mistral Small 4 is the one served by its *vendor's* API rather than OpenRouter,
which is why an earlier version of this table called it closed. It is not: the
API model `mistral-small-2603` and the Apache-2.0 release carry the same
119B/6.5B-active MoE shape (128 experts, 4 active), the same 256k context and
the same `2603` release code. The card does not state weight-identity in so many
words, so treat it as the same release rather than as a proof.

**Size parity got worse, not better, and it should be stated in any write-up.**
The open-weights members' active parameter counts are 6.5B, 13B and — Gemma being
dense — 31B, a factor of five where the previous pair sat inside a factor of two.
An agreement figure among them is therefore doing slightly more to measure model
size than it was. The alternative, Gemma 4 26B-A4B, would have widened the spread
in the other direction (3.8B active, 3.4× *below* DeepSeek) and given up dense-model
capability on exactly the boundary calls the rubric turns on — *Central* vs
*Secondaire*, *Négatif* vs *Très négatif*. Neither option was neutral; the dense
31B widens the spread less and does not trade away capability to do it.

### Why Gemma is routed through OpenRouter, not `GEMINI_API_KEY`

Gemma runs on the Gemini API and `common/llm_registry.py` has a `gemma-4` key
that reaches it. The panel deliberately uses `gemma-4-openrouter` instead, and
the reason is not performance:

- **Gemma is free-of-charge on the Gemini API with no paid tier, and Google's
  [pricing page](https://ai.google.dev/gemini-api/docs/pricing) states that
  free-tier content *is* used to improve its products.** This pipeline ships
  whole archival articles to whoever serves the model. That is precisely what
  `OPENROUTER_PROVIDER_PREFS`' `data_collection: "deny"` exists to prevent — and
  OpenRouter's own `:free` Gemma variant has the same problem and is filtered out
  by that policy anyway.
- **The free route is capped too tightly to finish anyway.** Measured 2026-08-14
  against the live API: 16,000 **input tokens per minute** for this model
  (`GenerateContentInputTokensPerModelPerMinute`, `quotaValue: 16000`), which a
  ~3,940-token article exhausts four at a time — a 429 after four consecutive
  calls, and ~51 h for the corpus. That is no faster than OpenRouter, for the
  privacy cost.

The per-call latency *is* far better on the Gemini route — **5.4 s** median
against 37–90 s through OpenRouter's third-party backends — so the trade is real
and worth restating whenever the routing is revisited. It is a policy choice, not
an oversight.

### Reasoning depth — comparable, but not identical

The panel is standardised on a middle reasoning setting. Vendors split on the
parameter name, so `LLMConfig` carries both and each client reads its own.
Verified against the live APIs on 2026-07-29, and 2026-08-14 for Gemma:

| Model | Parameter | Accepted values | Panel setting |
|---|---|---|---|
| GPT-5.6 Luna | `reasoning.effort` | none / low / **medium** / high / xhigh / max | `medium` |
| Gemma 4 31B | `reasoning.effort` | **MINIMAL or HIGH only** | `high` |
| DeepSeek V4 Flash 0731 | `reasoning.effort` | low / high / max | `high` (no medium level) |
| **Mistral Small 4** | `reasoning_effort` | **`none` or `high` only** | `high` |

**Only Luna now sits at a genuine middle setting; the other three are rounded
up.** Mistral's API rejects `low` and `medium` with a 400 and `MistralClient`
rounds a `medium` request up to `high`, so it stays in the reasoning regime with
the rest of the panel rather than dropping to non-reasoning. DeepSeek 0731 and
Gemma have no middle level either, and both are rounded up explicitly in
`PANEL_REASONING_OVERRIDES` rather than left to a client fallback, so a run
manifest records a decision instead of an accident.

**The Gemma swap made this worse and the write-up must say so.** With Flash-Lite
— which did have a real `MEDIUM` — the four-model panel split evenly, two members
at a middle setting and two rounded up. It is now **3 of 4 rounded up**, and no
Google model sits at a middle setting at all. That is a genuine cost of the swap,
accepted for the cost, open-weights and data-handling reasons above.

**Gemma's `high` is also the least legible of the three**, because OpenRouter
fans the request across third-party backends serving the same weights and they do
not agree on what an effort means or on how to report it. Measured on one article
on 2026-08-14:

| Effort sent | Output tokens | Reasoning | Latency |
|---|---|---|---|
| none | ~200 | none | 1.9–14.9 s |
| `medium` | 1,092–1,208 | 3.7–4.2k chars | 51–142 s |
| `high` | 1,012–1,140 | 3.3–3.9k chars | 57–79 s |

So Gemma does reason at the panel's setting — but `medium` and `high` are
**indistinguishable** in both latency and reasoning length, i.e. the thinking is
on/off through this route rather than graduated. One backend reasoned at
`minimal` too (897 tokens), and Chutes reports `reasoning_tokens: 1` while
emitting 3.7k characters of reasoning, so the usage counter cannot be trusted to
tell you whether thinking happened. Read the depth as *requested*, never as
measured — unlike Luna, Mistral and DeepSeek, which are each served by one vendor.

**Routing also decides the quantization, and it is not pinned.** Eleven of the
twelve probe calls landed on Chutes, which serves Gemma at **fp4**; other eligible
endpoints serve bf16 or fp8. The annotations therefore come from *a* quantization
of the open weights rather than from the weights themselves, which qualifies the
"re-runnable from archivable weights" claim for Gemma exactly as it already does
for DeepSeek. Pin `quantizations` in `OPENROUTER_PROVIDER_PREFS` if that matters
more than throughput — it is a shared setting, so it would apply to DeepSeek too.

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

# Open-weights models via OpenRouter (DeepSeek, Gemma) — required for two of the
# four panel slots. Gemma must NOT be routed via GEMINI_API_KEY; see above.
OPENROUTER_API_KEY=your_openrouter_api_key

# Only for a candidate served from your own GPU (see serving/README.md).
# No panel member needs these; leaving them unset skips the candidate.
SELFHOSTED_LLM_BASE_URL=http://localhost:8000/v1
SELFHOSTED_LLM_API_KEY=sk-...
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

Generation-2 IDs are
assigned when the vocabulary is updated and are **not** hardcoded anywhere —
`01` resolves the six terms of every selected member in a single request at
startup (24 for a full panel) via `common.iwac_config.resolve_property_ids`, and
fails loudly naming any that are missing rather than writing a partial
annotation set.

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
   four models, re-running the whole item because one timed out would waste
   three calls per retry.
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
