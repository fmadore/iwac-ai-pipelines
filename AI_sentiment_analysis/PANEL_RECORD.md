# Sentiment panel record

Dated measurements and the decisions they led to, for the generation-2 panel
described in the [README](README.md). The figures here describe the runs as they
happened; the README describes how to run the pipeline now. When a slot changed
hands and why is in [CHANGELOG.md](../CHANGELOG.md) ("Sentiment panel").

## Throughput

Median call latency, measured 2026-07-31 on real articles (2.1–3.7k chars) at
the panel's reasoning setting, 5 concurrent requests, zero rejections:

| Model | Median | Notes |
|---|---|---|
| GPT-5.6 Luna | **5.8 s** | 4.3 s at `low` |
| Mistral Small 4 | **5.8 s** | 2.1 s at `none` |
| DeepSeek V4 Flash 0731 | **~55 s** | Measured end to end, not on the 07-31 bench |
| Gemma 4 31B | **~72 s** | 3 calls, one article, 2026-08-14; 13 s with no effort sent |
| Qwen3.8 27B (self-hosted) | 37 s at one call; median 80 s, p90 281 s under `--concurrency 6` | 2026-08-16/18, festus |

Full-corpus wall clock, timed from the `ts` on the cache records at
`--concurrency 6`:

| Model | Wall clock | Items/hour |
|---|---|---|
| GPT-5.6 Luna | **2.7 h** | 4,511 |
| Mistral Small 4 | **3.7 h** | 3,318 |
| DeepSeek V4 Flash 0731 | **31.5 h** | 391 |
| Gemma 4 31B | **18.8 h** | 653 |
| Qwen3.8 27B | ~7 h per 4,084-article shard on one H100 | 361 on 2× L40S, 540–576 on one H100 |

DeepSeek is ~12× slower than Luna, and nothing like the retired preview's 9.7 s
median: 0731 has no middle reasoning level, so the panel sends it `high`.

Gemma's first corpus pass ran **2026-08-14/15**: 18.8 h for 12,240 annotated
articles. Per-call latency under-predicts a pass, because OpenRouter re-picks a
backend per call and the pool absorbs the slow ones: an 18-item trial ran at 9
items/min, the opening minutes at closer to 4, and the pass averaged 11. It ended
with **58 model-call failures** out of 12,298 eligible articles (0.5%), all
transient connection drops; re-running the same command retried exactly those
(61 items, 0 failures) and the member came out complete.

**`--model-timeout 300` for DeepSeek and Gemma.** The 120 s default allots
37.3 s per attempt while both models take ~55–72 s per item: a DeepSeek corpus
pass produced 91 model-call failures, 88 of which succeeded on a plain retry and
the other 3 at the larger budget. Gemma's slowest probe call was 142 s, so 300 is
the floor for it rather than a comfortable margin.

## Qwen3.8 27B on festus

### Reasoning ladder

Measured **2026-08-16** with `serving/probe_reasoning.py`, Qwen3.8-27B bf16 on
2× L40, one francophone article, two calls per level:

| Level | Median | Completion tokens | Reasoning chars |
|---|---|---|---|
| `low` | 28.5 s | 714 | 2,370 |
| `medium` | 37.0 s | 947 | 3,663 |
| `xhigh` | 89.1 s | 2,299 | 8,249 |

The ladder is real: the panel's requested `medium` is a rung this model has,
which before it only GPT-5.6 Luna could say. `xhigh` exceeded the 300 s request
timeout on one call in two on L40s.

### Corpus pass

Run 2026-08-17/18 at `medium`, prompt `#d14ace9ac192`, offline across three
shards (one L40S, two H100), over the 12,251 articles in its input; three retry
rounds ended 2026-08-24.

| | |
|---|---|
| Annotated, first pass | 10,975 / 12,251 (89.6%) |
| Annotated, after 3 retries | **12,098 / 12,251 (98.75%)** |
| Never annotated | 153 (1.25%), each attempted exactly 4 times |

The fault is nearly always the same: a null `subjectivite_score` beside a
non-null centralité, which the schema's cross-field validator rejects. Guided
decoding constrains shape, never logic. The first-pass rate held at **10.6% /
9.9% / 10.8%** across three slices and two hardware configurations, so it is a
property of the model on this task.

**Retrying converges, but not to zero.** Successive rounds recovered 58%, 48%
and 45% of what was left, and the failure *rate* climbed as the population
concentrated:

| Round | Attempted | Failed | Failure rate |
|---|---:|---:|---:|
| First pass | 12,251 | 1,276 | 10.4% |
| Retry 1 | 1,276 | 539 | 42.3% |
| Retry 2 | 539 | 278 | 51.6% |
| Retry 3 | 278 | 153 | 55.0% |

A transient residual would drain at a flat rate; this one gets harder each
round, which is what a hard core looks like. 145 of the 153 fail the
cross-field rule every time they are asked; the other 8 failed in other ways.

**The residual concentrates on low centrality.** Failure rate by the centralité
the model was trying to assign, over those 145:

| Centralité | Annotated | Never annotated | Failure rate |
|---|---:|---:|---:|
| `Marginal` | 1,440 | 83 | **5.45%** |
| `Secondaire` | 1,049 | 11 | 1.04% |
| `Très central` | 7,576 | 42 | 0.55% |
| `Central` | 1,746 | 9 | 0.51% |
| `Non abordé` | 287 | 0 | **0.00%** |

The model declines subjectivité whenever Islam is *peripheral*, and the
validator accepts a null only when Islam is *absent* (centralité « Non
abordé »). The prompt states the sufficient condition (« Non abordé » ⇒ null)
but never the converse, which exists only in the validator, and retries do not
tell the model why it was rejected. So this is a disagreement about the
instrument rather than a formatting failure, and Qwen's missing subjectivité is
**not missing at random**.

Distributions over the 12,098 valid annotations:

```
centralité    Très central 7,576 · Central 1,746 · Marginal 1,440 · Secondaire 1,049 · Non abordé 287
polarité      Neutre 6,298 · Positif 5,107 · Non applicable 288 · Négatif 224 · Très positif 176 · Très négatif 5
subjectivité  Plutôt objectif 7,431 · Plutôt subjectif 1,815 · Très objectif 1,471 · Très subjectif 797 · Mixte 297 · null 287
```

The null subjectivité count (287) matches « Non abordé » (287) exactly: the
rule is obeyed perfectly where it is obeyed. **Polarité is barely negative** —
229 of 12,098 (1.9%) — and has not yet been checked against what the other
members assigned on the same articles.

**The 153 stay unannotated, and that is the decision.** A fourth round would be
~30 min of H100 time for perhaps 70 items, and the trend says what it would
leave behind. Relaxing the validator would let the model's own reading of
"peripheral" enter the data as a null nobody chose. `serving/merge_shards.py`
writes a failure log beside the merged JSONL listing every item, its attempt
count and the fault each attempt hit.

**Dropping to `low` is ruled out.** On the 8 articles that stayed stuck through
two passes at `medium`, `low` returned valid output for 6 — but all six got
identical `subjectivite_score` (`Très objectif`) and `polarite` (`Neutre`),
labels occurring 8% and 49% of the time at `medium`. On six articles that is
suggestive rather than conclusive, but it is the failure this project has met
before: an unusable answer indistinguishable from a real one once stored.
Mixing depths would also forfeit the genuine `medium` rung that was this
member's main argument.

## Gemma 4 31B

### Why OpenRouter rather than the Gemini API

Gemma runs on the Gemini API, and the registry's `gemma-4` key reaches it. The
panel uses `gemma-4-openrouter` instead:

- **Gemma is free of charge on the Gemini API with no paid tier, and Google's
  [pricing page](https://ai.google.dev/gemini-api/docs/pricing) states that
  free-tier content is used to improve its products.** This pipeline sends whole
  archival articles. OpenRouter's own `:free` Gemma variant has the same problem
  and is excluded by `data_collection: "deny"`.
- **The free route is capped too tightly to finish anyway.** Measured
  2026-08-14: 16,000 input tokens per minute for this model, which a ~3,940-token
  article exhausts four at a time — ~51 h for the corpus.

Per-call latency *is* far better on the Gemini route (5.4 s median against
37–90 s through OpenRouter's backends), so the trade is real. It is a policy
choice, not an oversight.

### Reasoning through OpenRouter is on/off

Measured on one article on 2026-08-14:

| Effort sent | Output tokens | Reasoning | Latency |
|---|---|---|---|
| none | ~200 | none | 1.9–14.9 s |
| `medium` | 1,092–1,208 | 3.7–4.2k chars | 51–142 s |
| `high` | 1,012–1,140 | 3.3–3.9k chars | 57–79 s |

`medium` and `high` are indistinguishable in latency and reasoning length. One
backend reasoned at `minimal` too (897 tokens), and Chutes reports
`reasoning_tokens: 1` while emitting 3.7k characters of reasoning, so the usage
counter cannot say whether thinking happened. Read Gemma's depth as *requested*,
never as measured.

**Routing also decides the quantization, and it is not pinned.** Eleven of the
twelve probe calls landed on Chutes, which serves Gemma at **fp4**; other
eligible endpoints serve bf16 or fp8. Gemma's annotations come from *a*
quantization of the open weights, which qualifies the "re-runnable from
archived weights" claim exactly as it does for DeepSeek. `quantizations` in
`OPENROUTER_PROVIDER_PREFS` would pin it, for every OpenRouter model at once.

## Panel composition

**Size parity.** The open-weights members' active parameter counts are 6.5B
(Mistral Small 4), 13B (DeepSeek V4 Flash), 27B (Qwen3.8, dense) and 31B (Gemma
4, dense) — a factor of about five. An agreement figure among them measures
model size somewhat as well as the construct, and a write-up should say so. Gemma
4 26B-A4B would have widened the spread the other way (3.8B active) and given up
dense-model capability on exactly the boundary calls the rubric turns on.

**Mistral Small 4 is open weights**, although it is reached through Mistral's
own API: `mistral-small-2603` and the Apache-2.0 release share the 119B/6.5B-active
MoE shape (128 experts, 4 active), the 256k context and the `2603` release code.
The model card does not state weight identity in so many words, so treat it as
the same release rather than as a proof.

**Re-annotation is not reproducible**, with temperature vendor-owned (1.0 for
DeepSeek and Qwen): repairing 1,485 DeepSeek items returned a different
centralité for 19 of them. That is why an offline run is imported rather than
re-run.

## Cost

Measured or projected full-corpus figures (12,305 articles):

| Model | Full pass | How it was obtained |
|---|---|---|
| DeepSeek V4 Flash 0731 | **$10.95** | measured against the OpenRouter credits endpoint |
| Gemma 4 31B | **~$8–12** projected | 3 calls; 3,940 in / ~1,100 out at $0.09–0.15 / $0.34–0.40. Not yet re-measured against the credits endpoint |
| Qwen3.8 27B | GPU hours on festus | no token price |
| Gemini 3.5 Flash-Lite | **~$47** projected | retired from the panel 2026-08-14 without annotating |

Three traps, each of which has caught this repo:

- **Reasoning tokens dominate, and they invert the rate-card ranking.** Gemini
  3.5 Flash-Lite averaged 2,852 input / 159 answer / **1,037 thinking** tokens:
  87% of billed output is thinking. Assuming thinking is unbilled is wrong by
  2.5×, which is why the cheaper-looking tier cost 4× the DeepSeek pass.
- **The price in `MODEL_REGISTRY` is a description, not a source of truth.** On
  2026-08-06 the `gpt-5.6-luna` entry read `$1/$6` against a real
  `$0.20/$0.02/$1.20`, and a corpus estimate built on it came out 5× high (7×
  ignoring cached input). Re-check the provider's pricing page before quoting.
- **Prompt caching is not a rounding error.** On the summary pipeline 55% of
  input tokens were served from cache at 10% of the rate. Read `cached_tokens`.

Every client records its tokens, and cost where the provider states it, as
`client.usage` (`common.llm_provider.UsageTotals`). Sample articles spread
across the corpus: the first page of the article class is one newspaper's
consecutive issues.

## When the money ran out

An OpenRouter balance ran dry around article 11,500 of a DeepSeek pass. Nothing
recognised 402 as terminal, so each remaining call was retried three times with
backoff: the run produced **823 identical failures** and reported them as model
misbehaviour, and a retry reproduced exactly 823 again. Since then a 402, or a
429 that names a daily or billing cap, stops `01` on the first occurrence with
exit 2.

## The prompt

`sentiment_prompt.md` was rewritten on 2026-07-31, the first change since it was
committed. Generation 1 and all three 2026-07 pilots ran the original text, so a
generation-1 ↔ generation-2 difference confounds model change with prompt change.

- **Removed the checklist and self-verification instructions.** They asked for
  output with nowhere to go in a six-field schema, and explicit chain-of-thought
  is counterproductive on reasoning models.
- **Subjectivité became a label** (see the README).
- **Added worked examples**, removed on 2026-08-03 after an A/B measured them
  anchoring the label distribution; prose boundary rules replaced them.
- **Disambiguated polarité.** It measures the *article's* stance: reporting a
  hostile statement with attribution and counterpoint is Neutre, endorsing it is
  Négatif, and factual reporting of an attack is Neutre unless responsibility is
  extended to Muslims generally.
- **Added centralité boundary rules** for a Muslim actor in a secular story, and
  for cooperation with Arab states or Islamic organisations (Libye 383 articles,
  "saoudite" 1,559, Koweït 368, OCI 247, Iran 212, ISESCO 48), which is at least
  Marginal even when the surface topic is a loan or a hospital.
- **Added an OCR-noise instruction**, corrected on 2026-08-03 so that it covers
  the rare garbled article without claiming noise is common.

Every member of the current panel ran prompt `#d14ace9ac192`.
