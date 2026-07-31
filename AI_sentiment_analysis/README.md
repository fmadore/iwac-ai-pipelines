# AI Sentiment Analysis Pipeline

Analyse sentiment towards Islam and Muslims in French-language West African
media articles, using several AI models concurrently as an annotator panel.

## Scripts

| Script | Purpose | Writes to Omeka |
|---|---|---|
| `01_sentiment_analysis.py` | Production run: annotate an item set and store results | **yes** |
| `02_pilot_new_panel.py` | Trial a candidate panel on a sample of already-annotated articles | no |
| `03_pilot_report.py` | Agreement report for a pilot run | no |
| `sentiment_core.py` | Shared schema, prompt, vocabulary and analysis calls | no |

`sentiment_core` exists so the pilot and production runs are comparable: both
use the same Pydantic schema, the same `sentiment_prompt.md`, and the same call
path through `common/llm_provider.py`. A pilot whose prompt has drifted from
production measures nothing.

## Analysis dimensions

### Centralité (Centrality)
How central Islam/Muslims are to the article:

- **Très central** — Islam/Muslims are the main subject
- **Central** — important theme shared with others
- **Secondaire** — mentioned significantly but secondary
- **Marginal** — brief or anecdotal mention
- **Non abordé** — no mention

### Subjectivité (Subjectivity)
Score from 1-5 measuring objectivity:

1. **Très objectif** — pure facts, no opinions
2. **Plutôt objectif** — mostly factual with subtle opinions
3. **Mixte** — balanced mix of facts and opinions
4. **Plutôt subjectif** — clear opinions supported by facts
5. **Très subjectif** — strongly biased, editorial style

### Polarité (Polarity)
Sentiment towards Islam/Muslims:

- **Très positif** — extremely favourable portrayal
- **Positif** — favourable, optimistic
- **Neutre** — balanced or factual without emotion
- **Négatif** — critical, pessimistic
- **Très négatif** — extremely unfavourable, alarmist
- **Non applicable** — subject not addressed

## Model provenance — read this before comparing generations

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

`MODEL_KEYS` in `01_sentiment_analysis.py` has since moved on (the OpenAI slot
now points at `gpt-5.6-luna`), so **re-running `01` on already-annotated items
would mix two models under one property**. The script guards against this by
skipping items that already have values, but `--force-reanalyze` removes that
guard — do not use it on the annotated corpus without first deciding where the
new values will live.

> **Rule:** a new model gets a new property set named for the model. Never
> reuse a vendor slot. The canonical panel definition lives in
> `iwac_common/sentiment_panel.py` in the IWAC-Hugging-Face repository.

## Usage

### Production run

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123
```

```bash
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123,456 --skip-update
```

Flags: `--item-set-id` (comma-separated), `--skip-update` (analyse and cache
only), `--force-reanalyze` (bypass cache **and** the existing-values guard —
see the provenance warning above). Omitting `--item-set-id` prompts for it.

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

## The candidate (v2) panel

| Column prefix | Model id | Weights |
|---|---|---|
| `gemini_3_6_flash` | `gemini-3.6-flash` | closed |
| `gpt_5_6_luna` | `gpt-5.6-luna` | closed |
| `mistral_small_2603` | `mistral-small-2603` | closed |
| `qwen3_5_35b_a3b` | `qwen/qwen3.5-35b-a3b` | **Apache-2.0, runs locally** |
| `deepseek_v4_flash` | `deepseek/deepseek-v4-flash` | **open weights** |

The two OpenRouter members are deliberately the *open-weights* releases rather
than a vendor's hosted tier, so the annotations can be regenerated from weights
that are archivable alongside them.

### Reasoning depth — comparable, but not identical

The panel is standardised on a middle reasoning setting. Vendors split on the
parameter name, so `LLMConfig` carries both and each client reads its own.
Verified against the live APIs on 2026-07-29:

| Model | Parameter | Accepted values | Panel setting |
|---|---|---|---|
| Gemini 3.6 Flash | `thinking_level` | MINIMAL / LOW / **MEDIUM** / HIGH | `MEDIUM` |
| GPT-5.6 Luna | `reasoning.effort` | none / low / **medium** / high / xhigh / max | `medium` |
| Qwen3.5 35B-A3B | `reasoning.effort` (OpenRouter-normalised, ~50% budget) | minimal…xhigh | `medium` |
| DeepSeek V4 Flash | `reasoning.effort` (OpenRouter-normalised, ~50% budget) | minimal…xhigh | `medium` |
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

Six properties per model, listed here with the instance property IDs used by
the API:

| Model slot | Property | ID | Type |
|---|---|---|---|
| Gemini | `iwac:geminiCentralite` | 319 | resource:item |
| Gemini | `iwac:geminiCentraliteJustification` | 320 | literal |
| Gemini | `iwac:geminiPolarite` | 321 | resource:item |
| Gemini | `iwac:geminiPolariteJustification` | 322 | literal |
| Gemini | `iwac:geminiSubjectiviteScore` | 323 | resource:item |
| Gemini | `iwac:geminiSubjectiviteJustification` | 324 | literal |
| ChatGPT | `iwac:chatgpt…` | 325–330 | same order |
| Mistral | `iwac:mistral…` | 331–336 | same order |

Note that `SubjectiviteScore` is a **link to a controlled-vocabulary item**, not
a numeric literal — readers must resolve it through the item IDs below.

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

## Caching

`01_sentiment_analysis.py` caches to `cache/sentiment_cache.json`, keyed by
Omeka item ID. The cache allows resuming an interrupted run, automatically
re-analyses entries that recorded an error, and is bypassed with
`--force-reanalyze`. It stores no model identifiers, so it cannot be used to
reconstruct which model produced a cached value.

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
