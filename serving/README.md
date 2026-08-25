# Serving your own models

Run an open-weights model on hardware you control, and let the pipelines in this
repo call it exactly as they call a hosted API.

The pipelines reach it through `SELFHOSTED_LLM_BASE_URL`, which is an
OpenAI-compatible endpoint and nothing more specific than that. vLLM on a GPU
cluster is what this directory automates and what the walkthrough below covers,
but llama.cpp's server, LM Studio, TGI and Ollama all speak the same protocol —
if you already have one running, skip to [Point the pipelines at
it](#3-point-the-pipelines-at-it).

## Why

Two reasons, in the order they actually mattered here.

**Cost, for a corpus that is annotated repeatedly.** The sentiment panel runs
four models over ~12,300 articles, and it is re-run whenever the panel changes.
Qwen3.8-27B is $0.45/$3.20 per 1M tokens on OpenRouter — roughly twice the
panel's output-cost band, which is the ladder that has already disqualified
candidates ([issue #12](https://github.com/fmadore/iwac-ai-pipelines/issues/12)).
On university hardware the marginal cost of a pass is queue time.

**Measurability, which turned out to matter more.** A hosted router fans each
request across third-party backends that serve the same weights and disagree
about what a reasoning level means. Gemma 4's documented levels collapsed to
on/off when measured through OpenRouter — `medium` and `high` indistinguishable
in both latency and reasoning length, one backend reporting `reasoning_tokens:
1` while emitting 3.7k characters of it. On your own server the request goes to
one process whose logs you can read. `probe_reasoning.py` exists to hold a route
to that standard before anything is judged on its annotations.

A third reason applies to archival material specifically: the text never leaves
your network. The OpenRouter path pins `data_collection: "deny"`, which is a
contract with a third party; this is the same guarantee without the third party.

## What is here

| File | Role |
|---|---|
| `env.sh` | Every site-specific value, each one `${OVERRIDE:-default}`. Sourced by the other scripts. **The only file you should need to edit.** |
| `setup_env.sh` | One-time, on the login node: creates the vLLM venv, downloads the weights |
| `vllm_serve.sbatch` | The Slurm job that runs the server, for interactive use over a tunnel |
| `annotate_job.sbatch` | Unattended: serves, annotates a prepared corpus, stops |
| `annotate_offline.py` | The annotator that job runs. Local files only, no credentials |
| `merge_shards.py` | Merges the shards a run produced (last success per item) and writes the failure log — what never succeeded, after how many attempts, and why |
| `probe_reasoning.py` | Measures whether reasoning levels differ through a route. Reads Omeka, writes nothing |

There are two ways to use this, and which one fits depends on whether you will
be at the keyboard when the scheduler decides to start your job.

**Interactive** — `vllm_serve.sbatch` plus an SSH tunnel, driven from your
machine. Credentials stay where they already are, every pipeline works
unchanged, and you watch it happen. It needs the tunnel held open for the whole
run, so it suits a job you can start on demand.

**Unattended** — `annotate_job.sbatch`, for an overnight slot or any start time
the queue picks. See [Unattended runs](#unattended-runs) below.

## Prerequisites

- A GPU big enough for the model. Qwen3.8-27B in bf16 is ~56 GB of weights: one
  H100 (80 GB) fits it with room for the KV cache. See
  [Smaller GPUs](#smaller-gpus) if that is not what you have.
- Slurm, if you want the job scripts. Without it, run the `vllm serve` line from
  `vllm_serve.sbatch` by hand.
- Network access from your machine to the cluster — for most sites a VPN plus
  SSH to a login node.

## Walkthrough

### 1. Set up, once, on the login node

```bash
git clone https://github.com/fmadore/iwac-ai-pipelines.git
cd iwac-ai-pipelines
bash serving/setup_env.sh
```

This creates a venv (default `/workdir/$USER/vllm-serve/.venv`) with vLLM ≥0.27.1
and downloads the weights into `$HF_HOME`. Both steps need the internet, which
is why they happen here: compute nodes are frequently offline, and the job runs
with `HF_HUB_OFFLINE=1` against what this left in the cache.

Keep this venv separate from any other vLLM on the cluster. If you also run
[festus-transcribe](https://github.com/AM-Digital-Research-Environment/festus-transcribe),
its 0.21.0 pin is deliberate — upgrading it in place to satisfy this would break
that pipeline instead.

### 2. Start the server

Generate a token first. The server binds `0.0.0.0` so the login node can reach
it, which means every other user on the cluster network can too — a shared
compute node is not a private one.

```bash
export SERVE_API_KEY="sk-$(openssl rand -hex 24)"
sbatch serving/vllm_serve.sbatch
```

Keep the `sk-` prefix: it is what `common/log_redaction.py` masks, so an
accidental echo into a log file stays covered.

Find where it landed, and wait for the weights to load (a minute or two):

```bash
squeue --me --format="%.10i %.9P %.8T %.10M %N"
tail -f serving/logs/vllm-<jobid>.out
```

The job log prints its own hostname and port, plus the exact `ssh -L` line to
copy. Check it is answering, from the login node:

```bash
curl -H "Authorization: Bearer $SERVE_API_KEY" http://<node>:8000/v1/models
```

### 3. Point the pipelines at it

Compute nodes are rarely reachable from outside the cluster, so forward a local
port through the login node. Run this on **your own machine** and leave it open:

```bash
ssh -L 8000:<compute-node>:8000 <user>@<login-host>
```

Then in your local `.env` (gitignored):

```
SELFHOSTED_LLM_BASE_URL=http://localhost:8000/v1
SELFHOSTED_LLM_API_KEY=sk-...
```

Confirm the tunnel carries the same answer the login node got:

```bash
curl -H "Authorization: Bearer $SELFHOSTED_LLM_API_KEY" http://localhost:8000/v1/models
```

From here the model is an ordinary registry entry. Anything that takes a
`--model` will accept `qwen3.8` (see `common/llm_registry.py`), and the sentiment
pilot picks it up as a candidate:

```bash
python serving/probe_reasoning.py
python AI_sentiment_analysis/02_pilot_new_panel.py --models qwen3_8_27b --sample-size 50
```

### 4. Stop it

`scancel <jobid>`. Nothing persists between jobs except the venv and the weight
cache, so the next `sbatch` starts in about the time it takes to load the model.

## Unattended runs

A queued job starts when the scheduler says so, which on a busy cluster can be
the middle of the night. A tunnel cannot be held open for that, and the obvious
fix — putting your `.env` on the cluster so the job can do everything itself —
means your Omeka and provider keys sit on shared university storage.

`annotate_job.sbatch` avoids the choice by splitting the work where the secrets
are. Sample the corpus **on the machine that already has the credentials**, ship
a plain JSON file of article ids and text, and let the job annotate it against
`localhost`. The cluster never sees a key and never contacts anything but its
own server; you collect JSONL afterwards.

```bash
# 1. On your machine: prepare the corpus (needs your Omeka keys)
#    Write {"system_prompt", "prompt_fingerprint", "articles":[{item_id, language, content}]}
scp pilot_input.json <user>@<login-host>:~/iwac-ai-pipelines/serving/work/

# 2. Submit and walk away
ssh <user>@<login-host> "cd ~/iwac-ai-pipelines && sbatch serving/annotate_job.sbatch"

# 3. Later — collect
scp <user>@<login-host>:~/iwac-annotations/annotations-<jobid>.jsonl .
```

The server binds `127.0.0.1` for this job rather than `0.0.0.0`: only the job
talks to it, so there is no port to expose and no API key to manage. A `trap`
stops it on exit, including on `scancel` — an orphaned vLLM would hold the GPU
for the rest of the allocation.

Results are appended a line at a time and completed items are skipped on
restart, because the failure this is built for is the job hitting its walltime
mid-corpus. Resubmitting continues where it stopped. Records written under a
different prompt fingerprint are ignored rather than reused, so resuming across
a prompt edit cannot silently mix two instruments in one file.

Output lands in `serving/work/` on `/workdir` (purged on a schedule, not backed
up) and is copied to `~/iwac-annotations/` on `/home`, which is backed up. That
copy is best-effort and never fails the job.

Each article is annotated independently — system prompt plus one article, no
history, no batching of several articles into one request, nothing mutable
shared between workers. That is what makes `--concurrency` sound, and what
per-item agreement statistics downstream depend on.

**Throughput, measured 2026-08-16 on 2× L40 at `medium`:** 47 s per article
serial, 28 s wall at `--concurrency 6` (127 articles/hour). Weights load in
~135–150 s. A 200-article pass takes ~1.6 h; a full 12,300-article corpus would
be ~97 h at that rate, which is why concurrency is the default and why an H100
without the tensor-parallel split is worth queueing for. Per-call latency rises
with concurrency even as throughput improves — median 138 s and max 246 s with 6
in flight, against 35–58 s serial — so size `--timeout-budget` against the
concurrent figure, not the serial one.

**Expect failures, and read them.** In the same run 3 of 12 articles failed
after all three retries, every one of them the same way: the model returned a
null `subjectivite_score` beside a non-null centralité, which the schema's
cross-field validator rejects. Guided decoding enforces shape, never logic. A
failed item is written to the JSONL with its `analysis_error` and is **not**
counted as done, so re-running retries exactly those — the same rule the main
pipeline uses for its cache. Keep the failures: the rate is a finding about the
model, not noise to be cleaned up.

**Collect with `merge_shards.py`, not `cat`.** A run appends one record per
*attempt*, so an item annotated on the third try appears three times and a
concatenation of the shards is not a corpus. The script keys on `item_id`, keeps
the last record without an `analysis_error`, and refuses to merge shards written
under two different prompt fingerprints:

```bash
python serving/merge_shards.py --shards 'work/full-s*.jsonl' --dry-run   # report only
python serving/merge_shards.py --shards 'work/full-s*.jsonl' --output merged.jsonl
```

It also writes `<output>_failures.json` — every item that never succeeded, how
many times it was attempted, and the fault class of each attempt. That file is
the point. "Keep the failures" only means something if the rate is written down
somewhere other than three 6 MB shards, and a coverage gap that is documented is
a finding, while the same gap undocumented reads later as a run that broke. The
`--dry-run` summary also prints the retry convergence reconstructed from append
order, which is how you tell a transient residual (rate stays flat, population
drains) from a hard core (rate climbs each round) — on the Qwen3.8 full corpus
it climbed 10.4% → 42% → 52% → 55%, and the 153 survivors were retired rather
than retried a fourth time.

## What stays private

Nothing in this directory names a machine, an account, or a secret, and that is
deliberate — it is meant to be run by people who are not you, on clusters that
are not this one.

| | Where it lives |
|---|---|
| Your account name | `~/.ssh/config`, and your shell history. Never in the repo |
| SSH keys | `~/.ssh/`. Untouched by any of this |
| The API token | Cluster: passed to `sbatch` in your environment, never written to a file the repo tracks. Your machine: `SELFHOSTED_LLM_API_KEY` in gitignored `.env` |
| The endpoint URL | `SELFHOSTED_LLM_BASE_URL` in gitignored `.env`; a placeholder ships in `.env.example` |
| Job logs | `serving/logs/`, gitignored — they contain node names and your job ids |

A `~/.ssh/config` block keeps the account out of every command you type:

```
Host mycluster
    HostName <login-host>
    User <account>
```

after which the tunnel is `ssh -L 8000:<compute-node>:8000 mycluster`.

The cluster's *topology* — hostnames, partition names, GPU models — is not
secret and is often documented publicly by the site itself. Treat your account
and your keys as the things that matter.

## Worked example: Festus (University of Bayreuth)

The defaults in `env.sh` and `vllm_serve.sbatch` target this cluster. Its
topology is public, documented alongside
[festus-transcribe](https://github.com/AM-Digital-Research-Environment/festus-transcribe).
Verify the equivalents for your own site with `sinfo -o "%P %G %N"` before the
first submit — none of this is portable, and the partition casing in particular
is a trap.

| | |
|---|---|
| Login | `ssh <account>@festus.hpc.uni-bayreuth.de` — a normal university account works, no separate registration |
| H100 | `--partition=GPU` (uppercase; there is no lowercase `gpu`), `--gres=gpu:h100:1`, 24 h limit |
| L40 / L40S | `--partition=normal`, `--gres=gpu:l40:1` or `gpu:l40s:1`, 24 h |
| Quick tests | `--partition=dev`, `--gres=gpu:l40:1`, 90 min |
| Storage | `/home` 15 GB, backed up · `/workdir` 3 TB, 60 days, not backed up · `/scratch` shared, 10 days |
| Python module | `python/3.12.4` |

`/workdir/<account>` may not exist yet and cannot be created by hand — it
usually appears after your first batch job.

Two vLLM settings are exported by `env.sh` because the venv has no `nvcc`:
`VLLM_USE_FLASHINFER_SAMPLER=0` (its sampler JIT-compiles a CUDA kernel) and
`VLLM_USE_DEEP_GEMM=0` (FP8 models otherwise demand DeepSeek's `deep_gemm`).

Publications using Festus must carry the DFG funding acknowledgement, project
523317330.

## Notes

### Smaller GPUs

`SERVE_MODEL=Qwen/Qwen3.8-27B-FP8` halves the weights to ~28 GB and fits a
single L40S (48 GB), which on many clusters queues faster than an H100. Two
caveats worth stating in any write-up: FP8 kernel support for this hybrid
architecture is worth verifying on a short `dev` job before trusting a long run,
and a quantized serve weakens the re-runnability claim that open weights are
kept for — the annotations came from a compressed copy, not from the published
weights. `--tensor-parallel-size 2` across two cards is the alternative that
does not compress anything.

### Structured output

`generate_structured()` sends a `response_format` with a JSON schema, which vLLM
implements through its guided-decoding backend. No flag is needed.

The adapter keeps the same tolerant recovery path it uses for OpenRouter: if the
model fences its JSON or writes a sentence around it, the document is extracted
and validated anyway. That means a server whose guided decoding is *not* actually
constraining generation can look like it is working — the recovery hides it. This
is why `probe_reasoning.py` validates every response and counts how many needed
the fallback: on a healthy server that column should read `-`, and a rising fence
count is the signal that the schema is not reaching the sampler.

### Reasoning levels

vLLM passes reasoning depth into the model's chat template, so the adapter sends
`chat_template_kwargs: {"reasoning_effort": ...}` rather than the `reasoning`
block OpenRouter takes. Qwen3.8 accepts `low`, `medium` and `xhigh`, defaulting
to `xhigh`; the registry asks for `low` so an unconfigured bulk run does not
reason as hard as it can on shared hardware. `--reasoning-parser qwen3` in the
sbatch script is what keeps the `<think>` block out of the answer.

Two response-shape details, measured on vLLM 0.27.1 on 2026-08-16:

- the reasoning comes back under **`reasoning`**, and `reasoning_content` is
  absent entirely — other servers and OpenRouter use the latter, so anything
  reading it should try both;
- `completion_tokens_details` is null, so `reasoning_tokens` is unavailable on
  this route. Measure reasoning *length* instead, which is what
  `probe_reasoning.py` reports.

### Measured on Festus, 2026-08-16

First run: Qwen3.8-27B bf16, 2× L40 with `--tensor-parallel-size 2`, one
francophone article, two calls per level.

| Level | Median s | Completion tokens | Reasoning chars |
|---|---|---|---|
| `low` | 28.5 | 714 | 2,370 |
| `medium` | 37.0 | 947 | 3,663 |
| `xhigh` | 89.1 | 2,299 | 8,249 |

**The ladder is real.** Reasoning length grows ~3.5× from `low` to `xhigh` and
the middle rung sits cleanly between the two — which is precisely what Gemma 4
failed to do through OpenRouter, where `medium` and `high` were
indistinguishable. On this route a request for `medium` gets `medium`.

Guided decoding held: no response needed unfencing, so the schema really was
constraining generation rather than the recovery path papering over it.

Two caveats for anyone reading this as a green light. At `xhigh` one call in two
exceeded the 300 s default request timeout on L40s — an H100 or a raised
`request_timeout_seconds` is wanted before bulk work at that depth. And both
`medium` calls returned a *schema-valid but rule-invalid* answer (a null
`subjectivite_score` alongside a non-null centralité, which the Pydantic
validator rejects): guided decoding enforces shape, never cross-field logic.
Two calls is far too small a sample to conclude anything, but it is the kind of
thing the pilot has to count rather than assume away.

### Adding a different model

Add it to `MODEL_REGISTRY` in `common/llm_registry.py` with
`PROVIDER_SELFHOSTED`, where `model` is the name the server reports from
`/v1/models` — the repo id it was launched with, unless you passed
`--served-model-name`. Set `default_temperature` from the vendor's published
recipe for the mode you will run it in, never from a neighbouring entry; the
`supported_reasoning_efforts` tuple should list only levels the served model
really accepts. Then `SERVE_MODEL=<repo-id> sbatch serving/vllm_serve.sbatch`.
