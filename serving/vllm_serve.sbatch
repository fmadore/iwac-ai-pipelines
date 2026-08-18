#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one open-weights model over an OpenAI-compatible HTTP API, inside a
# Slurm job, for the pipelines in this repo to call.
#
#     SERVE_API_KEY=sk-... sbatch serving/vllm_serve.sbatch
#
# The #SBATCH lines below are read by Slurm BEFORE bash runs, so they cannot use
# variables. Override them on the command line instead — which is the expected
# way to run this anywhere but the cluster it was written on:
#
#     sbatch --partition=dev --gres=gpu:l40:2 --time=01:30:00 serving/vllm_serve.sbatch
#
# Check what your cluster actually offers before the first submit:
#
#     sinfo -o "%P %G %N"
#
# Defaults target a single H100 (80 GB), which fits Qwen3.8-27B in bf16 (~56 GB
# of weights) with room for the KV cache at the default 32k context.
# ---------------------------------------------------------------------------
#SBATCH --job-name=vllm-serve
#SBATCH --partition=GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=serving/logs/vllm-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO/serving/env.sh"

# Refuse to start without a token. The server binds 0.0.0.0 so the login node can
# reach it, which also means every other user on the cluster network can — a
# shared compute node is not a private one. `openssl rand -hex 24` is enough;
# prefix it `sk-` so common/log_redaction.py masks it if it ever reaches a log.
: "${SERVE_API_KEY:?set SERVE_API_KEY — see serving/README.md}"

load_python
activate_venv

# The weights were fetched on the login node by setup_env.sh. Compute nodes are
# often offline; say so explicitly rather than hanging on a DNS timeout.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cat <<EOF
==> serving on $(hostname):${SERVE_PORT}
    model    $SERVE_MODEL
    context  $SERVE_MAX_LEN
    venv     $VLLM_VENV
    job      ${SLURM_JOB_ID:-<none>}

Tunnel to it from your own machine with:

    ssh -L ${SERVE_PORT}:$(hostname):${SERVE_PORT} <user>@<login-host>

then set in .env:  SELFHOSTED_LLM_BASE_URL=http://localhost:${SERVE_PORT}/v1
EOF

# --reasoning-parser keeps the <think> block in `reasoning_content` instead of
# letting it run into `content`. Without it a thinking model's chain of thought
# arrives as part of the answer, and every structured call has to fish JSON out
# of an essay. `qwen3` is the right parser for the Qwen3.x family.
#
# Structured output needs no flag: vLLM implements `response_format` with a
# json_schema through its guided-decoding backend, which is what
# generate_structured() sends.
exec vllm serve "$SERVE_MODEL" \
  --host 0.0.0.0 \
  --port "$SERVE_PORT" \
  --api-key "$SERVE_API_KEY" \
  --max-model-len "$SERVE_MAX_LEN" \
  --max-num-seqs "$SERVE_MAX_SEQS" \
  --reasoning-parser qwen3 \
  ${SERVE_EXTRA_ARGS}
