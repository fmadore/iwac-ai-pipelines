#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Unattended: serve the model, annotate a prepared corpus against it, stop.
#
#     sbatch serving/annotate_job.sbatch
#
# Use this instead of vllm_serve.sbatch when nobody will be at the keyboard —
# an overnight slot, or any job whose start time the scheduler decides. Nothing
# is tunnelled and nothing is typed: the job produces a JSONL file you collect
# afterwards.
#
# Prepare the input on the machine that HAS the credentials, and copy it up:
#
#     scp pilot_input.json <user>@<login-host>:~/iwac-ai-pipelines/serving/work/
#
# Set ANNOTATE_INPUT / ANNOTATE_OUTPUT to override the defaults below.
#
# The server binds 127.0.0.1 here, not 0.0.0.0: only this job talks to it, so
# there is no reason to expose the port to the rest of the cluster, and no
# API key to manage.
# ---------------------------------------------------------------------------
#SBATCH --job-name=vllm-annotate
#SBATCH --partition=GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=serving/logs/annotate-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO/serving/env.sh"

WORK="${ANNOTATE_WORK:-$REPO/serving/work}"
INPUT="${ANNOTATE_INPUT:-$WORK/pilot_input.json}"
OUTPUT="${ANNOTATE_OUTPUT:-$WORK/annotations-${SLURM_JOB_ID:-local}.jsonl}"
PORT="${SERVE_PORT:-8000}"

[[ -f "$INPUT" ]] || { echo "ERROR: no input at $INPUT" >&2; exit 1; }

load_python
activate_venv
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "==> node $(hostname), job ${SLURM_JOB_ID:-none}"
echo "==> input  $INPUT"
echo "==> output $OUTPUT"

# Start the server in the background and make sure it dies with this script —
# including on scancel, which sends TERM. An orphaned vLLM would hold the GPU
# for the rest of the allocation.
vllm serve "$SERVE_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$SERVE_MAX_LEN" \
  --max-num-seqs "$SERVE_MAX_SEQS" \
  --reasoning-parser qwen3 \
  ${SERVE_EXTRA_ARGS:-} &
SERVER_PID=$!
trap 'echo "==> stopping server"; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT INT TERM

echo "==> waiting for the server to load the weights"
for _ in $(seq 1 120); do   # 120 × 15s = 30 min ceiling
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "==> server ready after ${SECONDS}s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: server exited before becoming ready — see above" >&2
    exit 1
  fi
  sleep 15
done
curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  echo "ERROR: server never became ready" >&2; exit 1; }

export SELFHOSTED_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
unset SELFHOSTED_LLM_API_KEY   # started without --api-key; the client sends EMPTY

python "$REPO/serving/annotate_offline.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  ${ANNOTATE_EXTRA_ARGS:-}

echo "==> annotations at $OUTPUT"
wc -l "$OUTPUT" || true

# /workdir is purged on a schedule and is not backed up; copy the result to
# /home, which is. Best effort — never fail the job over the archive step.
ARCHIVE="${ANNOTATE_ARCHIVE:-$HOME/iwac-annotations}"
if [[ "$ARCHIVE" != "off" ]]; then
  mkdir -p "$ARCHIVE" 2>/dev/null && cp "$OUTPUT" "$ARCHIVE"/ 2>/dev/null \
    && echo "==> archived to $ARCHIVE" || echo "WARN: archive step failed" >&2
fi
