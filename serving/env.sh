#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared settings for the serving scripts. Sourced by setup_env.sh and
# vllm_serve.sbatch.
#
# Every cluster-specific value lives HERE, in one place, and every one of them
# can be overridden from the environment without editing the file:
#
#     SERVE_MODEL=Qwen/Qwen3.8-27B-FP8 sbatch serving/vllm_serve.sbatch
#
# That is what keeps this directory publishable: nothing below names a machine,
# an account, or a secret. The defaults are the University of Bayreuth "Festus"
# cluster because that is where it was written and verified; they are ordinary
# `module avail` / `sinfo` facts, not configuration you inherit. Check yours.
# ---------------------------------------------------------------------------

# Repo root. Under `sbatch` the submit script sets REPO from $SLURM_SUBMIT_DIR
# first (the spooled copy of the script cannot locate the repo itself). Run
# directly on the login node, derive it from this file's own location.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Load $REPO/.env (KEY=VALUE). Variables already set in the environment win, so
# `SERVE_API_KEY=... sbatch ...` still overrides the file. $USER/$HOME expand.
#
# On the cluster this file only needs the SERVE_* values; the API keys the
# pipelines use stay on your own machine. It is gitignored either way.
load_dotenv() {
  local f="$REPO/.env" line key val
  [[ -f "$f" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"; val="${line#*=}"
    key="${key//[[:space:]]/}"
    [[ -z "$key" ]] && continue
    val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    val="${val//\$\{USER\}/$USER}"; val="${val//\$USER/$USER}"
    val="${val//\$\{HOME\}/$HOME}"; val="${val//\$HOME/$HOME}"
    [[ -n "${!key+x}" ]] && continue   # don't clobber an already-set variable
    export "$key=$val"
  done < "$f"
}
load_dotenv

# ---------------------------------------------------------------------------
# What to serve
# ---------------------------------------------------------------------------

# The model to serve, as a Hugging Face repo id. Whatever is set here is also
# the name the server reports from /v1/models, which is what
# `common/llm_registry.py` records as this model's id — change one and the other
# stops matching. Serving under a different name means passing
# --served-model-name and editing SELFHOSTED_QWEN38_MODEL to agree.
SERVE_MODEL="${SERVE_MODEL:-Qwen/Qwen3.8-27B}"

SERVE_PORT="${SERVE_PORT:-8000}"

# Context window. 32k is far above what this repo's text stages need — the
# sentiment prompt plus a newspaper article is a few thousand tokens — and the
# memory it does not reserve for KV cache is memory the 27B weights can use.
# Raise it for longer documents, and expect to lower --gpu-memory-utilization or
# move to a bigger card if you do.
SERVE_MAX_LEN="${SERVE_MAX_LEN:-32768}"

# Anything else to hand `vllm serve`, e.g. --gpu-memory-utilization 0.92,
# --tensor-parallel-size 2, --quantization fp8.
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

# Python venv created by setup_env.sh. Deliberately NOT under $REPO: a cluster
# home directory is typically small and backed up, and a vLLM install is neither
# small nor worth backing up. /workdir on Festus is 3 TB.
VLLM_VENV="${VLLM_VENV:-/workdir/$USER/vllm-serve/.venv}"

# Hugging Face cache. Same reasoning — 27B of weights does not belong in /home.
export HF_HOME="${HF_HOME:-/workdir/$USER/hf_cache}"

# Lmod module providing Python. The exact version string differs per cluster —
# check with:  module avail python
PYTHON_MODULE="${SERVE_PYTHON_MODULE:-python/3.12.4}"

# ---------------------------------------------------------------------------
# Cluster plumbing
# ---------------------------------------------------------------------------

# Run `module load` safely under `set -euo pipefail`. Lmod's bash init
# references $LD_LIBRARY_PATH; if it is unset (as on a freshly powered-up node)
# `set -u` makes `module` abort with "LD_LIBRARY_PATH: unbound variable" — which
# silently leaves the module unloaded, and the venv interpreter then cannot find
# its shared libraries. Bind the variable and relax `set -u` only around the
# call, then restore it.
_module_load() {
  command -v module >/dev/null 2>&1 || return 0
  local had_u=0; case "$-" in *u*) had_u=1;; esac
  set +u
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  module load "$@"
  local rc=$?
  [ "$had_u" -eq 1 ] && set -u
  return $rc
}

# A no-op off-cluster, e.g. when testing this on a workstation with its own GPU.
load_python() { _module_load "$PYTHON_MODULE"; }

activate_venv() {
  if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
    echo "ERROR: no venv at $VLLM_VENV — run serving/setup_env.sh on the login node first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
}

# vLLM's FlashInfer sampler JIT-compiles a CUDA kernel, which needs nvcc — absent
# from a plain pip venv, and the cluster CUDA module often does not match torch's.
# Use the native sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# FP8 models otherwise demand DeepSeek's `deep_gemm`, which also needs nvcc.
# Force vLLM's built-in FP8 path.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
