#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup, run on the cluster LOGIN node (not in a job):
#
#     bash serving/setup_env.sh
#
# Creates the vLLM venv and downloads the weights. Both are slow and both need
# the internet, which is why they happen here rather than inside the job —
# compute nodes are frequently offline, and the job then runs with
# HF_HUB_OFFLINE=1 against what this script left in the cache.
# ---------------------------------------------------------------------------
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# vLLM moves fast and breaks its own engine APIs between minor releases, so this
# is a floor rather than a range: Qwen3.8's hybrid architecture (Gated DeltaNet
# interleaved with Gated Attention) needs a build that knows about it, and
# 0.27.1 is the first stable release verified to serve it.
#
# Keep this venv SEPARATE from any other vLLM you have on the cluster. In
# particular, if you also run AM-Digital-Research-Environment/festus-transcribe,
# its 0.21.0 pin is deliberate and tuned against its own backends — upgrading it
# in place to satisfy this script would break that pipeline instead.
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.27.1}"

echo "==> Python module: $PYTHON_MODULE"
load_python
python3 --version

if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
  echo "==> creating venv at $VLLM_VENV"
  mkdir -p "$(dirname "$VLLM_VENV")"
  python3 -m venv "$VLLM_VENV"
fi
# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

echo "==> installing $VLLM_SPEC (this takes a while — it pulls torch)"
pip install --upgrade pip
pip install "$VLLM_SPEC"
python -c "import vllm; print('vllm', vllm.__version__)"

echo "==> prefetching $SERVE_MODEL into $HF_HOME"
mkdir -p "$HF_HOME"
# `hf download` ships with huggingface_hub, which vLLM already depends on.
# Gated repos would need `hf auth login` first; Qwen3.8 is Apache-2.0 and open,
# so no token is involved and none should be stored on a shared filesystem.
hf download "$SERVE_MODEL"

cat <<EOF

==> done.

    venv    $VLLM_VENV
    weights $HF_HOME
    model   $SERVE_MODEL

Next: submit the server and open a tunnel to it. See serving/README.md.

    sbatch serving/vllm_serve.sbatch
EOF
