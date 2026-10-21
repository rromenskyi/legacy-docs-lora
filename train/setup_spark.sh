#!/usr/bin/env bash
# Set up an Axolotl venv for the NemotronH QLoRA recipe, on a DGX Spark
# (GB10, Grace CPU = aarch64, unlike the x86_64 RunPod pod this install
# order was originally validated on in quant-ternary/nemotron-extreme-quant --
# see that project's poc/setup_pod.sh for the full history this is adapted
# from). The wheel-cache shortcuts that script used for flash-attn/
# causal-conv1d/mamba_ssm are x86_64-only and WILL miss here -- this
# always source-builds. Untested on real Spark hardware; written from the
# documented install order/gotchas, not run.
#
# No sudo, nothing outside $HOME -- this is borrowed hardware
# (see quant-ternary memory: "the Spark is a friend's machine").
#
# Usage: ssh dgx 'bash -s' < setup_spark.sh
set -euo pipefail

VENV="${VENV:-$HOME/legacy-docs-lora-venv}"

if [ -x "$VENV/bin/python3" ] && "$VENV/bin/python3" -c 'import axolotl, flash_attn, cut_cross_entropy, causal_conv1d, mamba_ssm' 2>/dev/null; then
  echo "venv already set up at $VENV, skipping"
  exit 0
fi

python3 -m venv "$VENV"
# Activate, don't call binaries by full path -- torch's BuildExtension
# looks for the ninja EXECUTABLE via PATH, not just the pip package; an
# unactivated venv leaves venv/bin off PATH and silently falls back to a
# slow single-file build (bit the original RunPod setup once).
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install packaging ninja wheel setuptools

# Axolotl first, let it pick whatever torch it wants -- pre-pinning torch
# here is pointless, axolotl's own resolution replaces it anyway. Fix
# torch to match the real system CUDA toolkit only AFTER this.
pip install axolotl

CUDA_BIN=$(dirname "$(readlink -f /usr/local/cuda/bin/nvcc 2>/dev/null || echo /usr/local/cuda-12.8/bin/nvcc)")
CUDA_HOME=$(dirname "$CUDA_BIN")
CUDA_VER=$("$CUDA_BIN/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
TORCH_CU_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
echo "--- system CUDA toolkit: $CUDA_VER ($TORCH_CU_TAG) at $CUDA_HOME ---"
pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/$TORCH_CU_TAG"
EXPECTED_TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')

# Restrict the build to this box's actual GPU arch -- flash-attn's
# setup.py otherwise builds 4 archs unconditionally (FLASH_ATTN_CUDA_ARCHS
# defaults to "80;90;100;120"), multiplying both build time and the
# per-compile-job memory footprint (an unrestricted build OOM-killed even
# at MAX_JOBS=6 on the original pod).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export TORCH_CUDA_ARCH_LIST="$ARCH"
export FLASH_ATTN_CUDA_ARCHS="$(echo "$ARCH" | tr -d '.')"

MEM_LIMIT_BYTES=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
if [[ -z "$MEM_LIMIT_BYTES" || "$MEM_LIMIT_BYTES" == "max" ]]; then
  MEM_LIMIT_BYTES=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || free -b | awk '/^Mem:/{print $2}')
fi
MEM_LIMIT_GB=$(( MEM_LIMIT_BYTES / 1024 / 1024 / 1024 ))
CPU_COUNT=$(nproc)
PER_JOB_GB=10
MAX_JOBS_BY_MEM=$(( MEM_LIMIT_GB / PER_JOB_GB ))
export MAX_JOBS=$(( CPU_COUNT < MAX_JOBS_BY_MEM ? CPU_COUNT : MAX_JOBS_BY_MEM ))
[[ $MAX_JOBS -lt 1 ]] && export MAX_JOBS=1
echo "--- building flash-attn (arch=$FLASH_ATTN_CUDA_ARCHS, mem=${MEM_LIMIT_GB}GB, cpu=${CPU_COUNT}, MAX_JOBS=$MAX_JOBS) ---"
export PATH="$CUDA_HOME:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/../lib64:${LD_LIBRARY_PATH:-}"
pip install flash-attn --no-build-isolation

# Axolotl's cut_cross_entropy plugin needs this fork; stock PyPI
# cut-cross-entropy lacks transformers support and fails at model-load.
pip uninstall -y cut-cross-entropy 2>/dev/null || true
pip install 'cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa522'

# Without these, NemotronH's Mamba2 mixer falls back to a reference
# PyTorch chunk-scan that tried to allocate a single 45GB tensor during
# LoRA backward on the original pod. --no-deps on mamba_ssm is load-
# bearing: its own dependency resolution was observed to silently
# force-upgrade torch to an incompatible CUDA stack otherwise.
pip install --no-build-isolation causal-conv1d
pip install --no-build-isolation --no-deps mamba_ssm

ACTUAL_TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
if [[ "$ACTUAL_TORCH_VERSION" != "$EXPECTED_TORCH_VERSION" ]]; then
  echo "ERROR: torch changed from $EXPECTED_TORCH_VERSION to $ACTUAL_TORCH_VERSION during mamba_ssm install" >&2
  exit 1
fi

echo "=== venv ready at $VENV ==="
