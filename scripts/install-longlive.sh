#!/usr/bin/env bash
# Install LongLive real-time long video generator (NVlabs/LongLive).
#
# LongLive extends Wan 2.1 T2V 1.3B with causal long-video generation —
# up to 240s @ ~20 FPS on a single H100, ICLR 2026, Apache 2.0. It ships
# as a torchrun-launched Python pipeline (not a pip package), so we
# clone it into an isolated checkout + venv to avoid polluting the host
# Python with CUDA-specific deps (diffusers==0.31.0, torchao, flash-attn
# optional, etc.).
#
# CUDA only — LongLive needs flash attention and bfloat16 on CUDA. The
# install script bails out on Darwin.
#
# Usage:  ./scripts/install-longlive.sh
#
# Layout after install:
#   ~/.chaosengine/longlive/
#     repo/                          clone of NVlabs/LongLive
#     venv/                          isolated Python venv
#     longlive_models/models/        base + lora checkpoints (HF download)
#     ready.marker                   written on successful install

set -euo pipefail

LONGLIVE_REPO="https://github.com/NVlabs/LongLive.git"
LONGLIVE_REF="${LONGLIVE_REF:-main}"
LONGLIVE_ROOT="${CHAOSENGINE_LONGLIVE_ROOT:-$HOME/.chaosengine/longlive}"
LONGLIVE_HF_REPO="Efficient-Large-Model/LongLive-1.3B"
WAN_HF_REPO="Wan-AI/Wan2.1-T2V-1.3B"
PYTHON_BIN="${CHAOSENGINE_PYTHON:-python3}"

case "$(uname -s)" in
  Darwin)
    echo "error: LongLive requires CUDA; macOS is not supported." >&2
    echo "       Install on a Windows or Linux machine with a recent NVIDIA GPU." >&2
    exit 2
    ;;
esac

if ! command -v git &>/dev/null; then
  echo "error: git not found on PATH." >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "error: python not found (set CHAOSENGINE_PYTHON)." >&2
  exit 1
fi

mkdir -p "$LONGLIVE_ROOT"
cd "$LONGLIVE_ROOT"

REPO_DIR="$LONGLIVE_ROOT/repo"
VENV_DIR="$LONGLIVE_ROOT/venv"
WEIGHTS_DIR="$LONGLIVE_ROOT/longlive_models/models"
WAN_DIR="$LONGLIVE_ROOT/wan_base"

echo "==> LongLive install target: $LONGLIVE_ROOT"

# 1. Clone or update repo
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "==> updating existing checkout"
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" checkout "$LONGLIVE_REF"
  git -C "$REPO_DIR" reset --hard "origin/$LONGLIVE_REF"
else
  echo "==> cloning $LONGLIVE_REPO ($LONGLIVE_REF)"
  git clone --depth 1 --branch "$LONGLIVE_REF" "$LONGLIVE_REPO" "$REPO_DIR"
fi

# 2. Build isolated venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> upgrading pip"
pip install --upgrade pip setuptools wheel

echo "==> installing LongLive requirements"
pip install -r "$REPO_DIR/requirements.txt"

# flash-attn is optional but strongly recommended; install best-effort
if ! python -c "import flash_attn" 2>/dev/null; then
  echo "==> installing flash-attn (optional, may take several minutes)"
  pip install flash-attn --no-build-isolation || {
    echo "warning: flash-attn install failed — LongLive will run but slower" >&2
  }
fi

pip install huggingface-hub

# 3. Download LongLive LoRA + generator checkpoints
mkdir -p "$WEIGHTS_DIR"
echo "==> downloading LongLive checkpoints from $LONGLIVE_HF_REPO"
python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${LONGLIVE_HF_REPO}",
    local_dir="${WEIGHTS_DIR}",
    local_dir_use_symlinks=False,
)
PY

# 4. Download Wan 2.1 T2V 1.3B base weights (needed by WanDiffusionWrapper)
mkdir -p "$WAN_DIR"
echo "==> downloading Wan 2.1 T2V 1.3B base from $WAN_HF_REPO"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${WAN_HF_REPO}",
    local_dir="${WAN_DIR}",
    local_dir_use_symlinks=False,
)
PY

# 5. Record install info
{
  echo "repo_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "repo_ref=$LONGLIVE_REF"
  echo "installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "longlive_root=$LONGLIVE_ROOT"
} > "$LONGLIVE_ROOT/ready.marker"

echo
echo "==> LongLive install complete"
echo "Repo:     $REPO_DIR"
echo "Venv:     $VENV_DIR"
echo "Weights:  $WEIGHTS_DIR"
echo "Wan base: $WAN_DIR"
echo "Marker:   $LONGLIVE_ROOT/ready.marker"
echo
echo "Set CHAOSENGINE_LONGLIVE_ROOT=$LONGLIVE_ROOT (or use the default path)."
