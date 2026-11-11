#!/usr/bin/env bash
# ChaosEngineAI Linux GPU check (DGX Spark, GH200, any Linux + NVIDIA machine).
#
# Collects what we need to know from a tester in one report:
#   1. the machine: OS, CPU arch, NVIDIA driver, GPU, memory
#   2. the app: backend health and the CUDA llama-server status (when the app is running)
#   3. a short llama.cpp benchmark on the GPU with a 1.1 GB model
#
# Nothing is uploaded. The report is written to ~/chaosengine-gpu-report.md;
# paste it into the GitHub issue. Skip the benchmark with --no-bench.
#
# The benchmark uses the CUDA llama-server build the app installs from
# Settings > Accelerators (~/.chaosengine/bin/llama-cuda/<tag>/), and downloads
# LiquidAI/LFM3-1.7B-GGUF (Q4_K_M, 1.1 GB) once into ~/.chaosengine/gpu-check/.

set -u

REPORT="${HOME}/chaosengine-gpu-report.md"
PORT="${CHAOSENGINE_PORT:-8876}"
TAG="${CHAOSENGINE_LLAMA_CPP_TAG:-b12292}"
CUDA_DIR="${HOME}/.chaosengine/bin/llama-cuda/${TAG}"
MODEL_DIR="${HOME}/.chaosengine/gpu-check"
MODEL="${MODEL_DIR}/LFM3-1.7B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf"
BENCH=1
[ "${1:-}" = "--no-bench" ] && BENCH=0

section() { printf '\n## %s\n\n' "$1" >> "$REPORT"; }
block() {
  # Run a command and put its output (or the failure) in a code block.
  printf '```\n$ %s\n' "$*" >> "$REPORT"
  if ! "$@" >> "$REPORT" 2>&1; then
    printf '(exit code %s)\n' "$?" >> "$REPORT"
  fi
  printf '```\n' >> "$REPORT"
}
have() { command -v "$1" >/dev/null 2>&1; }

printf '# ChaosEngineAI Linux GPU check\n\nGenerated %s\n' "$(date -u '+%Y-%m-%d %H:%M UTC')" > "$REPORT"
echo "Writing ${REPORT}"

section "Machine"
block uname -srm
[ -r /etc/os-release ] && block grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
block ldd --version
have free && block free -g
if have nvidia-smi; then
  block nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv
  block nvidia-smi -L
else
  printf 'nvidia-smi not found: the NVIDIA driver is not installed or not on PATH.\n' >> "$REPORT"
fi
have nvcc && block nvcc --version

section "App"
if have curl && curl -fsS -m 5 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
  block curl -fsS -m 10 "http://127.0.0.1:${PORT}/api/setup/llama-cuda-status"
  block curl -fsS -m 10 "http://127.0.0.1:${PORT}/api/health"
else
  printf 'The app is not running on port %s (start it to include its status).\n' "$PORT" >> "$REPORT"
fi

section "CUDA llama-server"
if [ -x "${CUDA_DIR}/llama-server" ]; then
  block "${CUDA_DIR}/llama-server" --version
  block "${CUDA_DIR}/llama-server" --list-devices
else
  printf 'Not installed at %s. Install it from Settings > Accelerators, then run this again.\n' "$CUDA_DIR" >> "$REPORT"
  BENCH=0
fi

if [ "$BENCH" = 1 ]; then
  section "Benchmark (LFM3 1.7B Q4_K_M, llama-bench; the backend column shows CUDA or CPU)"
  if [ ! -s "$MODEL" ]; then
    echo "Downloading the 1.1 GB test model (once)..."
    mkdir -p "$MODEL_DIR"
    if ! curl -fL --retry 3 --progress-bar -o "${MODEL}.part" "$MODEL_URL"; then
      printf 'Model download failed.\n' >> "$REPORT"
      BENCH=0
    else
      mv "${MODEL}.part" "$MODEL"
    fi
  fi
  if [ "$BENCH" = 1 ]; then
    echo "Running the benchmark..."
    block "${CUDA_DIR}/llama-bench" -m "$MODEL" -ngl 99 -p 512 -n 128 -r 3
  fi
fi

printf '\n---\nPaste this file into the GitHub issue. It holds no personal data beyond the machine details above.\n' >> "$REPORT"
echo "Done: ${REPORT}"
