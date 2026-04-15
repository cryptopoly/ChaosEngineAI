#!/usr/bin/env bash
# Build llama-server-turbo from the johndpope/llama-cpp-turboquant fork.
#
# This fork extends standard llama-server with additional KV cache
# quantisation types (iso3/4, planar3/4, turbo2/3/4) required by the
# RotorQuant and TurboQuant cache strategies, while remaining fully
# compatible with all standard cache types.
#
# The binary is installed as ``llama-server-turbo`` (alongside the
# standard ``llama-server``) into ~/.chaosengine/bin/ so that
# ChaosEngineAI can discover and use it automatically at runtime.
#
# Usage:
#   ./scripts/build-llama-turbo.sh
#
# Environment variables:
#   LLAMA_TURBO_DIR      Source checkout dir  (default: /tmp/llama-cpp-turboquant)
#   CHAOSENGINE_BIN_DIR  Install destination  (default: ~/.chaosengine/bin)
#   LLAMA_TURBO_BRANCH   Git branch to build  (default: planarquant-kv-cache)
#   LLAMA_TURBO_JOBS     Parallel build jobs  (default: $(nproc) or sysctl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TURBO_REPO="https://github.com/TheTom/llama-cpp-turboquant.git"
TURBO_BRANCH="${LLAMA_TURBO_BRANCH:-feature/turboquant-kv-cache}"
TURBO_DIR="${LLAMA_TURBO_DIR:-/tmp/llama-cpp-turboquant}"
INSTALL_DIR="${CHAOSENGINE_BIN_DIR:-$HOME/.chaosengine/bin}"

# Detect parallel jobs
if command -v nproc &>/dev/null; then
  JOBS="${LLAMA_TURBO_JOBS:-$(nproc)}"
elif command -v sysctl &>/dev/null; then
  JOBS="${LLAMA_TURBO_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
else
  JOBS="${LLAMA_TURBO_JOBS:-4}"
fi

echo "==> llama-server-turbo builder"
echo "    repo:     $TURBO_REPO"
echo "    branch:   $TURBO_BRANCH"
echo "    source:   $TURBO_DIR"
echo "    install:  $INSTALL_DIR"
echo "    jobs:     $JOBS"
echo

# Clone or update the source checkout
if [[ -d "$TURBO_DIR/.git" ]]; then
  echo "==> updating existing checkout"
  cd "$TURBO_DIR"
  git fetch --all --prune
  git checkout "$TURBO_BRANCH"
  git reset --hard "origin/$TURBO_BRANCH"
else
  echo "==> cloning $TURBO_REPO (branch: $TURBO_BRANCH)"
  git clone --branch "$TURBO_BRANCH" "$TURBO_REPO" "$TURBO_DIR"
  cd "$TURBO_DIR"
fi

# Platform-specific CMake flags
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
case "$(uname -s)" in
  Darwin)
    CMAKE_FLAGS+=(-DGGML_METAL=ON)
    ;;
  Linux)
    # Enable CUDA if nvcc is available
    if command -v nvcc &>/dev/null; then
      CMAKE_FLAGS+=(-DGGML_CUDA=ON)
    fi
    ;;
esac

echo "==> cmake configure"
cmake -B build "${CMAKE_FLAGS[@]}"

echo "==> building llama-server + llama-cli"
cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli

# Install into the ChaosEngineAI-managed bin directory
echo "==> installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp build/bin/llama-server "$INSTALL_DIR/llama-server-turbo"
cp build/bin/llama-cli "$INSTALL_DIR/llama-cli-turbo"
chmod +x "$INSTALL_DIR/llama-server-turbo" "$INSTALL_DIR/llama-cli-turbo"

# Write version tracking file so ChaosEngineAI can detect updates.
# Line 1: commit hash of the built revision
# Line 2: branch name used for the build
VERSION_FILE="$INSTALL_DIR/llama-server-turbo.version"
{
  git rev-parse HEAD
  echo "$TURBO_BRANCH"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} > "$VERSION_FILE"
echo "==> version tracked in $VERSION_FILE"

echo
echo "==> build complete"
echo "llama-server-turbo installed to $INSTALL_DIR/llama-server-turbo"
echo "ChaosEngineAI will auto-detect it on next model load."
echo "Restart the app if it is currently running."
