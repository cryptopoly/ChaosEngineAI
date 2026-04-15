#!/usr/bin/env bash
# Update llama-server-turbo to the latest commit and rebuild.
#
# This is the companion to build-llama-turbo.sh — it fetches the latest
# changes from the TurboQuant fork and rebuilds the binary in-place.
#
# Usage:  ./scripts/update-llama-turbo.sh
#
# Override the source dir with LLAMA_TURBO_DIR if your checkout lives
# somewhere other than /tmp/llama-cpp-turboquant.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TURBO_BRANCH="${LLAMA_TURBO_BRANCH:-feature/planarquant-kv-cache}"
TURBO_DIR="${LLAMA_TURBO_DIR:-/tmp/llama-cpp-turboquant}"
INSTALL_DIR="${CHAOSENGINE_BIN_DIR:-$HOME/.chaosengine/bin}"
VERSION_FILE="$INSTALL_DIR/llama-server-turbo.version"

# Detect parallel jobs
if command -v nproc &>/dev/null; then
  JOBS="${LLAMA_TURBO_JOBS:-$(nproc)}"
elif command -v sysctl &>/dev/null; then
  JOBS="${LLAMA_TURBO_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
else
  JOBS="${LLAMA_TURBO_JOBS:-4}"
fi

# If no checkout exists yet, delegate to the full build script.
if [[ ! -d "$TURBO_DIR/.git" ]]; then
  echo "No existing checkout at $TURBO_DIR — running full build instead."
  exec "$SCRIPT_DIR/build-llama-turbo.sh"
fi

cd "$TURBO_DIR"

# Show current version
if [[ -f "$VERSION_FILE" ]]; then
  CURRENT_COMMIT=$(head -1 "$VERSION_FILE")
  echo "Current installed commit: $CURRENT_COMMIT"
else
  CURRENT_COMMIT=""
  echo "No version file found — will rebuild regardless."
fi

echo "==> fetching latest changes"
git fetch --all --prune

echo "==> checking out $TURBO_BRANCH"
git checkout "$TURBO_BRANCH"

REMOTE_COMMIT=$(git rev-parse "origin/$TURBO_BRANCH")
echo "Remote HEAD: $REMOTE_COMMIT"

if [[ "$CURRENT_COMMIT" == "$REMOTE_COMMIT" ]]; then
  echo
  echo "Already up to date. No rebuild needed."
  exit 0
fi

echo "==> resetting to origin/$TURBO_BRANCH"
git reset --hard "origin/$TURBO_BRANCH"

# Platform-specific CMake flags
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
case "$(uname -s)" in
  Darwin)
    CMAKE_FLAGS+=(-DGGML_METAL=ON)
    ;;
  Linux)
    if command -v nvcc &>/dev/null; then
      CMAKE_FLAGS+=(-DGGML_CUDA=ON)
    fi
    ;;
esac

echo "==> cmake configure"
cmake -B build "${CMAKE_FLAGS[@]}"

echo "==> rebuilding llama-server + llama-cli"
cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli

echo "==> installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp build/bin/llama-server "$INSTALL_DIR/llama-server-turbo"
cp build/bin/llama-cli "$INSTALL_DIR/llama-cli-turbo"
chmod +x "$INSTALL_DIR/llama-server-turbo" "$INSTALL_DIR/llama-cli-turbo"

# Update version tracking
{
  git rev-parse HEAD
  echo "$TURBO_BRANCH"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} > "$VERSION_FILE"

echo
echo "==> update complete"
echo "Updated from ${CURRENT_COMMIT:0:12} to $(git rev-parse --short HEAD)"
echo "Restart ChaosEngineAI to pick up the new binary."
