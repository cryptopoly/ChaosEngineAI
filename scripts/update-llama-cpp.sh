#!/usr/bin/env bash
# Update the local llama.cpp checkout to the latest commit and rebuild
# llama-server. Required whenever llama.cpp adds support for a new
# model architecture your current build doesn't recognise (e.g. gemma4).
#
# Usage:  ./scripts/update-llama-cpp.sh
#
# Override the source dir with LLAMA_CPP_DIR if your checkout lives
# somewhere other than ../llama.cpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="${LLAMA_CPP_DIR:-$(cd "$REPO_ROOT/.." && pwd)/llama.cpp}"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "error: $LLAMA_DIR is not a git checkout of llama.cpp" >&2
  echo "Set LLAMA_CPP_DIR to the correct path and re-run." >&2
  exit 1
fi

echo "Updating llama.cpp at $LLAMA_DIR"
cd "$LLAMA_DIR"

echo "==> git fetch"
git fetch --all --tags --prune

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UPSTREAM="origin/$CURRENT_BRANCH"
if ! git rev-parse --verify "$UPSTREAM" >/dev/null 2>&1; then
  echo "error: no upstream branch $UPSTREAM — aborting." >&2
  exit 1
fi

# Warn loudly if there are uncommitted changes — then refuse unless FORCE=1.
if [[ -n "$(git status --porcelain)" ]]; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "error: uncommitted changes in $LLAMA_DIR. Commit/stash them or re-run with FORCE=1." >&2
    git status --short >&2
    exit 1
  fi
  echo "warning: uncommitted changes detected but FORCE=1, proceeding."
fi

echo "==> resetting $CURRENT_BRANCH to $UPSTREAM (handles force-pushes / divergence)"
git reset --hard "$UPSTREAM"

echo "==> configure (Metal ON)"
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release

echo "==> build llama-server + llama-cli"
cmake --build build --config Release -j --target llama-server llama-cli

echo
echo "==> build complete"
"$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -5

echo
echo "Restart ChaosEngineAI (Ctrl-C and re-run npm run tauri:dev) to pick"
echo "up the new llama-server binary."
