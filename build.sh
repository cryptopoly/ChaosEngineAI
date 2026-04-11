#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detect OS ────────────────────────────────────────────
case "$(uname -s)" in
  Darwin)             PLATFORM=darwin  ;;
  Linux)              PLATFORM=linux   ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM=windows ;;
  *)  echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac
echo "==> Platform: $PLATFORM"

# ── Python venv ──────────────────────────────────────────
if [ ! -d .venv ]; then
  echo "==> Creating Python venv..."
  python3 -m venv .venv
fi

echo "==> Installing Python dependencies..."
case "$PLATFORM" in
  darwin)
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install mlx mlx-lm gguf fastapi psutil uvicorn pypdf python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  linux)
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install fastapi psutil uvicorn pypdf python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  windows)
    .venv/Scripts/pip install --upgrade pip -q
    .venv/Scripts/pip install fastapi psutil uvicorn pypdf python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
    ;;
esac

# ── npm dependencies ─────────────────────────────────────
echo "==> Installing npm dependencies..."
npm ci --silent

# ── Build ────────────────────────────────────────────────
case "$PLATFORM" in
  darwin)  BUNDLES="app,dmg"     ;;
  linux)   BUNDLES="appimage,deb" ;;
  windows) BUNDLES="nsis"         ;;
esac

echo "==> Building Tauri app (bundles: $BUNDLES)..."
npx tauri build --bundles "$BUNDLES"

echo ""
echo "==> Build complete!"
echo "    Artifacts: src-tauri/target/release/bundle/"
