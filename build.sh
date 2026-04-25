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
# vendor/ChaosEngine declares `license = "Apache-2.0"` per PEP 639. Setuptools
# < 77 rejects the string form, so bump it across every platform before the
# vendor install in stage-runtime.mjs runs.
case "$PLATFORM" in
  darwin)
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install --upgrade "setuptools>=77,<82" wheel -q
    .venv/bin/pip install mlx mlx-lm gguf fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  linux)
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install --upgrade "setuptools>=77,<82" wheel -q
    .venv/bin/pip install fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  windows)
    .venv/Scripts/pip install --upgrade pip -q
    .venv/Scripts/pip install --upgrade "setuptools>=77,<82" wheel -q
    .venv/Scripts/pip install fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
    ;;
esac

# ── Optional GPU bundle ──────────────────────────────────
# By default the installer ships CHAT-ONLY (no torch, no diffusers). Users
# click "Install GPU support" inside the app, which installs CUDA torch +
# diffusers into ~/.chaosengine/extras/ at runtime. Set
# CHAOSENGINE_BUNDLE_GPU=1 to include the GPU stack in the installer
# itself (adds ~1.3 GB, only useful for air-gapped deployments).
if [ "${CHAOSENGINE_BUNDLE_GPU:-}" = "1" ]; then
  echo "==> CHAOSENGINE_BUNDLE_GPU=1 -- bundling [images] extras"
  case "$PLATFORM" in
    darwin)  .venv/bin/pip install -q -e ".[desktop,images]" ;;
    linux)   .venv/bin/pip install -q -e ".[desktop,images]" ;;
    windows) .venv/Scripts/pip install -q -e ".[desktop,images]" ;;
  esac
fi

# ── npm dependencies ─────────────────────────────────────
echo "==> Installing npm dependencies..."
npm ci --silent

# ── llama.cpp pre-flight ─────────────────────────────────
# Release-mode stage-runtime.mjs throws if it can't locate a llama.cpp
# build dir AND the prebuilt-download fallback fails (e.g. macos-arm64
# is not in the latest ggml-org release assets). Mirror the Windows
# preflight: if no local build is found, allow the installer to ship
# without llama-server so the Tauri bundle step still produces a dmg.
# Users install llama-server later via the Setup page.
#
# Opt-out: CHAOSENGINE_REQUIRE_LLAMA=1 keeps the hard error (useful for
# CI that expects a fully self-contained installer).
LLAMA_BIN_DIR="${CHAOSENGINE_LLAMA_BIN_DIR:-$(dirname "$SCRIPT_DIR")/llama.cpp/build/bin}"
LLAMA_SERVER_BIN="$LLAMA_BIN_DIR/llama-server"
case "$PLATFORM" in
  windows) LLAMA_SERVER_BIN="$LLAMA_BIN_DIR/llama-server.exe" ;;
esac
if [ ! -x "$LLAMA_SERVER_BIN" ] && [ ! -f "$LLAMA_SERVER_BIN" ]; then
  if [ "${CHAOSENGINE_REQUIRE_LLAMA:-}" = "1" ]; then
    echo ""
    echo "==> ERROR: llama-server not found at $LLAMA_SERVER_BIN"
    echo "    CHAOSENGINE_REQUIRE_LLAMA=1 -- build llama.cpp or unset the flag."
    exit 1
  fi
  echo "==> llama-server not found locally at $LLAMA_SERVER_BIN"
  echo "    stage-runtime will try an auto-download; if that also fails"
  echo "    (macos-arm64 is not always in ggml-org's release assets),"
  echo "    the installer will ship without inference and users install"
  echo "    it via the Setup page."
  export CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA=1
fi

# ── Patch tauri.conf.json for local builds ───────────────
# Delegated to scripts/patch-tauri-conf.mjs so macOS and Windows share
# one source of truth. That helper wires beforeBundleCommand to
# ``npm run stage:runtime:release`` (NOT the dev variant — the dev
# variant writes mode=development into the manifest and ships an empty
# 6 MB installer with no Python runtime).
echo "==> Patching tauri.conf.json for local build..."
node scripts/patch-tauri-conf.mjs patch

# ── Build ────────────────────────────────────────────────
case "$PLATFORM" in
  darwin)  BUNDLES="app,dmg"     ;;
  linux)   BUNDLES="appimage,deb" ;;
  windows) BUNDLES="nsis"         ;;
esac

echo "==> Building Tauri app (bundles: $BUNDLES)..."
npx tauri build --bundles "$BUNDLES"

# Restore the committed tauri.conf.json — same helper as Windows.
node scripts/patch-tauri-conf.mjs restore 2>/dev/null || true

# ── Publish installers to /assets ────────────────────────
# The Tauri bundle tree is three directories deep and differs per target.
# Copy the shippable artifacts into a flat assets/ folder at the repo root
# so every build lands in the same place regardless of platform.
echo "==> Publishing artifacts to assets/..."
node scripts/publish-artifacts.mjs --bundles="$BUNDLES"

echo ""
echo "==> Build complete!"
echo "    Artifacts: assets/ (also in src-tauri/target/release/bundle/)"
