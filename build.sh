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

# â”€â”€ Release Python venv â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# A release build stages a Python environment into the installer, so it must
# be a clean, pinned one. It uses its own venv (.venv-release, created on
# demand, kept out of git) and installs with release-constraints.txt, the same
# as .github/workflows/release.yml. The dev .venv is never touched: it holds
# torch, diffusers, bitsandbytes and more, which pushed the Windows runtime
# archive to 3 GB (NSIS cannot map past 2 GB) and ships untested versions.
RELEASE_VENV="$SCRIPT_DIR/.venv-release"
case "$PLATFORM" in
  windows) RELEASE_PY="$RELEASE_VENV/Scripts/python.exe" ;;
  *)       RELEASE_PY="$RELEASE_VENV/bin/python3" ;;
esac

# A venv left over from a GPU-bundle build would drag torch into a chat-only
# installer; start over when the requested flavour changed.
if [ -x "$RELEASE_PY" ] && [ "${CHAOSENGINE_BUNDLE_GPU:-}" != "1" ] \
   && "$RELEASE_PY" -m pip show torch >/dev/null 2>&1; then
  echo "==> $RELEASE_VENV holds torch from a GPU-bundle build; recreating it..."
  rm -rf "$RELEASE_VENV"
fi

if [ ! -x "$RELEASE_PY" ]; then
  echo "==> Creating release Python venv (.venv-release)..."
  # release-constraints.txt is frozen on CPython 3.12; prefer that interpreter.
  if [ -n "${CHAOSENGINE_RELEASE_PYTHON:-}" ]; then
    BASE_PY=("$CHAOSENGINE_RELEASE_PYTHON")
  elif [ "$PLATFORM" = "windows" ] && command -v py >/dev/null 2>&1 && py -3.12 -c "" >/dev/null 2>&1; then
    BASE_PY=(py -3.12)
  elif command -v python3.12 >/dev/null 2>&1; then
    BASE_PY=(python3.12)
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PY=(python3)
  else
    BASE_PY=(python)
  fi
  "${BASE_PY[@]}" -m venv "$RELEASE_VENV"
fi

echo "==> Installing Python dependencies into .venv-release (release-constraints.txt)..."
# vendor/ChaosEngine declares `license = "Apache-2.0"` per PEP 639. Setuptools
# < 77 rejects the string form, so bump it across every platform before the
# vendor install in stage-runtime.mjs runs.
"$RELEASE_PY" -m pip install --upgrade pip -q
"$RELEASE_PY" -m pip install -c release-constraints.txt --upgrade "setuptools>=77,<82" wheel -q
case "$PLATFORM" in
  darwin)
    "$RELEASE_PY" -m pip install -c release-constraints.txt mlx mlx-lm gguf fastapi psutil uvicorn "pypdf>=6.21.4" python-multipart huggingface_hub babel -q
    ;;
  linux|windows)
    "$RELEASE_PY" -m pip install -c release-constraints.txt fastapi psutil uvicorn "pypdf>=6.21.4" python-multipart huggingface_hub babel -q
    ;;
esac
export CHAOSENGINE_EMBED_PYTHON_BIN="$RELEASE_PY"

# â”€â”€ Optional GPU bundle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# By default the installer ships CHAT-ONLY (no torch, no diffusers). Users
# click "Install GPU support" inside the app, which installs CUDA torch +
# diffusers into ~/.chaosengine/extras/ at runtime. Set
# CHAOSENGINE_BUNDLE_GPU=1 to include the GPU stack in the installer
# itself (adds ~1.3 GB, only useful for air-gapped deployments). On Windows
# the result exceeds the 1.9 GB NSIS limit and stage-runtime.mjs refuses it.
if [ "${CHAOSENGINE_BUNDLE_GPU:-}" = "1" ]; then
  echo "==> WARNING: CHAOSENGINE_BUNDLE_GPU=1 -- bundling [images] extras (adds ~1.3 GB;"
  echo "    a Windows NSIS installer cannot hold it, use a chat-only build there)"
  "$RELEASE_PY" -m pip install -c release-constraints.txt -q -e ".[desktop,images]"
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
