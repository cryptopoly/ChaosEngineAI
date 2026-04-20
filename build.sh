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
    .venv/bin/pip install --upgrade "setuptools>=77" wheel -q
    .venv/bin/pip install mlx mlx-lm gguf fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  linux)
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install --upgrade "setuptools>=77" wheel -q
    .venv/bin/pip install fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    ;;
  windows)
    .venv/Scripts/pip install --upgrade pip -q
    .venv/Scripts/pip install --upgrade "setuptools>=77" wheel -q
    .venv/Scripts/pip install fastapi psutil uvicorn "pypdf>=6.10.2" python-multipart huggingface_hub -q
    export CHAOSENGINE_EMBED_PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
    ;;
esac

# ── npm dependencies ─────────────────────────────────────
echo "==> Installing npm dependencies..."
npm ci --silent

# ── Patch tauri.conf.json for local builds ───────────────
# Save the fields we're about to change so we can surgically restore them
# at the end without clobbering other pending edits (like a version bump).
echo "==> Patching tauri.conf.json for local build..."
ORIGINAL_BEFORE_BUNDLE=$(node -e "console.log(JSON.parse(require('fs').readFileSync('src-tauri/tauri.conf.json','utf8')).build.beforeBundleCommand || '')")
ORIGINAL_CREATE_UPDATER=$(node -e "console.log(JSON.parse(require('fs').readFileSync('src-tauri/tauri.conf.json','utf8')).bundle?.createUpdaterArtifacts ?? true)")
export ORIGINAL_BEFORE_BUNDLE ORIGINAL_CREATE_UPDATER
node -e "
  const fs = require('fs');
  const conf = JSON.parse(fs.readFileSync('src-tauri/tauri.conf.json', 'utf8'));
  conf.build.beforeBundleCommand = 'npm run stage:runtime';
  // Disable updater artifact signing for local builds (requires CI-only secrets)
  // Keep the pubkey and endpoints intact — the updater plugin needs them at runtime.
  conf.bundle = conf.bundle || {};
  conf.bundle.createUpdaterArtifacts = false;
  fs.writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(conf, null, 2) + '\n');
"

# ── Build ────────────────────────────────────────────────
case "$PLATFORM" in
  darwin)  BUNDLES="app,dmg"     ;;
  linux)   BUNDLES="appimage,deb" ;;
  windows) BUNDLES="nsis"         ;;
esac

echo "==> Building Tauri app (bundles: $BUNDLES)..."
npx tauri build --bundles "$BUNDLES"

# Restore only the two fields we patched — don't blow away other pending
# edits such as version bumps or plugin config changes the developer made
# between the start of this build and now.
node -e "
  const fs = require('fs');
  const conf = JSON.parse(fs.readFileSync('src-tauri/tauri.conf.json', 'utf8'));
  const origBefore = process.env.ORIGINAL_BEFORE_BUNDLE;
  if (origBefore) conf.build.beforeBundleCommand = origBefore;
  const origCreate = process.env.ORIGINAL_CREATE_UPDATER;
  if (origCreate !== undefined && origCreate !== '') {
    conf.bundle = conf.bundle || {};
    conf.bundle.createUpdaterArtifacts = origCreate === 'true';
  }
  fs.writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(conf, null, 2) + '\n');
" 2>/dev/null || true

# ── Publish installers to /assets ────────────────────────
# The Tauri bundle tree is three directories deep and differs per target.
# Copy the shippable artifacts into a flat assets/ folder at the repo root
# so every build lands in the same place regardless of platform.
echo "==> Publishing artifacts to assets/..."
node scripts/publish-artifacts.mjs --bundles="$BUNDLES"

echo ""
echo "==> Build complete!"
echo "    Artifacts: assets/ (also in src-tauri/target/release/bundle/)"
