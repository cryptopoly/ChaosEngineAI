$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ── Pre-flight cleanup ───────────────────────────────────
# Windows Defender / Explorer / pip occasionally hold file handles on the
# staging tree. Clear it up front so the Node rmSync in stage-runtime.mjs
# never has to fight with locked artefacts from a prior failed run.
if (Test-Path .runtime-stage) {
    Write-Host "==> Clearing stale .runtime-stage..."
    Remove-Item -Recurse -Force .runtime-stage -ErrorAction SilentlyContinue
}

# ── Python venv ──────────────────────────────────────────
if (-not (Test-Path .venv)) {
    Write-Host "==> Creating Python venv..."
    python -m venv .venv
}

Write-Host "==> Installing Python dependencies..."
.\.venv\Scripts\pip install --upgrade pip -q
# Install the same extras that stage-runtime.mjs::validateBundledPythonPackages
# checks for (desktop + images). Without `images` a release build fails strict
# validation; in dev mode it merely warns.
.\.venv\Scripts\pip install -q -e ".[desktop,images]"

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# ── npm dependencies ─────────────────────────────────────
Write-Host "==> Installing npm dependencies..."
npm ci --silent

# ── Patch tauri.conf.json for local builds ───────────────
Write-Host "==> Patching tauri.conf.json for local build..."
node -e "const fs = require('fs'); const conf = JSON.parse(fs.readFileSync('src-tauri/tauri.conf.json', 'utf8')); conf.build.beforeBundleCommand = 'npm run stage:runtime'; conf.bundle = conf.bundle || {}; conf.bundle.createUpdaterArtifacts = false; fs.writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(conf, null, 2) + '\n');"

# ── Build ────────────────────────────────────────────────
Write-Host "==> Building Tauri app (NSIS installer)..."
npx tauri build --bundles nsis

# Restore tauri.conf.json
git checkout src-tauri/tauri.conf.json 2>&1 | Out-Null

Write-Host ""
Write-Host "==> Build complete!"
Write-Host "    Artifacts: src-tauri\target\release\bundle\nsis\"
