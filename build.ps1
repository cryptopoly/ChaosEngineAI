$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ── Python venv ──────────────────────────────────────────
if (-not (Test-Path .venv)) {
    Write-Host "==> Creating Python venv..."
    python -m venv .venv
}

Write-Host "==> Installing Python dependencies..."
.\.venv\Scripts\pip install --upgrade pip -q
.\.venv\Scripts\pip install fastapi psutil uvicorn pypdf python-multipart huggingface_hub -q

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# ── npm dependencies ─────────────────────────────────────
Write-Host "==> Installing npm dependencies..."
npm ci --silent

# ── Patch tauri.conf.json for local builds ───────────────
Write-Host "==> Patching tauri.conf.json for local build..."
node -e "const fs = require('fs'); const conf = JSON.parse(fs.readFileSync('src-tauri/tauri.conf.json', 'utf8')); conf.build.beforeBundleCommand = 'npm run stage:runtime'; if (conf.plugins && conf.plugins.updater) { delete conf.plugins.updater.pubkey; } conf.bundle = conf.bundle || {}; conf.bundle.createUpdaterArtifacts = false; fs.writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(conf, null, 2) + '\n');"

# ── Build ────────────────────────────────────────────────
Write-Host "==> Building Tauri app (NSIS installer)..."
npx tauri build --bundles nsis

# Restore tauri.conf.json
git checkout src-tauri/tauri.conf.json 2>&1 | Out-Null

Write-Host ""
Write-Host "==> Build complete!"
Write-Host "    Artifacts: src-tauri\target\release\bundle\nsis\"
