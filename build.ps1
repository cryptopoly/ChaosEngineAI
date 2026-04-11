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

# ── Build ────────────────────────────────────────────────
Write-Host "==> Building Tauri app (NSIS installer)..."
npx tauri build --bundles nsis

Write-Host ""
Write-Host "==> Build complete!"
Write-Host "    Artifacts: src-tauri\target\release\bundle\nsis\"
