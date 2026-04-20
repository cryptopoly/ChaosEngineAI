$ErrorActionPreference = "Stop"

# PowerShell's $ErrorActionPreference = "Stop" only halts on cmdlet errors —
# native command failures (pip, npm, node, npx, git) are silently ignored
# unless we check $LASTEXITCODE ourselves. Without this helper, the previous
# build printed "Build complete!" even when `tauri build` had failed.
function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

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
    Assert-LastExit "python -m venv"
}

Write-Host "==> Installing Python dependencies..."
.\.venv\Scripts\pip install --upgrade pip -q
Assert-LastExit "pip install --upgrade pip"

# On Windows, `pip install torch` from PyPI delivers the CPU-only wheel —
# which leaves an RTX 4090 idle and makes FLUX.1 Dev spend ~8+ minutes on
# a single step. Install the CUDA 12.1 wheel first (cu121 is the broadest
# match for driver 525+ and works on 12.x toolchains); the subsequent
# `.[desktop,images]` install sees torch already satisfies `>=2.4.0` and
# leaves it alone. Override with CHAOSENGINE_TORCH_INDEX_URL if needed
# (e.g. set to "" to opt out, or point at cu124 for newer systems).
$torchIndex = if ($null -ne $env:CHAOSENGINE_TORCH_INDEX_URL) {
    $env:CHAOSENGINE_TORCH_INDEX_URL
} else {
    "https://download.pytorch.org/whl/cu121"
}
if ($torchIndex -ne "") {
    Write-Host "==> Installing CUDA-enabled torch from $torchIndex..."
    .\.venv\Scripts\pip install -q --index-url $torchIndex "torch>=2.4.0"
    Assert-LastExit "pip install torch (CUDA)"
}

# Install the same extras that stage-runtime.mjs::validateBundledPythonPackages
# checks for (desktop + images). Without `images` a release build fails strict
# validation; in dev mode it merely warns.
.\.venv\Scripts\pip install -q -e ".[desktop,images]"
Assert-LastExit "pip install -e .[desktop,images]"

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# ── npm dependencies ─────────────────────────────────────
Write-Host "==> Installing npm dependencies..."
npm ci --silent
Assert-LastExit "npm ci"

# ── llama.cpp pre-flight ─────────────────────────────────
# Release-mode staging is strict: if llama.cpp is not built locally the
# install will refuse to ship without inference. Detect upfront so the user
# gets a clear message before npm/cargo spend 20 minutes building, and let
# them opt into shipping a diffusers-only installer with one env var.
$llamaBinDir = if ($env:CHAOSENGINE_LLAMA_BIN_DIR) {
    $env:CHAOSENGINE_LLAMA_BIN_DIR
} else {
    Join-Path (Split-Path -Parent $ScriptDir) "llama.cpp\build\bin"
}
$llamaServerExe = Join-Path $llamaBinDir "llama-server.exe"
if (-not (Test-Path $llamaServerExe)) {
    if ($env:CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA -eq "1") {
        Write-Warning "llama-server.exe not found at $llamaServerExe — proceeding because CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA=1."
        Write-Warning "The installer will ship without llama.cpp; users can install it via the Setup page."
    } else {
        Write-Host ""
        Write-Host "==> ERROR: llama-server.exe not found at $llamaServerExe" -ForegroundColor Red
        Write-Host ""
        Write-Host "    A release build needs llama.cpp compiled locally so the bundled"
        Write-Host "    installer ships with native inference support. Pick one:"
        Write-Host ""
        # No parens in these strings. PS 5.1 mis-tokenizes a backslash
        # immediately before a parenthesis as a subexpression start, and
        # unbalanced parens across split strings have tripped the parser
        # in other places. Plain prose is safest.
        Write-Host "    1. Build llama.cpp: clone to ..\llama.cpp then run"
        Write-Host "       cmake -B build; cmake --build build --config Release"
        Write-Host "    2. Set CHAOSENGINE_LLAMA_BIN_DIR to your llama.cpp build directory"
        Write-Host "    3. Ship without it: set CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA=1"
        Write-Host ""
        throw "llama-server.exe missing — see message above."
    }
}

# ── Patch tauri.conf.json for local builds ───────────────
# Delegated to a dedicated .mjs (see scripts/patch-tauri-conf.mjs). The
# previous inline `node -e "..."` was fragile under PowerShell quoting —
# a silent misparse left the JSON empty and tauri build failed with a
# misleading EOF error.
Write-Host "==> Patching tauri.conf.json for local build..."
node scripts/patch-tauri-conf.mjs patch
Assert-LastExit "patch tauri.conf.json"

# ── Build ────────────────────────────────────────────────
Write-Host "==> Building Tauri app (NSIS installer)..."
$buildFailed = $false
try {
    npx tauri build --bundles nsis
    Assert-LastExit "tauri build"
} catch {
    $buildFailed = $true
    $buildError = $_
}

# ── Restore tauri.conf.json ──────────────────────────────
# Always run the restore, even if the build failed, so the working tree
# is left clean for the next attempt. If the build succeeded and git
# complains, surface it; if the build already failed, restore is
# best-effort and the build error takes precedence.
Write-Host "==> Restoring tauri.conf.json..."
try {
    node scripts/patch-tauri-conf.mjs restore
    if (-not $buildFailed) {
        Assert-LastExit "restore tauri.conf.json"
    }
} catch {
    if (-not $buildFailed) { throw }
    Write-Warning "Restore failed: $_ — continuing to report original build error."
}

if ($buildFailed) {
    throw $buildError
}

Write-Host ""
Write-Host "==> Build complete!"
Write-Host "    Artifacts in src-tauri\target\release\bundle\nsis"
