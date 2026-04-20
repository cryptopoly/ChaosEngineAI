$ErrorActionPreference = "Stop"

# PowerShell's $ErrorActionPreference = "Stop" only halts on cmdlet errors --
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

# -- Pre-flight cleanup -----------------------------------
# Windows Defender / Explorer / pip occasionally hold file handles on the
# staging tree. Clear it up front so the Node rmSync in stage-runtime.mjs
# never has to fight with locked artefacts from a prior failed run.
if (Test-Path .runtime-stage) {
    Write-Host "==> Clearing stale .runtime-stage..."
    Remove-Item -Recurse -Force .runtime-stage -ErrorAction SilentlyContinue
}

# -- Python venv ------------------------------------------
if (-not (Test-Path .venv)) {
    Write-Host "==> Creating Python venv..."
    python -m venv .venv
    Assert-LastExit "python -m venv"
}

Write-Host "==> Installing Python dependencies..."
.\.venv\Scripts\pip install --upgrade pip -q
Assert-LastExit "pip install --upgrade pip"

# On Windows, pip install torch from PyPI delivers the CPU-only wheel,
# which leaves an RTX 4090 idle and makes FLUX.1 Dev spend ~8+ minutes on
# a single step. Install the CUDA wheel first; the subsequent
# .[desktop,images] install sees torch already satisfies >=2.4.0 and
# leaves it alone.
#
# cu124 is the broadest match in 2026: it covers Python 3.9-3.13 and
# works with driver 525+. cu121 only ships wheels for Python 3.8-3.12
# so fresh Windows installs (which default to 3.13) fail outright with
# "Could not find a version that satisfies the requirement torch". We
# walk the list from newest-broad to oldest-broad and stop on the first
# success. If CHAOSENGINE_TORCH_INDEX_URL is set it overrides everything
# (set to "" to opt out entirely).
$torchIndexCandidates = if ($null -ne $env:CHAOSENGINE_TORCH_INDEX_URL) {
    if ($env:CHAOSENGINE_TORCH_INDEX_URL -eq "") {
        @()
    } else {
        @($env:CHAOSENGINE_TORCH_INDEX_URL)
    }
} else {
    @(
        "https://download.pytorch.org/whl/cu124",
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu121"
    )
}

$torchInstalled = $false
foreach ($idx in $torchIndexCandidates) {
    Write-Host "==> Installing CUDA-enabled torch from $idx..."
    .\.venv\Scripts\pip install -q --index-url $idx "torch>=2.4.0"
    if ($LASTEXITCODE -eq 0) {
        $torchInstalled = $true
        break
    }
    Write-Warning "torch install from $idx failed (exit $LASTEXITCODE) -- trying next CUDA index."
}

if (-not $torchInstalled -and $torchIndexCandidates.Count -gt 0) {
    Write-Warning "CUDA torch wheels unavailable from all candidates -- continuing with CPU torch."
    Write-Warning "You can install CUDA torch later from the Running on CPU banner inside the app."
}

# Install the same extras that stage-runtime.mjs::validateBundledPythonPackages
# checks for (desktop + images). Without `images` a release build fails strict
# validation; in dev mode it merely warns.
.\.venv\Scripts\pip install -q -e ".[desktop,images]"
Assert-LastExit "pip install -e .[desktop,images]"

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# -- npm dependencies -------------------------------------
Write-Host "==> Installing npm dependencies..."
npm ci --silent
Assert-LastExit "npm ci"

# -- llama.cpp pre-flight ---------------------------------
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
        Write-Warning "llama-server.exe not found at $llamaServerExe -- proceeding because CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA=1."
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
        throw "llama-server.exe missing -- see message above."
    }
}

# -- Patch tauri.conf.json for local builds ---------------
# Delegated to a dedicated .mjs (see scripts/patch-tauri-conf.mjs). The
# previous inline `node -e "..."` was fragile under PowerShell quoting --
# a silent misparse left the JSON empty and tauri build failed with a
# misleading EOF error.
Write-Host "==> Patching tauri.conf.json for local build..."
node scripts/patch-tauri-conf.mjs patch
Assert-LastExit "patch tauri.conf.json"

# -- Build ------------------------------------------------
Write-Host "==> Building Tauri app (NSIS installer)..."
$buildFailed = $false
try {
    npx tauri build --bundles nsis
    Assert-LastExit "tauri build"
} catch {
    $buildFailed = $true
    $buildError = $_
}

# -- Restore tauri.conf.json ------------------------------
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
    Write-Warning "Restore failed: $_ -- continuing to report original build error."
}

if ($buildFailed) {
    throw $buildError
}

Write-Host ""
Write-Host "==> Build complete!"
# Single-quoted literal so the parser never has to interpret the
# backslash sequences. Keep this file pure ASCII - PS 5.1 on Windows
# reads BOM-less scripts as Windows-1252, so any em-dash / box-drawing
# char elsewhere in the file corrupts tokenization and surfaces here
# as a spurious "string missing terminator" error.
Write-Host '    Artifacts in src-tauri\target\release\bundle\nsis'
