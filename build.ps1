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

# vendor/ChaosEngine declares `license = "Apache-2.0"` per PEP 639. Setuptools
# < 77 rejects the string form ("project.license must be valid exactly by one
# definition"), and fresh venvs on Windows sometimes resolve an older
# setuptools than the one bundled with CPython. Force an upgrade here so
# stage-runtime.mjs can install the vendor package via `pip install --target`.
#
# Upper bound of <82 is load-bearing: recent torch wheels (e.g. 2.11.x)
# declare ``setuptools<82`` as a runtime metadata constraint, and pip's
# dependency-warning heuristic surfaces that as a loud yellow warning on
# every invocation after setuptools 82 is installed. 77..81 covers PEP 639
# while staying inside torch's supported range.
.\.venv\Scripts\pip install --upgrade "setuptools>=77,<82" wheel -q
Assert-LastExit "pip install --upgrade setuptools wheel"

# Chat-only bundle: no torch, no diffusers, no CUDA DLLs. The installer
# ships ~500 MB instead of ~1.9 GB. Users who want Image / Video Studio
# click "Install GPU support" inside the app, which downloads CUDA torch
# + diffusers to a persistent ``%LOCALAPPDATA%\ChaosEngineAI\extras\``
# directory that survives app updates. See backend_service/routes/setup.py
# :install_gpu_bundle for the runtime install flow.
#
# To include the GPU stack in the installer anyway (e.g. for air-gapped
# deployments that can't download at runtime), set CHAOSENGINE_BUNDLE_GPU=1.
.\.venv\Scripts\pip install -q -e ".[desktop]"
Assert-LastExit "pip install -e .[desktop]"

if ($env:CHAOSENGINE_BUNDLE_GPU -eq "1") {
    Write-Host "==> CHAOSENGINE_BUNDLE_GPU=1 -- also bundling [images] extras"
    .\.venv\Scripts\pip install -q -e ".[desktop,images]"
    Assert-LastExit "pip install -e .[desktop,images]"
}

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# -- npm dependencies -------------------------------------
Write-Host "==> Installing npm dependencies..."
npm ci --silent
Assert-LastExit "npm ci"

# -- llama.cpp pre-flight ---------------------------------
# If llama.cpp isn't built locally, stage-runtime.mjs auto-downloads a
# prebuilt release from ggml-org/llama.cpp (Vulkan wheel — works on every
# GPU without bundling cuDNN). That makes the installer truly "works out
# of the box" on a fresh Windows box with no cmake / VS Build Tools setup.
# The ALLOW_NO_LLAMA flag still kicks in if the network is unreachable,
# so CI and air-gapped builds don't break.
$llamaBinDir = if ($env:CHAOSENGINE_LLAMA_BIN_DIR) {
    $env:CHAOSENGINE_LLAMA_BIN_DIR
} else {
    Join-Path (Split-Path -Parent $ScriptDir) "llama.cpp\build\bin"
}
$llamaServerExe = Join-Path $llamaBinDir "llama-server.exe"
if (-not (Test-Path $llamaServerExe)) {
    # Opt OUT of the auto-download fallback with CHAOSENGINE_REQUIRE_LLAMA=1.
    if ($env:CHAOSENGINE_REQUIRE_LLAMA -eq "1") {
        Write-Host ""
        Write-Host "==> ERROR: llama-server.exe not found at $llamaServerExe" -ForegroundColor Red
        Write-Host ""
        Write-Host "    CHAOSENGINE_REQUIRE_LLAMA=1 is set, so the build must ship with"
        Write-Host "    a locally-compiled llama.cpp. Pick one:"
        Write-Host ""
        Write-Host "    1. Build llama.cpp: clone to ..\llama.cpp then run"
        Write-Host "       cmake -B build; cmake --build build --config Release"
        Write-Host "    2. Set CHAOSENGINE_LLAMA_BIN_DIR to your llama.cpp build directory"
        Write-Host "    3. Unset CHAOSENGINE_REQUIRE_LLAMA and let stage-runtime"
        Write-Host "       auto-download a prebuilt release from ggml-org/llama.cpp"
        Write-Host ""
        throw "llama-server.exe missing -- see message above."
    }

    Write-Host "==> llama-server.exe not found locally at $llamaServerExe"
    Write-Host "    stage-runtime.mjs will attempt to auto-download a prebuilt release"
    Write-Host "    from ggml-org/llama.cpp (Vulkan wheel). If download fails, the"
    Write-Host "    installer ships without inference and users install it later."
    # Allow graceful skip if the auto-download also fails (offline builds).
    $env:CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA = "1"
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

# -- Publish installers to /assets ------------------------
# The Tauri bundle tree is three directories deep and differs per target,
# which makes "where is my installer" a recurring question. Copy the
# shippable artifacts into a flat ``assets/`` folder at the repo root.
Write-Host "==> Publishing artifacts to assets/..."
node scripts/publish-artifacts.mjs --bundles=nsis
Assert-LastExit "publish-artifacts"

Write-Host ""
Write-Host "==> Build complete!"
# Single-quoted literal so the parser never has to interpret the
# backslash sequences. Keep this file pure ASCII - PS 5.1 on Windows
# reads BOM-less scripts as Windows-1252, so any em-dash / box-drawing
# char elsewhere in the file corrupts tokenization and surfaces here
# as a spurious "string missing terminator" error.
Write-Host '    Artifacts in assets\ (also in src-tauri\target\release\bundle\nsis)'
