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

# -- Verify CUDA torch actually installed -----------------
# If nvidia-smi is on PATH, the build machine has an NVIDIA GPU — the
# bundled torch MUST have CUDA support or the installer ships with a
# silent CPU-only runtime. Catching this at build time beats users
# opening the app and seeing "Running on CPU" on an RTX 4090.
$nvidiaPresent = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($nvidiaPresent) {
    Write-Host "==> Verifying bundled torch has CUDA support..."
    $cudaCheck = & .\.venv\Scripts\python.exe -c @"
import sys
try:
    import torch
except Exception as exc:
    print(f'torch-import-error:{exc}')
    sys.exit(2)
has_cuda = bool(getattr(torch.cuda, 'is_available', lambda: False)())
build = getattr(torch.version, 'cuda', None)
print(f'torch={torch.__version__} cuda_available={has_cuda} cuda_build={build}')
sys.exit(0 if has_cuda else 1)
"@ 2>&1
    $cudaExitCode = $LASTEXITCODE
    Write-Host "    $cudaCheck"
    if ($cudaExitCode -ne 0) {
        Write-Warning ""
        Write-Warning "==> CUDA torch NOT detected on a machine with nvidia-smi!"
        Write-Warning "    The bundled installer will run on CPU, which on an RTX 4090 means"
        Write-Warning "    ~8 minutes per diffusion step instead of ~1 second."
        Write-Warning ""
        Write-Warning "    The pip install from download.pytorch.org probably fell back to the"
        Write-Warning "    CPU wheel because your Python version doesn't have a matching CUDA"
        Write-Warning "    wheel in the default cu124 index. Try another CUDA index:"
        Write-Warning "      `$env:CHAOSENGINE_TORCH_INDEX_URL='https://download.pytorch.org/whl/cu126'"
        Write-Warning "    or downgrade to Python 3.13 (cu124 doesn't publish 3.14 wheels yet)."
        Write-Warning ""
        Write-Warning "    Continuing the build. Set CHAOSENGINE_REQUIRE_CUDA_TORCH=1 to fail"
        Write-Warning "    instead of shipping CPU-only torch."
        if ($env:CHAOSENGINE_REQUIRE_CUDA_TORCH -eq "1") {
            throw "CUDA torch required but not detected -- see warning above."
        }
    }
}

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
