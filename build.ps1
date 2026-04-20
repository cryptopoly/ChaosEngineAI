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
    # Walk newest-first — if the newest driver is available we prefer the
    # matching CUDA version to avoid the PyTorch JIT reconfiguring kernels
    # at startup. cu130/cu129 cover the newest Ada/Hopper drivers (RTX 50xx
    # + updated RTX 40xx firmware); cu121 is the oldest we still accept.
    @(
        "https://download.pytorch.org/whl/cu130",
        "https://download.pytorch.org/whl/cu129",
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu124",
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
# If nvidia-smi is on PATH, the build machine has an NVIDIA GPU. A silent
# CPU-only torch is the single most common "my RTX 4090 is running at CPU
# speed" failure, so we ABORT by default rather than ship a pointless
# installer. Set CHAOSENGINE_ALLOW_CPU_TORCH=1 to build anyway (e.g. if
# you genuinely want to ship a CPU-only build on an NVIDIA host for
# testing, or if you've bolted on a non-CUDA GPU backend).
$nvidiaPresent = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($nvidiaPresent) {
    Write-Host "==> Verifying bundled torch has CUDA support..."
    $cudaCheck = & .\.venv\Scripts\python.exe -c @"
import sys
info = {'python': f'{sys.version_info.major}.{sys.version_info.minor}'}
try:
    import torch
    info['torch'] = torch.__version__
    info['cuda_build'] = str(getattr(torch.version, 'cuda', None))
    info['cuda_available'] = str(bool(getattr(torch.cuda, 'is_available', lambda: False)()))
except Exception as exc:
    info['import_error'] = str(exc).splitlines()[0][:120]
print(' '.join(f'{k}={v}' for k, v in info.items()))
has_cuda = info.get('cuda_available') == 'True'
sys.exit(0 if has_cuda else 1)
"@ 2>&1
    $cudaExitCode = $LASTEXITCODE
    Write-Host "    $cudaCheck"
    if ($cudaExitCode -ne 0) {
        # Extract the Python version from the diagnostic output so the
        # actionable hint points at the specific version mismatch.
        $pyVer = if ($cudaCheck -match "python=(\d+\.\d+)") { $Matches[1] } else { "unknown" }

        Write-Host ""
        Write-Host "==> CUDA torch NOT detected on a machine with nvidia-smi." -ForegroundColor Red
        Write-Host ""
        Write-Host "    Build machine: Python $pyVer + NVIDIA GPU (nvidia-smi present)"
        Write-Host "    Bundled torch: $cudaCheck"
        Write-Host ""
        Write-Host "    The pip install from download.pytorch.org fell back to the CPU wheel"
        Write-Host "    because no CUDA index publishes wheels for Python $pyVer yet."
        Write-Host ""
        Write-Host "    FIX OPTIONS, in order of preference:"
        Write-Host ""
        Write-Host "    1. Use Python 3.13 for the build (CUDA wheels exist for 3.9-3.13):"
        Write-Host "         Remove-Item -Recurse -Force .venv"
        Write-Host "         py -3.13 -m venv .venv"
        Write-Host "         .\build.ps1"
        Write-Host ""
        Write-Host "    2. Point at a newer CUDA index that does publish $pyVer wheels:"
        Write-Host "         `$env:CHAOSENGINE_TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu130'"
        Write-Host "         .\build.ps1"
        Write-Host ""
        Write-Host "    3. Ship CPU-only torch deliberately (not recommended on RTX GPUs):"
        Write-Host "         `$env:CHAOSENGINE_ALLOW_CPU_TORCH = '1'"
        Write-Host "         .\build.ps1"
        Write-Host ""
        if ($env:CHAOSENGINE_ALLOW_CPU_TORCH -ne "1") {
            throw "Refusing to bundle CPU-only torch on an NVIDIA host -- see options above. Set CHAOSENGINE_ALLOW_CPU_TORCH=1 to bypass."
        }
        Write-Warning "CHAOSENGINE_ALLOW_CPU_TORCH=1 is set -- continuing with CPU torch."
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
