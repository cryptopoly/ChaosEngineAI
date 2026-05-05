#!/usr/bin/env pwsh
# Windows PowerShell port of build-llama-turbo.sh.
#
# Build llama-server-turbo from the TheTom/llama-cpp-turboquant fork.
# This fork extends standard llama-server with extra KV cache quantization
# types (iso3/4, planar3/4, turbo2/3/4) required by the RotorQuant and
# TurboQuant cache strategies, while staying compatible with all standard
# cache types.
#
# The binary is installed as ``llama-server-turbo.exe`` into
# %USERPROFILE%\.chaosengine\bin\ alongside the standard ``llama-server.exe``
# so ChaosEngineAI auto-detects it at runtime.
#
# Usage:
#   .\scripts\build-llama-turbo.ps1
#
# Prerequisites:
#   * Visual Studio 2022 Build Tools (cmake + MSVC C++)
#   * Git for Windows
#   * Optional: CUDA Toolkit 12+ for the GGML_CUDA build path
#
# Environment variables:
#   LLAMA_TURBO_DIR      Source checkout dir  (default: $env:TEMP\llama-cpp-turboquant)
#   CHAOSENGINE_BIN_DIR  Install destination  (default: $HOME\.chaosengine\bin)
#   LLAMA_TURBO_BRANCH   Git branch to build  (default: feature/turboquant-kv-cache)
#   LLAMA_TURBO_JOBS     Parallel build jobs  (default: $env:NUMBER_OF_PROCESSORS)
#   CHAOSENGINE_LLAMA_TURBO_NO_CUDA  Set to 1 to force CPU-only build even when CUDA is present.

$ErrorActionPreference = "Stop"

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

$TurboRepo   = "https://github.com/TheTom/llama-cpp-turboquant.git"
$TurboBranch = if ($env:LLAMA_TURBO_BRANCH) { $env:LLAMA_TURBO_BRANCH } else { "feature/turboquant-kv-cache" }
$TurboDir    = if ($env:LLAMA_TURBO_DIR)    { $env:LLAMA_TURBO_DIR }    else { Join-Path $env:TEMP "llama-cpp-turboquant" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$Jobs        = if ($env:LLAMA_TURBO_JOBS)   { $env:LLAMA_TURBO_JOBS }   else { $env:NUMBER_OF_PROCESSORS }
if (-not $Jobs) { $Jobs = "4" }

Write-Host "==> llama-server-turbo builder (Windows)"
Write-Host "    repo:     $TurboRepo"
Write-Host "    branch:   $TurboBranch"
Write-Host "    source:   $TurboDir"
Write-Host "    install:  $InstallDir"
Write-Host "    jobs:     $Jobs"
Write-Host ""

# Clone or update the source checkout
if (Test-Path (Join-Path $TurboDir ".git")) {
    Write-Host "==> updating existing checkout"
    Push-Location $TurboDir
    git fetch --all --prune
    Assert-LastExit "git fetch"
    git checkout $TurboBranch
    Assert-LastExit "git checkout"
    git reset --hard "origin/$TurboBranch"
    Assert-LastExit "git reset"
} else {
    Write-Host "==> cloning $TurboRepo (branch: $TurboBranch)"
    git clone --branch $TurboBranch $TurboRepo $TurboDir
    Assert-LastExit "git clone"
    Push-Location $TurboDir
}

try {
    # CMake flags. Static link mirrors the .sh shape so the installed
    # binary doesn't drag a .dll trail. CUDA is opt-in: detected via
    # ``nvcc`` on PATH unless CHAOSENGINE_LLAMA_TURBO_NO_CUDA is set.
    $cmakeFlags = @(
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF"
    )
    $forceNoCuda = $env:CHAOSENGINE_LLAMA_TURBO_NO_CUDA -eq "1"
    $hasCuda = -not $forceNoCuda -and (Get-Command nvcc -ErrorAction SilentlyContinue)
    if ($hasCuda) {
        Write-Host "==> CUDA detected (nvcc on PATH); enabling GGML_CUDA"
        $cmakeFlags += "-DGGML_CUDA=ON"
    } else {
        Write-Host "==> CUDA not detected (or disabled); building CPU-only"
    }

    Write-Host "==> cmake configure"
    cmake -B build @cmakeFlags
    Assert-LastExit "cmake configure"

    Write-Host "==> building llama-server + llama-cli"
    cmake --build build --config Release -j $Jobs --target llama-server llama-cli
    Assert-LastExit "cmake build"

    # MSVC drops .exe artefacts under build\bin\Release\ on multi-config
    # generators (the default on Windows). Single-config Ninja drops
    # them under build\bin\. Probe both.
    $candidates = @(
        "build\bin\Release\llama-server.exe",
        "build\bin\llama-server.exe"
    )
    $serverExe = $null
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $serverExe = $candidate; break }
    }
    if (-not $serverExe) {
        throw "llama-server.exe not found under build\bin — check build output."
    }
    $cliExe = $serverExe.Replace("llama-server.exe", "llama-cli.exe")

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }
    Write-Host "==> installing to $InstallDir"
    Copy-Item $serverExe (Join-Path $InstallDir "llama-server-turbo.exe") -Force
    if (Test-Path $cliExe) {
        Copy-Item $cliExe (Join-Path $InstallDir "llama-cli-turbo.exe") -Force
    }

    # Version tracking. Same shape as the .sh so the same Setup-page
    # detector works on both platforms.
    $commit = (git rev-parse HEAD).Trim()
    $versionFile = Join-Path $InstallDir "llama-server-turbo.version"
    @(
        $commit,
        $TurboBranch,
        ((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))
    ) | Set-Content -Path $versionFile -Encoding ascii
    Write-Host "==> version tracked in $versionFile"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> build complete"
Write-Host "llama-server-turbo installed to $InstallDir\llama-server-turbo.exe"
Write-Host "ChaosEngineAI will auto-detect it on next model load."
Write-Host "Restart the app if it is currently running."
