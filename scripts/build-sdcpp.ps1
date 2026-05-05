#!/usr/bin/env pwsh
# Windows PowerShell port of build-sdcpp.sh.
#
# Build the ``sd`` CLI binary from leejet/stable-diffusion.cpp (FU-008).
# Cross-platform diffusion runtime: SD 1.x/2.x/XL, FLUX.1/2, Wan 2.1 / 2.2
# video, Qwen Image, Z-Image. Wired into ChaosEngineAI as a subprocess
# engine via ``backend_service/sdcpp_video_runtime.py``.
#
# Usage:
#   .\scripts\build-sdcpp.ps1
#
# Prerequisites:
#   * Visual Studio 2022 Build Tools (cmake + MSVC C++)
#   * Git for Windows
#   * Optional: CUDA Toolkit 12+ for the SD_CUBLAS build path
#
# Environment variables:
#   SDCPP_DIR            Source checkout dir  (default: $env:TEMP\stable-diffusion.cpp)
#   CHAOSENGINE_BIN_DIR  Install destination  (default: $HOME\.chaosengine\bin)
#   SDCPP_BRANCH         Git branch to build  (default: master)
#   SDCPP_JOBS           Parallel build jobs  (default: $env:NUMBER_OF_PROCESSORS)
#   CHAOSENGINE_SDCPP_NO_CUDA  Set to 1 to force CPU-only build even when CUDA is present.

$ErrorActionPreference = "Stop"

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

$SdcppRepo   = "https://github.com/leejet/stable-diffusion.cpp.git"
$SdcppBranch = if ($env:SDCPP_BRANCH)        { $env:SDCPP_BRANCH }        else { "master" }
$SdcppDir    = if ($env:SDCPP_DIR)           { $env:SDCPP_DIR }           else { Join-Path $env:TEMP "stable-diffusion.cpp" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$Jobs        = if ($env:SDCPP_JOBS)          { $env:SDCPP_JOBS }          else { $env:NUMBER_OF_PROCESSORS }
if (-not $Jobs) { $Jobs = "4" }

Write-Host "==> stable-diffusion.cpp builder (Windows)"
Write-Host "    repo:     $SdcppRepo"
Write-Host "    branch:   $SdcppBranch"
Write-Host "    source:   $SdcppDir"
Write-Host "    install:  $InstallDir"
Write-Host "    jobs:     $Jobs"
Write-Host ""

if (Test-Path (Join-Path $SdcppDir ".git")) {
    Write-Host "==> updating existing checkout"
    Push-Location $SdcppDir
    git fetch --all --prune
    Assert-LastExit "git fetch"
    git checkout $SdcppBranch
    Assert-LastExit "git checkout"
    git reset --hard "origin/$SdcppBranch"
    Assert-LastExit "git reset"
    git submodule update --init --recursive
    Assert-LastExit "git submodule update"
} else {
    Write-Host "==> cloning $SdcppRepo (branch: $SdcppBranch)"
    git clone --recursive --branch $SdcppBranch $SdcppRepo $SdcppDir
    Assert-LastExit "git clone"
    Push-Location $SdcppDir
}

try {
    # CMake flags. Static link so the installed sd.exe doesn't trail
    # .dll dependencies. CUDA opt-in via nvcc detection.
    $cmakeFlags = @(
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF"
    )
    $forceNoCuda = $env:CHAOSENGINE_SDCPP_NO_CUDA -eq "1"
    $hasCuda = -not $forceNoCuda -and (Get-Command nvcc -ErrorAction SilentlyContinue)
    if ($hasCuda) {
        Write-Host "==> CUDA detected (nvcc on PATH); enabling SD_CUBLAS"
        $cmakeFlags += "-DSD_CUBLAS=ON"
    } else {
        Write-Host "==> CUDA not detected (or disabled); building CPU-only"
    }

    Write-Host "==> cmake configure"
    cmake -B build @cmakeFlags
    Assert-LastExit "cmake configure"

    Write-Host "==> building sd-cli binary"
    # Upstream renamed the CLI target ``sd`` -> ``sd-cli`` around master-590
    # (2026-04). Build the new target; install with the legacy ``sd.exe``
    # name so the runtime resolver in sdcpp_video_runtime.py and
    # stage-runtime.mjs keep working without a path rename.
    cmake --build build --config Release -j $Jobs --target sd-cli
    Assert-LastExit "cmake build"

    $candidates = @(
        "build\bin\Release\sd-cli.exe",
        "build\bin\sd-cli.exe"
    )
    $sdExe = $null
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $sdExe = $candidate; break }
    }
    if (-not $sdExe) {
        throw "sd-cli.exe not found under build\bin -- check build output."
    }

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }
    Write-Host "==> installing to $InstallDir"
    Copy-Item $sdExe (Join-Path $InstallDir "sd.exe") -Force

    $commit = (git rev-parse HEAD).Trim()
    $versionFile = Join-Path $InstallDir "sd.version"
    @(
        $commit,
        $SdcppBranch,
        ((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))
    ) | Set-Content -Path $versionFile -Encoding ascii
    Write-Host "==> version tracked in $versionFile"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> build complete"
Write-Host "sd installed to $InstallDir\sd.exe"
Write-Host "ChaosEngineAI will auto-detect it on next video / image generate request."
Write-Host "Restart the app if it is currently running."
