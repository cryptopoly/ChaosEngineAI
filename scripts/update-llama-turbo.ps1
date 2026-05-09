#!/usr/bin/env pwsh
# Windows PowerShell port of update-llama-turbo.sh.
#
# Update llama-server-turbo to the latest commit and rebuild.
# Companion to build-llama-turbo.ps1 — fetches the latest changes
# from the TurboQuant fork, compares against the version file from
# the previous build, and rebuilds only if the commit has moved.
#
# When the commit is unchanged this script exits in a couple of
# seconds without touching the binary on disk. Same .sh pattern, ported
# to PowerShell so Windows users have the same one-shot update path.
#
# Usage:
#   .\scripts\update-llama-turbo.ps1
#
# Override the source dir with $env:LLAMA_TURBO_DIR if your checkout
# lives somewhere other than $env:TEMP\llama-cpp-turboquant.

$ErrorActionPreference = "Stop"

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

$ScriptDir   = $PSScriptRoot
$TurboBranch = if ($env:LLAMA_TURBO_BRANCH) { $env:LLAMA_TURBO_BRANCH } else { "feature/turboquant-kv-cache" }
$TurboDir    = if ($env:LLAMA_TURBO_DIR)    { $env:LLAMA_TURBO_DIR }    else { Join-Path $env:TEMP "llama-cpp-turboquant" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$VersionFile = Join-Path $InstallDir "llama-server-turbo.version"

# If no checkout exists yet, delegate to the full build script.
if (-not (Test-Path (Join-Path $TurboDir ".git"))) {
    Write-Host "No existing checkout at $TurboDir -- running full build instead."
    & (Join-Path $ScriptDir "build-llama-turbo.ps1")
    exit $LASTEXITCODE
}

Push-Location $TurboDir
try {
    # Show current installed version
    if (Test-Path $VersionFile) {
        $CurrentCommit = (Get-Content $VersionFile -First 1).Trim()
        Write-Host "Current installed commit: $CurrentCommit"
    } else {
        $CurrentCommit = ""
        Write-Host "No version file found -- will rebuild regardless."
    }

    Write-Host "==> fetching latest changes"
    git fetch --all --prune
    Assert-LastExit "git fetch"

    Write-Host "==> checking out $TurboBranch"
    git checkout $TurboBranch
    Assert-LastExit "git checkout"

    $RemoteCommit = (git rev-parse "origin/$TurboBranch").Trim()
    Write-Host "Remote HEAD: $RemoteCommit"

    if ($CurrentCommit -eq $RemoteCommit) {
        Write-Host ""
        Write-Host "Already up to date. No rebuild needed."
        exit 0
    }
}
finally {
    Pop-Location
}

# Commit moved — delegate the actual rebuild to build-llama-turbo.ps1.
# That script handles the existing-checkout fast path (fetch + reset
# + cmake configure + cmake build + install + version-file write) and
# is the canonical place for the MSVC/CUDA toolchain plumbing. Calling
# it here avoids the maintenance burden of two near-identical build
# pipelines for the same binary.
Write-Host "==> rebuilding via build-llama-turbo.ps1"
& (Join-Path $ScriptDir "build-llama-turbo.ps1")
exit $LASTEXITCODE
