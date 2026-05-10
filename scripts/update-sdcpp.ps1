#!/usr/bin/env pwsh
# Windows PowerShell port of update-sdcpp.sh.
#
# Update the ``sd`` binary from leejet/stable-diffusion.cpp.
# Companion to build-sdcpp.ps1 — fetches the latest commit on the
# tracked branch, compares to the previous build's version file, and
# rebuilds only if the commit has moved.
#
# Same shape as update-llama-turbo.ps1; both delegate the actual
# rebuild to their build-*.ps1 sibling so the MSVC/CUDA toolchain
# plumbing lives in one place.
#
# Usage:
#   .\scripts\update-sdcpp.ps1
#
# Override the source dir with $env:SDCPP_DIR if your checkout lives
# somewhere other than $env:TEMP\stable-diffusion.cpp.

$ErrorActionPreference = "Stop"

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

$ScriptDir   = $PSScriptRoot
$SdcppBranch = if ($env:SDCPP_BRANCH)        { $env:SDCPP_BRANCH }        else { "master" }
$SdcppDir    = if ($env:SDCPP_DIR)           { $env:SDCPP_DIR }           else { Join-Path $env:TEMP "stable-diffusion.cpp" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$VersionFile = Join-Path $InstallDir "sd.version"

if (-not (Test-Path (Join-Path $SdcppDir ".git"))) {
    Write-Host "No existing checkout at $SdcppDir -- running full build instead."
    & (Join-Path $ScriptDir "build-sdcpp.ps1")
    exit $LASTEXITCODE
}

Push-Location $SdcppDir
try {
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

    Write-Host "==> checking out $SdcppBranch"
    git checkout $SdcppBranch
    Assert-LastExit "git checkout"

    $RemoteCommit = (git rev-parse "origin/$SdcppBranch").Trim()
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

Write-Host "==> rebuilding via build-sdcpp.ps1"
& (Join-Path $ScriptDir "build-sdcpp.ps1")
exit $LASTEXITCODE
