#!/usr/bin/env pwsh
# Windows PowerShell port of update-sdcpp.sh.
#
# Update the ``sd`` binary from leejet/stable-diffusion.cpp.
# Companion to build-sdcpp.ps1: rebuilds when the installed commit
# (first line of sd.version) differs from the pinned SDCPP_REF.
#
# Same shape as update-llama-turbo.ps1; the actual rebuild is delegated
# to build-sdcpp.ps1 so the MSVC/CUDA toolchain plumbing lives in one place
# (it takes the same environment variables, e.g. CHAOSENGINE_SDCPP_CUDA_REDIST=1).
#
# Usage:
#   .\scripts\update-sdcpp.ps1

$ErrorActionPreference = "Stop"

$ScriptDir   = $PSScriptRoot
# Keep in step with SDCPP_REF in build-sdcpp.sh / build-sdcpp.ps1 (tests/test_check_upstreams.py checks they match).
$SdcppRef    = if ($env:SDCPP_REF)           { $env:SDCPP_REF }           else { "3b12781fac1a22eda27ad0f08d0884ef816f24d7" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$VersionFile = Join-Path $InstallDir "sd.version"

if (Test-Path $VersionFile) {
    $CurrentCommit = (Get-Content $VersionFile -First 1).Trim()
    Write-Host "Current installed commit: $CurrentCommit"
} else {
    $CurrentCommit = ""
    Write-Host "No version file found -- will rebuild regardless."
}
Write-Host "Pinned commit: $SdcppRef"

if ($CurrentCommit -eq $SdcppRef -and (Test-Path (Join-Path $InstallDir "sd.exe"))) {
    Write-Host ""
    Write-Host "Already up to date. No rebuild needed."
    exit 0
}

Write-Host "==> rebuilding via build-sdcpp.ps1"
$env:SDCPP_REF = $SdcppRef
& (Join-Path $ScriptDir "build-sdcpp.ps1")
exit $LASTEXITCODE
