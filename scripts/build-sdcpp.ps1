#!/usr/bin/env pwsh
# Windows PowerShell port of build-sdcpp.sh.
#
# Build the ``sd`` CLI binary from leejet/stable-diffusion.cpp (FU-008).
# Cross-platform diffusion runtime: SD 1.x/2.x/XL, FLUX.1/2, Wan 2.1 / 2.2
# video, Qwen Image, Z-Image. Wired into ChaosEngineAI as a subprocess
# engine via ``backend_service/sdcpp_image_runtime.py`` and
# ``backend_service/sdcpp_video_runtime.py``.
#
# Usage:
#   .\scripts\build-sdcpp.ps1
#
# Prerequisites:
#   * Visual Studio 2022 Build Tools (cmake + MSVC C++; the Ninja that ships with them is used)
#   * Git for Windows
#   * For a CUDA build, one of:
#       - the CUDA Toolkit 12+ (CUDA_PATH set, or nvcc on PATH), or
#       - CHAOSENGINE_SDCPP_CUDA_REDIST=1: download NVIDIA's portable CUDA archives
#         (cuda_nvcc, cuda_cudart, libcublas, cuda_cccl, cuda_profiler_api; about 700 MB,
#         sha256 checked against NVIDIA's redistrib JSON) into a folder of their own.
#         Nothing is installed system-wide. This is the recipe that built the RTX 4090
#         binary on 2026-11-19 (docs/progress/rtx4090-progress.md).
#
# Output: sd.exe in the install dir and, for a CUDA build, the CUDA runtime DLLs it loads
# (cudart64_*, cublas64_*, cublasLt64_*; about 790 MB for CUDA 12.8) beside it, so
# scripts/stage-runtime.mjs can copy the folder into the Windows bundle.
#
# Environment variables:
#   SDCPP_DIR                          Source checkout dir  (default: $env:TEMP\stable-diffusion.cpp)
#   CHAOSENGINE_BIN_DIR                Install destination  (default: $HOME\.chaosengine\bin)
#   SDCPP_REF                          Commit to build      (default: the pinned SDCPP_REF below)
#   SDCPP_JOBS                         Parallel build jobs  (default: $env:NUMBER_OF_PROCESSORS)
#   SDCPP_CUDA_ARCHITECTURES           CMAKE_CUDA_ARCHITECTURES (default: ggml's own list; "89" builds only for an RTX 40xx, much faster)
#   CHAOSENGINE_SDCPP_NO_CUDA          Set to 1 to force a CPU-only build even when CUDA is present.
#   CHAOSENGINE_SDCPP_CUDA_REDIST      Set to 1 to download NVIDIA's portable CUDA archives when no toolkit is found.
#   CHAOSENGINE_SDCPP_CUDA_VERSION     CUDA redist version  (default: 12.8.0)
#   CHAOSENGINE_SDCPP_CUDA_REDIST_DIR  Where the archives unpack (default: $HOME\.chaosengine\cuda-redist\<version>)

$ErrorActionPreference = "Stop"

# Shared MSVC/CUDA CMake helpers (Resolve-CmakeWindowsBuildContext,
# Sync-CudaVsIntegration, Get-CmakeWindowsConfigureArgs,
# Invoke-CmakeStaleCacheWipe). Same logic also drives build-llama-turbo.ps1.
. (Join-Path $PSScriptRoot "lib\windows-msvc-cuda.ps1")

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

# Load the MSVC developer environment (cl.exe, link.exe and the VS-bundled cmake/ninja) into this session
# when it is not there already. nvcc needs cl.exe on PATH, and Ninja avoids the CUDA Visual Studio
# integration files, which NVIDIA's portable archives do not install.
function Import-VcVarsIfNeeded {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return }
    $vsRoot = & $vswhere -latest -prerelease -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsRoot) { return }
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { return }
    Write-Host "==> loading the MSVC environment from $vcvars"
    $bat = Join-Path $env:TEMP "chaosengine-sdcpp-vcvars.bat"
    Set-Content -Path $bat -Encoding ascii -Value "@call `"$vcvars`" >nul 2>&1`r`n@set"
    try {
        foreach ($line in (& cmd.exe /c $bat)) {
            if ($line -match '^([^=]+)=(.*)$') {
                Set-Item -Path "env:$($matches[1])" -Value $matches[2]
            }
        }
    } finally {
        Remove-Item $bat -ErrorAction SilentlyContinue
    }
}

# Download and unpack NVIDIA's portable CUDA archives into one toolkit-shaped folder.
function Get-CudaRedistToolkit {
    param([Parameter(Mandatory)] [string] $Version, [Parameter(Mandatory)] [string] $Root)
    $nvccExe = Join-Path $Root "bin\nvcc.exe"
    $marker = Join-Path $Root ".chaosengine-complete"
    if ((Test-Path $marker) -and (Test-Path $nvccExe)) {
        Write-Host "==> CUDA $Version portable archives already unpacked at $Root"
        return $Root
    }
    $base = "https://developer.download.nvidia.com/compute/cuda/redist"
    Write-Host "==> reading $base/redistrib_$Version.json"
    $manifest = Invoke-RestMethod -UseBasicParsing "$base/redistrib_$Version.json"
    $downloads = Join-Path $Root "_downloads"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    foreach ($component in @("cuda_nvcc", "cuda_cudart", "libcublas", "cuda_cccl", "cuda_profiler_api")) {
        $entry = $manifest.$component.'windows-x86_64'
        if (-not $entry) { throw "redistrib_$Version.json has no windows-x86_64 archive for $component" }
        $zip = Join-Path $downloads (Split-Path $entry.relative_path -Leaf)
        $sizeMb = [math]::Round([double]$entry.size / 1MB)
        Write-Host "==> $component $($manifest.$component.version) ($sizeMb MB)"
        & curl.exe -L --fail --silent --show-error -o $zip "$base/$($entry.relative_path)"
        Assert-LastExit "download $component"
        $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
        if ($hash -ne $entry.sha256) {
            Remove-Item $zip -Force
            throw "$component sha256 mismatch: expected $($entry.sha256), got $hash"
        }
        $unpack = Join-Path $downloads "$component-unpacked"
        if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
        New-Item -ItemType Directory -Force -Path $unpack | Out-Null
        & tar.exe -xf $zip -C $unpack
        Assert-LastExit "unpack $component"
        # Each archive holds one top folder (<component>-windows-x86_64-<version>-archive) with bin, include, lib.
        $inner = Get-ChildItem -Path $unpack -Directory | Select-Object -First 1
        Copy-Item -Path (Join-Path $inner.FullName "*") -Destination $Root -Recurse -Force
        Remove-Item -Recurse -Force $unpack
        Remove-Item -Force $zip
    }
    Remove-Item -Recurse -Force $downloads
    if (-not (Test-Path $nvccExe)) { throw "nvcc.exe missing after unpacking the CUDA archives into $Root" }
    Set-Content -Path $marker -Encoding ascii -Value $Version
    return $Root
}

# Pinned commit, kept in step with SDCPP_REF in build-sdcpp.sh, update-sdcpp.sh and update-sdcpp.ps1
# (tests/test_check_upstreams.py checks they match).
$SdcppRef    = if ($env:SDCPP_REF)           { $env:SDCPP_REF }           else { "532d2c15c430a785a712b2d94281f04e733008a4" }
$SdcppRepo   = "https://github.com/leejet/stable-diffusion.cpp.git"
$SdcppDir    = if ($env:SDCPP_DIR)           { $env:SDCPP_DIR }           else { Join-Path $env:TEMP "stable-diffusion.cpp" }
$InstallDir  = if ($env:CHAOSENGINE_BIN_DIR) { $env:CHAOSENGINE_BIN_DIR } else { Join-Path $HOME ".chaosengine\bin" }
$Jobs        = if ($env:SDCPP_JOBS)          { $env:SDCPP_JOBS }          else { $env:NUMBER_OF_PROCESSORS }
if (-not $Jobs) { $Jobs = "4" }

Write-Host "==> stable-diffusion.cpp builder (Windows)"
Write-Host "    repo:     $SdcppRepo"
Write-Host "    commit:   $SdcppRef"
Write-Host "    source:   $SdcppDir"
Write-Host "    install:  $InstallDir"
Write-Host "    jobs:     $Jobs"
Write-Host ""

if ((Test-Path (Join-Path $SdcppDir ".git"))) {
    Write-Host "==> updating existing checkout"
    Push-Location $SdcppDir
    git fetch --all --prune
    Assert-LastExit "git fetch"
} else {
    Write-Host "==> cloning $SdcppRepo"
    git clone $SdcppRepo $SdcppDir
    Assert-LastExit "git clone"
    Push-Location $SdcppDir
}

try {
    git checkout --detach $SdcppRef
    Assert-LastExit "git checkout"
    git submodule update --init --recursive
    Assert-LastExit "git submodule update"

    Import-VcVarsIfNeeded

    # Static link so sd.exe carries ggml itself; a CUDA build still loads the CUDA runtime DLLs.
    $cmakeFlags = @(
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF"
    )

    # CUDA toolkit: CUDAToolkit_ROOT / CUDA_PATH, then nvcc on PATH, then (opt-in) NVIDIA's portable archives.
    $cudaRoot = $null
    if ($env:CHAOSENGINE_SDCPP_NO_CUDA -ne "1") {
        foreach ($candidate in @($env:CUDAToolkit_ROOT, $env:CUDA_PATH)) {
            if ($candidate -and (Test-Path (Join-Path $candidate "bin\nvcc.exe"))) { $cudaRoot = $candidate; break }
        }
        if (-not $cudaRoot) {
            $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
            if ($nvcc) { $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc.Source) }
        }
        if (-not $cudaRoot -and $env:CHAOSENGINE_SDCPP_CUDA_REDIST -eq "1") {
            $cudaVersion = if ($env:CHAOSENGINE_SDCPP_CUDA_VERSION) { $env:CHAOSENGINE_SDCPP_CUDA_VERSION } else { "12.8.0" }
            $redistDir = if ($env:CHAOSENGINE_SDCPP_CUDA_REDIST_DIR) { $env:CHAOSENGINE_SDCPP_CUDA_REDIST_DIR } `
                else { Join-Path $HOME ".chaosengine\cuda-redist\$cudaVersion" }
            $cudaRoot = Get-CudaRedistToolkit -Version $cudaVersion -Root $redistDir
        }
    }
    if ($cudaRoot) {
        $cudaRootCmake = $cudaRoot -replace '\\', '/'
        Write-Host "==> CUDA toolkit: $cudaRoot; enabling SD_CUDA"
        # SD_CUDA is the option at the pinned commit; SD_CUBLAS does not exist there and was silently ignored.
        $cmakeFlags += @(
            "-DSD_CUDA=ON",
            "-DCUDAToolkit_ROOT=$cudaRootCmake",
            "-DCMAKE_CUDA_COMPILER=$cudaRootCmake/bin/nvcc.exe"
        )
        if ($env:SDCPP_CUDA_ARCHITECTURES) {
            $cmakeFlags += "-DCMAKE_CUDA_ARCHITECTURES=$($env:SDCPP_CUDA_ARCHITECTURES)"
        }
        $env:CUDA_PATH = $cudaRoot
        $env:PATH = (Join-Path $cudaRoot "bin") + ";" + $env:PATH
    } else {
        Write-Host "==> no CUDA toolkit found (or disabled); building CPU-only."
        Write-Host "    For an NVIDIA build install the CUDA Toolkit, or rerun with CHAOSENGINE_SDCPP_CUDA_REDIST=1."
    }

    # Resolve generator + VS install (same Windows toolchain plumbing as
    # build-llama-turbo.ps1). Ninja is picked when it is on PATH, which the
    # MSVC environment above provides.
    $buildCtx = Resolve-CmakeWindowsBuildContext `
        -ProductLabel "stable-diffusion.cpp (sd-cli)" `
        -GeneratorEnv "CHAOSENGINE_SDCPP_GENERATOR"
    Write-Host "==> cmake generator: $($buildCtx.Generator)"

    $cudaIntegrationJustCopied = $false
    if ($cudaRoot -and $buildCtx.VsInstance) {
        $cudaIntegrationJustCopied = Sync-CudaVsIntegration -VsRoot $buildCtx.VsInstance
    }

    Invoke-CmakeStaleCacheWipe -Generator $buildCtx.Generator `
        -CudaIntegrationJustCopied $cudaIntegrationJustCopied

    $configureArgs = Get-CmakeWindowsConfigureArgs -Context $buildCtx -ExtraFlags $cmakeFlags

    Write-Host "==> cmake configure"
    cmake @configureArgs
    Assert-LastExit "cmake configure"

    Write-Host "==> building sd-cli binary"
    # Upstream renamed the CLI target ``sd`` -> ``sd-cli`` around master-590
    # (2026-04). Build the new target; install with the legacy ``sd.exe``
    # name so the runtime resolver and stage-runtime.mjs keep working.
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

    $backend = "cpu"
    if ($cudaRoot) {
        $backend = "cuda"
        # The CUDA runtime DLLs sd.exe loads. CUDA 12 keeps them in bin, CUDA 13 in bin\x64.
        $dllDirs = @((Join-Path $cudaRoot "bin"), (Join-Path $cudaRoot "bin\x64")) | Where-Object { Test-Path $_ }
        foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
            $dll = Get-ChildItem -Path $dllDirs -Filter $pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $dll) { throw "CUDA runtime DLL $pattern not found under $cudaRoot\bin" }
            Copy-Item $dll.FullName (Join-Path $InstallDir $dll.Name) -Force
            Write-Host "    + $($dll.Name) ($([math]::Round($dll.Length / 1MB)) MB)"
        }
    }

    $commit = (git rev-parse HEAD).Trim()
    $versionFile = Join-Path $InstallDir "sd.version"
    @(
        $commit,
        "pinned",
        ((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")),
        $backend
    ) | Set-Content -Path $versionFile -Encoding ascii
    Write-Host "==> version tracked in $versionFile"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> build complete ($backend)"
Write-Host "sd installed to $InstallDir\sd.exe"
Write-Host "ChaosEngineAI will auto-detect it on next video / image generate request."
Write-Host "Restart the app if it is currently running."
