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

    # Pick a CMake generator explicitly. Without -G, cmake defaults to
    # "NMake Makefiles" on Windows, which dies with
    # "Running 'nmake' '-?' failed" unless the user happens to be inside
    # a Visual Studio Developer Command Prompt. The user's expected
    # entry point is a vanilla PowerShell, so probe for a usable
    # generator in this order:
    #   1. $env:CHAOSENGINE_LLAMA_TURBO_GENERATOR (manual override)
    #   2. Ninja  -- single-config, fast, but optional
    #   3. Visual Studio 17 2022 (cmake locates VS via vswhere even
    #      from outside a developer prompt, as long as VS 2022 Build
    #      Tools are installed -- which the script header lists as a
    #      prerequisite anyway).
    if ($env:CHAOSENGINE_LLAMA_TURBO_GENERATOR) {
        $generator = $env:CHAOSENGINE_LLAMA_TURBO_GENERATOR
    } elseif (Get-Command ninja -ErrorAction SilentlyContinue) {
        $generator = "Ninja"
    } else {
        $generator = "Visual Studio 17 2022"
    }

    # Pre-flight: confirm a usable C++ toolchain is actually installed.
    # CMake's failure message ("could not find any instance of Visual
    # Studio") is correct but easy to misread as a script bug -- and on
    # CUDA hosts it's especially confusing because nvcc was detected
    # successfully. nvcc proxies to cl.exe on Windows, so CUDA without
    # MSVC cannot compile anything. Detect the missing-toolchain state
    # up front and surface the install link the user actually needs.
    #
    # -all is required: VS Build Tools installs frequently report
    # isComplete=0 (Microsoft's installer flags some optional component
    # as missing) even when cl.exe works fine. vswhere -latest WITHOUT
    # -all silently excludes those, and so does CMake's own internal
    # probe -- which is why a working install can still produce
    # "could not find any instance of Visual Studio" from cmake. Probe
    # with -all, then verify cl.exe truly exists, then pass the install
    # path explicitly to cmake via CMAKE_GENERATOR_INSTANCE so it
    # doesn't repeat the same -latest filter and fail again.
    $vsInstance = $null
    if ($generator -like "Visual Studio*") {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        $clCandidates = @()
        if (Test-Path $vswhere) {
            $clCandidates = & $vswhere -all -prerelease -products * `
                -find "VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe" 2>$null
        }
        if ($clCandidates) {
            # Pick the highest version dir under VC\Tools\MSVC.
            $clExe = $clCandidates | Sort-Object -Descending | Select-Object -First 1
            # Walk up from
            #   <root>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe
            # to <root>: 8 segments to strip (x64, Hostx64, bin, <ver>,
            # MSVC, Tools, VC, cl.exe-the-leaf-itself).
            $vsInstance = $clExe
            for ($i = 0; $i -lt 8; $i++) { $vsInstance = Split-Path -Parent $vsInstance }
            Write-Host "==> Visual Studio detected at: $vsInstance"
            Write-Host "    cl.exe: $clExe"
        } else {
            $msg = @(
                "",
                "Visual Studio 2022 with the C++ workload is not installed.",
                "llama-server-turbo cannot build without an MSVC toolchain --",
                "and on CUDA hosts, nvcc itself proxies to cl.exe, so even the",
                "CUDA path requires MSVC. Install one of:",
                "",
                "  * Visual Studio 2022 Community (free, full IDE):",
                "      https://visualstudio.microsoft.com/vs/community/",
                "  * Visual Studio Build Tools 2022 (compiler only, smaller):",
                "      https://visualstudio.microsoft.com/visual-cpp-build-tools/",
                "",
                "During install, tick 'Desktop development with C++'",
                "(or, in Build Tools, the 'C++ build tools' workload).",
                "Re-run this script afterwards.",
                ""
            ) -join [Environment]::NewLine
            throw $msg
        }
    }
    Write-Host "==> cmake generator: $generator"
    $configureArgs = @("-B", "build", "-G", $generator)
    if ($generator -like "Visual Studio*") {
        $configureArgs += @("-A", "x64")
        # Pin CMake to the install we just verified, so it doesn't run
        # its own -latest probe and reject an isComplete=0 install.
        if ($vsInstance) {
            $configureArgs += @("-DCMAKE_GENERATOR_INSTANCE=$vsInstance")
        }
    }
    $configureArgs += $cmakeFlags

    # CMake refuses to switch generators in an existing build directory --
    # a previous failed run that defaulted to "NMake Makefiles" leaves a
    # CMakeCache.txt that aborts subsequent runs with "Does not match the
    # generator used previously". Detect a generator mismatch and wipe
    # build/ so the user doesn't have to clean up by hand.
    #
    # Note: do NOT use -SimpleMatch on the Select-String pattern -- it
    # disables regex, which makes the leading ^ a literal character and
    # silently misses every line. Use a regex anchor instead.
    $cachePath = "build\CMakeCache.txt"
    if (Test-Path $cachePath) {
        $cachedGeneratorLine = Select-String -Path $cachePath -Pattern '^CMAKE_GENERATOR:INTERNAL=' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cachedGeneratorLine) {
            $cachedGenerator = ($cachedGeneratorLine.Line -split "=", 2)[1].Trim()
            if ($cachedGenerator -and ($cachedGenerator -ne $generator)) {
                Write-Host "==> stale CMake cache (was '$cachedGenerator', want '$generator'); wiping build\"
                Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
            }
        }
    }

    Write-Host "==> cmake configure"
    cmake @configureArgs
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
        throw "llama-server.exe not found under build\bin -- check build output."
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
