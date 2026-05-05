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

    # Helper: ensure CUDA's MSBuild integration files (.props/.targets/etc.)
    # are copied into the VS BuildCustomizations dir. CMake's CUDA detection
    # bails with "No CUDA toolset found" when these files are missing --
    # which happens whenever CUDA was installed before Visual Studio, or
    # when the CUDA installer's "Visual Studio Integration" component was
    # unticked. Auto-elevates via UAC if the target dir isn't writable.
    function Sync-CudaVsIntegration {
        param(
            [Parameter(Mandatory)] [string] $VsRoot
        )
        $cudaPath = $env:CUDA_PATH
        if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
            Write-Host "==> CUDA_PATH not set; skipping VS integration sync"
            return
        }
        $cudaSrc = Join-Path $cudaPath "extras\visual_studio_integration\MSBuildExtensions"
        $vsTarget = Join-Path $VsRoot "MSBuild\Microsoft\VC\v170\BuildCustomizations"
        if (-not (Test-Path $cudaSrc)) {
            Write-Host "==> CUDA integration source not found at $cudaSrc; skipping sync"
            return
        }
        if (-not (Test-Path $vsTarget)) {
            Write-Host "==> VS BuildCustomizations dir not found at $vsTarget; skipping sync"
            return
        }
        $sourceFiles = Get-ChildItem -Path $cudaSrc -File -ErrorAction SilentlyContinue
        $missing = @($sourceFiles | Where-Object { -not (Test-Path (Join-Path $vsTarget $_.Name)) })
        if (-not $missing -or $missing.Count -eq 0) {
            Write-Host "==> CUDA VS integration already present in $vsTarget"
            return $false
        }
        Write-Host "==> CUDA VS integration missing $($missing.Count) file(s) from $vsTarget"
        $missing | ForEach-Object { Write-Host "    - $($_.Name)" }

        # Try direct copy first; fall back to elevated copy via UAC if the
        # target dir refuses our writes.
        $copied = $true
        try {
            foreach ($file in $missing) {
                Copy-Item -LiteralPath $file.FullName -Destination $vsTarget -Force -ErrorAction Stop
            }
            Write-Host "==> CUDA VS integration files copied (direct)"
        } catch {
            $copied = $false
            Write-Host "==> Direct copy denied; relaunching as admin via UAC..."
            # Build a per-file Copy-Item script. Cannot use a wildcard with
            # -LiteralPath -- it treats the * as a literal character and
            # silently copies nothing -- so iterate over the missing files
            # by full path. We also verify each landing in the elevated
            # session and exit non-zero if any failed, so the parent script
            # detects partial copies.
            $copyCommands = $missing | ForEach-Object {
                $srcEsc = $_.FullName.Replace("'", "''")
                $dstEsc = $vsTarget.Replace("'", "''")
                "Copy-Item -LiteralPath '$srcEsc' -Destination '$dstEsc' -Force"
            }
            $verifyLine = (
                "if (@(Get-ChildItem -LiteralPath '" + $vsTarget.Replace("'", "''") +
                "' -Filter 'CUDA *.props' -ErrorAction SilentlyContinue).Count -eq 0) { exit 1 }"
            )
            $script = ($copyCommands + @($verifyLine)) -join "; "
            $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $script)
            try {
                $proc = Start-Process -FilePath powershell -ArgumentList $argList -Verb RunAs -Wait -PassThru
                if ($proc.ExitCode -eq 0) {
                    # Re-verify from the parent shell so a buggy elevated
                    # script can't claim success without leaving files.
                    $stillMissing = @($sourceFiles | Where-Object {
                        -not (Test-Path (Join-Path $vsTarget $_.Name))
                    })
                    if ($stillMissing.Count -eq 0) {
                        $copied = $true
                        Write-Host "==> CUDA VS integration files copied (elevated)"
                    } else {
                        Write-Host "==> Elevated copy reported success but $($stillMissing.Count) file(s) still missing:"
                        $stillMissing | ForEach-Object { Write-Host "    - $($_.Name)" }
                    }
                } else {
                    Write-Host "==> Elevated copy exited with code $($proc.ExitCode)"
                }
            } catch {
                Write-Host "==> UAC copy failed: $_"
            }
        }
        if (-not $copied) {
            $manualCopy = $missing | ForEach-Object {
                "  Copy-Item -LiteralPath '$($_.FullName)' -Destination '$vsTarget' -Force"
            }
            $msg = @(
                "",
                "Could not install CUDA's Visual Studio integration files.",
                "Run the following in an Administrator PowerShell, then retry:",
                ""
            ) + $manualCopy + @("")
            throw ($msg -join [Environment]::NewLine)
        }
        return $true
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
    $vsInstanceVersion = $null
    if ($generator -like "Visual Studio*") {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        $clCandidates = @()
        $vsInstalls = @()
        if (Test-Path $vswhere) {
            $clCandidates = & $vswhere -all -prerelease -products * `
                -find "VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe" 2>$null
            $vsInstallsJson = & $vswhere -all -prerelease -products * -format json 2>$null
            if ($vsInstallsJson) {
                $vsInstalls = $vsInstallsJson | ConvertFrom-Json
            }
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
            # Match the resolved root against the JSON listing to grab
            # installationVersion. CMake's generator wants
            # "<path>,version=<version>" when an isComplete=0 install
            # isn't present in the Installer's known-instances registry,
            # otherwise it bails with "instance is not known to the
            # Visual Studio Installer".
            $matchedInstall = $vsInstalls | Where-Object {
                $_.installationPath.TrimEnd('\') -eq $vsInstance.TrimEnd('\')
            } | Select-Object -First 1
            if ($matchedInstall) {
                $vsInstanceVersion = $matchedInstall.installationVersion
            }
            Write-Host "==> Visual Studio detected at: $vsInstance"
            if ($vsInstanceVersion) {
                Write-Host "    version: $vsInstanceVersion"
            }
            Write-Host "    cl.exe:  $clExe"
            # CMake's CUDA detection needs CUDA's MSBuild .props/.targets
            # files installed under VS. Sync them now if missing.
            $cudaIntegrationJustCopied = $false
            if ($hasCuda) {
                $cudaIntegrationJustCopied = Sync-CudaVsIntegration -VsRoot $vsInstance
            }
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
        # Append ",version=<x>" so CMake doesn't reject the path with
        # "instance is not known to the Visual Studio Installer" -- the
        # Installer registry skips isComplete=0 entries.
        if ($vsInstance) {
            $instanceArg = if ($vsInstanceVersion) {
                "$vsInstance,version=$vsInstanceVersion"
            } else {
                $vsInstance
            }
            $configureArgs += @("-DCMAKE_GENERATOR_INSTANCE=$instanceArg")
        }
    }
    $configureArgs += $cmakeFlags

    # CMake refuses to switch generators in an existing build directory --
    # a previous failed run that defaulted to "NMake Makefiles" leaves a
    # CMakeCache.txt that aborts subsequent runs with "Does not match the
    # generator used previously". Detect a generator mismatch and wipe
    # build/ so the user doesn't have to clean up by hand.
    #
    # We also wipe build/ when CUDA's VS integration was just installed,
    # because the previous configure cached "no CUDA toolset" results
    # that won't re-evaluate even though the underlying state changed.
    #
    # Note: do NOT use -SimpleMatch on the Select-String pattern -- it
    # disables regex, which makes the leading ^ a literal character and
    # silently misses every line. Use a regex anchor instead.
    $cachePath = "build\CMakeCache.txt"
    if (Test-Path $cachePath) {
        $shouldWipe = $false
        $wipeReason = $null
        $cachedGeneratorLine = Select-String -Path $cachePath -Pattern '^CMAKE_GENERATOR:INTERNAL=' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cachedGeneratorLine) {
            $cachedGenerator = ($cachedGeneratorLine.Line -split "=", 2)[1].Trim()
            if ($cachedGenerator -and ($cachedGenerator -ne $generator)) {
                $shouldWipe = $true
                $wipeReason = "generator changed from '$cachedGenerator' to '$generator'"
            }
        }
        if (-not $shouldWipe -and $cudaIntegrationJustCopied) {
            $shouldWipe = $true
            $wipeReason = "CUDA VS integration was just installed"
        }
        if ($shouldWipe) {
            Write-Host "==> wiping build\ ($wipeReason)"
            Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
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
