# Shared Windows toolchain helpers for CMake-based builders
# (build-llama-turbo.ps1, build-sdcpp.ps1, ...).
#
# Functions:
#   Resolve-CmakeWindowsBuildContext  -- pick a generator and probe VS
#   Sync-CudaVsIntegration            -- copy CUDA's MSBuild .props/.targets
#                                        into the VS BuildCustomizations dir
#   Get-CmakeWindowsConfigureArgs     -- expand generator/instance into -G ... flags
#   Invoke-CmakeStaleCacheWipe        -- nuke build/ when its cache is stale
#
# All four are no-ops on non-Windows (the .sh scripts call native cmake
# directly without needing this layer), so dot-sourcing is safe to gate
# behind ``$IsWindows``.

function Resolve-CmakeWindowsBuildContext {
    <#
    .SYNOPSIS
    Pick a CMake generator and locate a working VS install.

    .DESCRIPTION
    Without -G, cmake defaults to "NMake Makefiles" on Windows, which
    fails outside a Developer Command Prompt. Probe in this order:
      1. -GeneratorEnv override (e.g. CHAOSENGINE_LLAMA_TURBO_GENERATOR)
      2. Ninja, when on PATH
      3. "Visual Studio 17 2022"

    For the Visual Studio path, locate cl.exe via vswhere with -all so
    isComplete=0 installs (Microsoft's installer flagging optional
    components as missing) are still accepted. Pass the install path
    AND its version back so the caller can hand them to CMake via
    CMAKE_GENERATOR_INSTANCE -- otherwise CMake re-runs its own -latest
    probe and rejects the same install with "instance is not known to
    the Visual Studio Installer".

    .PARAMETER ProductLabel
    Short label for the binary being built (e.g. "llama-server-turbo")
    used in the "install Visual Studio" error message.

    .PARAMETER GeneratorEnv
    Name of an environment variable that overrides generator selection
    (e.g. "CHAOSENGINE_LLAMA_TURBO_GENERATOR").
    #>
    param(
        [Parameter(Mandatory)] [string] $ProductLabel,
        [Parameter(Mandatory)] [string] $GeneratorEnv
    )

    $generator = $null
    $envOverride = (Get-Item "env:$GeneratorEnv" -ErrorAction SilentlyContinue).Value
    if ($envOverride) {
        $generator = $envOverride
    } elseif (Get-Command ninja -ErrorAction SilentlyContinue) {
        $generator = "Ninja"
    } else {
        $generator = "Visual Studio 17 2022"
    }

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
            $clExe = $clCandidates | Sort-Object -Descending | Select-Object -First 1
            # Walk up from <root>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe
            # to <root>: 8 segments to strip (x64, Hostx64, bin, <ver>,
            # MSVC, Tools, VC, cl.exe-the-leaf-itself).
            $vsInstance = $clExe
            for ($i = 0; $i -lt 8; $i++) { $vsInstance = Split-Path -Parent $vsInstance }
            $matchedInstall = $vsInstalls | Where-Object {
                $_.installationPath.TrimEnd('\') -eq $vsInstance.TrimEnd('\')
            } | Select-Object -First 1
            if ($matchedInstall) {
                $vsInstanceVersion = $matchedInstall.installationVersion
            }
            Write-Host "==> Visual Studio detected at: $vsInstance"
            if ($vsInstanceVersion) { Write-Host "    version: $vsInstanceVersion" }
            Write-Host "    cl.exe:  $clExe"
        } else {
            $msg = @(
                "",
                "Visual Studio 2022 with the C++ workload is not installed.",
                "$ProductLabel cannot build without an MSVC toolchain --",
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

    return [pscustomobject]@{
        Generator         = $generator
        VsInstance        = $vsInstance
        VsInstanceVersion = $vsInstanceVersion
    }
}

function Sync-CudaVsIntegration {
    <#
    .SYNOPSIS
    Copy CUDA's MSBuild integration files into the VS BuildCustomizations dir.

    .DESCRIPTION
    CMake's CUDA detection bails with "No CUDA toolset found" when these
    files are missing -- which happens whenever CUDA was installed
    before Visual Studio, or when the CUDA installer's "Visual Studio
    Integration" component was unticked. Auto-elevates via UAC if the
    target dir isn't writable.

    Returns $true when files were actually copied (caller should wipe
    build/CMakeCache.txt so CMake re-detects), $false when up to date
    or skipped.
    #>
    param(
        [Parameter(Mandatory)] [string] $VsRoot
    )
    $cudaPath = $env:CUDA_PATH
    if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
        Write-Host "==> CUDA_PATH not set; skipping VS integration sync"
        return $false
    }
    $cudaSrc = Join-Path $cudaPath "extras\visual_studio_integration\MSBuildExtensions"
    $vsTarget = Join-Path $VsRoot "MSBuild\Microsoft\VC\v170\BuildCustomizations"
    if (-not (Test-Path $cudaSrc)) {
        Write-Host "==> CUDA integration source not found at $cudaSrc; skipping sync"
        return $false
    }
    if (-not (Test-Path $vsTarget)) {
        Write-Host "==> VS BuildCustomizations dir not found at $vsTarget; skipping sync"
        return $false
    }
    $sourceFiles = Get-ChildItem -Path $cudaSrc -File -ErrorAction SilentlyContinue
    $missing = @($sourceFiles | Where-Object { -not (Test-Path (Join-Path $vsTarget $_.Name)) })
    if (-not $missing -or $missing.Count -eq 0) {
        Write-Host "==> CUDA VS integration already present in $vsTarget"
        return $false
    }
    Write-Host "==> CUDA VS integration missing $($missing.Count) file(s) from $vsTarget"
    $missing | ForEach-Object { Write-Host "    - $($_.Name)" }

    $copied = $true
    try {
        foreach ($file in $missing) {
            Copy-Item -LiteralPath $file.FullName -Destination $vsTarget -Force -ErrorAction Stop
        }
        Write-Host "==> CUDA VS integration files copied (direct)"
    } catch {
        $copied = $false
        Write-Host "==> Direct copy denied; relaunching as admin via UAC..."
        # Per-file Copy-Item: -LiteralPath does NOT support wildcards, so
        # an "...\*" pattern silently copies nothing. Iterate by full path
        # and verify each file lands.
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

function Get-CmakeWindowsConfigureArgs {
    <#
    .SYNOPSIS
    Expand a build context into -G/-A/-DCMAKE_GENERATOR_INSTANCE flags.
    #>
    param(
        [Parameter(Mandatory)] $Context,
        [string[]] $ExtraFlags = @()
    )
    $args = @("-B", "build", "-G", $Context.Generator)
    if ($Context.Generator -like "Visual Studio*") {
        $args += @("-A", "x64")
        if ($Context.VsInstance) {
            $instanceArg = if ($Context.VsInstanceVersion) {
                "$($Context.VsInstance),version=$($Context.VsInstanceVersion)"
            } else {
                $Context.VsInstance
            }
            $args += @("-DCMAKE_GENERATOR_INSTANCE=$instanceArg")
        }
    }
    return $args + $ExtraFlags
}

function Invoke-CmakeStaleCacheWipe {
    <#
    .SYNOPSIS
    Wipe build/ when the cached generator no longer matches, or when
    CUDA integration was just installed.

    .DESCRIPTION
    CMake refuses to switch generators in an existing build directory
    ("Does not match the generator used previously"). And it caches
    CUDA-language detection results, so installing the integration
    files between runs doesn't get re-evaluated unless we wipe.

    Pattern detail: do NOT use -SimpleMatch on the regex -- it disables
    regex parsing, making the leading ^ a literal character, and the
    cache line never matches.
    #>
    param(
        [Parameter(Mandatory)] [string] $Generator,
        [bool] $CudaIntegrationJustCopied = $false
    )
    $cachePath = "build\CMakeCache.txt"
    if (-not (Test-Path $cachePath)) { return }

    $shouldWipe = $false
    $wipeReason = $null
    $cachedGeneratorLine = Select-String -Path $cachePath `
        -Pattern '^CMAKE_GENERATOR:INTERNAL=' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($cachedGeneratorLine) {
        $cachedGenerator = ($cachedGeneratorLine.Line -split "=", 2)[1].Trim()
        if ($cachedGenerator -and ($cachedGenerator -ne $Generator)) {
            $shouldWipe = $true
            $wipeReason = "generator changed from '$cachedGenerator' to '$Generator'"
        }
    }
    if (-not $shouldWipe -and $CudaIntegrationJustCopied) {
        $shouldWipe = $true
        $wipeReason = "CUDA VS integration was just installed"
    }
    if ($shouldWipe) {
        Write-Host "==> wiping build\ ($wipeReason)"
        Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
    }
}
