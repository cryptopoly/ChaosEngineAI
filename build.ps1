$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ── Pre-flight cleanup ───────────────────────────────────
# Windows Defender / Explorer / pip occasionally hold file handles on the
# staging tree. Clear it up front so the Node rmSync in stage-runtime.mjs
# never has to fight with locked artefacts from a prior failed run.
if (Test-Path .runtime-stage) {
    Write-Host "==> Clearing stale .runtime-stage..."
    Remove-Item -Recurse -Force .runtime-stage -ErrorAction SilentlyContinue
}

# ── Python venv ──────────────────────────────────────────
if (-not (Test-Path .venv)) {
    Write-Host "==> Creating Python venv..."
    python -m venv .venv
}

Write-Host "==> Installing Python dependencies..."
.\.venv\Scripts\pip install --upgrade pip -q
# Install the same extras that stage-runtime.mjs::validateBundledPythonPackages
# checks for (desktop + images). Without `images` a release build fails strict
# validation; in dev mode it merely warns.
.\.venv\Scripts\pip install -q -e ".[desktop,images]"

$env:CHAOSENGINE_EMBED_PYTHON_BIN = "$ScriptDir\.venv\Scripts\python.exe"

# ── npm dependencies ─────────────────────────────────────
Write-Host "==> Installing npm dependencies..."
npm ci --silent

# ── llama.cpp pre-flight ─────────────────────────────────
# Release-mode staging is strict: if llama.cpp isn't built locally the install
# will refuse to ship without inference. Detect upfront so the user gets a
# clear message before npm/cargo spend 20 minutes building, and let them opt
# into shipping a diffusers-only installer with one env var.
$llamaBinDir = if ($env:CHAOSENGINE_LLAMA_BIN_DIR) {
    $env:CHAOSENGINE_LLAMA_BIN_DIR
} else {
    Join-Path (Split-Path -Parent $ScriptDir) "llama.cpp\build\bin"
}
$llamaServerExe = Join-Path $llamaBinDir "llama-server.exe"
if (-not (Test-Path $llamaServerExe)) {
    if ($env:CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA -eq "1") {
        Write-Warning "llama-server.exe not found at $llamaServerExe — proceeding because CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA=1."
        Write-Warning "The installer will ship without llama.cpp; users can install it via the Setup page."
    } else {
        Write-Host ""
        Write-Host "==> ERROR: llama-server.exe not found at $llamaServerExe" -ForegroundColor Red
        Write-Host ""
        Write-Host "    A release build needs llama.cpp compiled locally so the bundled"
        Write-Host "    installer ships with native inference support. Pick one:"
        Write-Host ""
        # Use '; ' instead of ' && ' — Windows PowerShell 5.1's parser
        # chokes on '&&' as an invalid statement separator even when it
        # appears inside a double-quoted string literal. '; ' is portable
        # shell syntax that works everywhere we care about.
        Write-Host "    1. Build llama.cpp at ..\llama.cpp\ (cmake -B build; cmake --build build)"
        Write-Host "    2. Set `$env:CHAOSENGINE_LLAMA_BIN_DIR to your llama.cpp build directory"
        Write-Host "    3. Ship without it: `$env:CHAOSENGINE_RELEASE_ALLOW_NO_LLAMA = `"1`""
        Write-Host ""
        throw "llama-server.exe missing — see message above."
    }
}

# ── Patch tauri.conf.json for local builds ───────────────
# IMPORTANT: use stage:runtime:release (not stage:runtime). The dev variant
# writes ``mode=development`` into the manifest AND skips building the
# tar.gz runtime archive — both of which break the shipped installer:
#   - Tauri (lib.rs::resolve_embedded_runtime) sees mode=development and
#     looks for a live source workspace at the customer's install path,
#     which doesn't exist. Result: the backend never boots.
#   - Without the tar.gz, the Python runtime + backend code are simply
#     not in the installer. The result is a tiny (~3 MB) installer that
#     can't actually run anything.
Write-Host "==> Patching tauri.conf.json for local build..."
node -e "const fs = require('fs'); const conf = JSON.parse(fs.readFileSync('src-tauri/tauri.conf.json', 'utf8')); conf.build.beforeBundleCommand = 'npm run stage:runtime:release'; conf.bundle = conf.bundle || {}; conf.bundle.createUpdaterArtifacts = false; fs.writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(conf, null, 2) + '\n');"

# ── Build ────────────────────────────────────────────────
Write-Host "==> Building Tauri app (NSIS installer)..."
npx tauri build --bundles nsis

# ── Restore tauri.conf.json ──────────────────────────────
# ``git checkout`` writes "Updated 1 path from the index" to stderr even on
# success. With $ErrorActionPreference = "Stop" set at the top of this
# script, the ``2>&1 | Out-Null`` pattern wraps that stderr text as a
# terminating NativeCommandError — crashing the build *after* the installer
# is already produced. ``--quiet`` silences the success message and
# ``2>$null`` discards anything else by file redirect (which bypasses
# PowerShell's stream-wrapping logic). We still gate on $LASTEXITCODE so a
# real git failure (working tree dirty etc.) surfaces as a build error.
git checkout --quiet src-tauri/tauri.conf.json 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to restore src-tauri/tauri.conf.json (git exit $LASTEXITCODE)"
}

Write-Host ""
Write-Host "==> Build complete!"
Write-Host "    Artifacts: src-tauri\target\release\bundle\nsis\"
