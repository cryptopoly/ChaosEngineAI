#!/usr/bin/env node
/**
 * Pre-build quality gate for ChaosEngineAI.
 *
 * Cross-platform port of pre-build-check.sh. Runs the same 7 checks
 * (Python tests, TS tests, tsc, NOTICES grep, cache-strategy probe,
 * upstream commit comparison, binary availability) on macOS, Linux,
 * and Windows by leaning on Node's stdlib + spawn().
 *
 * Usage:
 *   node scripts/pre-build-check.mjs
 *   npm run pre-build-check         # if wired into package.json
 *
 * Exit code 0 = all gates passed, 1 = at least one failure.
 *
 * Behaviour parity with the .sh version:
 *  - PASS / FAIL / WARN per check, summary at the end
 *  - FAIL is blocking; WARN is informational (e.g. turbo binary
 *    missing)
 *  - Output streams live so CI logs show progress without buffering
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { homedir, platform } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");

const results = [];
let passCount = 0;
let failCount = 0;
let warnCount = 0;

function pass(label) {
  passCount += 1;
  results.push(["PASS", label]);
}
function fail(label) {
  failCount += 1;
  results.push(["FAIL", label]);
}
function warn(label) {
  warnCount += 1;
  results.push(["WARN", label]);
}

/**
 * Run a command, stream stdout/stderr to the parent, return exit code.
 * `cmd` is the executable, `args` is an array. Cross-platform — Node's
 * child_process.spawnSync handles the .exe lookup on Windows when
 * `shell: true` is set, but we explicitly resolve venv Python below to
 * avoid PATH ambiguity.
 */
function run(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, {
    stdio: "inherit",
    cwd: options.cwd ?? REPO_ROOT,
    env: { ...process.env, ...(options.env ?? {}) },
    shell: false,
  });
  if (result.error) {
    return { ok: false, code: -1, error: result.error };
  }
  return { ok: result.status === 0, code: result.status ?? -1 };
}

/**
 * Capture-mode invocation — used by the cache-strategy probe + upstream
 * commit checks where we need to inspect the output. Returns
 * { ok, stdout, stderr, code }.
 */
function capture(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, {
    cwd: options.cwd ?? REPO_ROOT,
    env: { ...process.env, ...(options.env ?? {}) },
    encoding: "utf8",
    timeout: options.timeout ?? 30_000,
  });
  return {
    ok: !result.error && result.status === 0,
    code: result.status ?? -1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

/** Locate the Python interpreter — venv first, then PATH. */
function venvPython() {
  const isWin = platform() === "win32";
  const candidates = isWin
    ? [
        path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
        path.join(REPO_ROOT, ".venv", "bin", "python.exe"),
      ]
    : [
        path.join(REPO_ROOT, ".venv", "bin", "python"),
        path.join(REPO_ROOT, ".venv", "bin", "python3"),
      ];
  return candidates.find((p) => existsSync(p)) ?? (isWin ? "python" : "python3");
}

/** Locate npm. On Windows it's npm.cmd; on Unix it's npm. */
function npmCommand() {
  return platform() === "win32" ? "npm.cmd" : "npm";
}

/** Locate npx. Same shape as npm. */
function npxCommand() {
  return platform() === "win32" ? "npx.cmd" : "npx";
}

console.log("=== ChaosEngineAI Pre-Build Checks ===\n");

// ------------------------------------------------------------------
// 1. Python tests
// ------------------------------------------------------------------
console.log("[1/8] Python tests...");
{
  const py = venvPython();
  const result = run(py, ["-m", "pytest", "tests/", "-q", "--tb=line"]);
  if (result.ok) pass("Python tests");
  else fail("Python tests — see output above");
}
console.log();

// ------------------------------------------------------------------
// 2. TypeScript tests
// ------------------------------------------------------------------
console.log("[2/8] TypeScript tests...");
{
  // The vitest config defaults to watch mode in TTY; --run forces a
  // single CI-friendly pass.
  const result = run(npmCommand(), ["test", "--", "--run"]);
  if (result.ok) pass("TypeScript tests");
  else fail("TypeScript tests — see output above");
}
console.log();

// ------------------------------------------------------------------
// 3. TypeScript type checking
// ------------------------------------------------------------------
console.log("[3/8] TypeScript type checking...");
{
  const result = run(npxCommand(), ["tsc", "--noEmit"]);
  if (result.ok) pass("TypeScript types");
  else fail("TypeScript type errors — see output above");
}
console.log();

// ------------------------------------------------------------------
// 4. Licence notices
// ------------------------------------------------------------------
console.log("[4/8] Licence notices...");
{
  const noticesPath = path.join(REPO_ROOT, "THIRD_PARTY_NOTICES.md");
  if (!existsSync(noticesPath) || statSync(noticesPath).size === 0) {
    fail("THIRD_PARTY_NOTICES.md missing or empty");
  } else {
    const content = readFileSync(noticesPath, "utf8").toLowerCase();
    const required = [
      "llama.cpp",
      "llama-cpp-turboquant",
      "dflash-mlx",
      "turboquant",
    ];
    const missing = required.filter((dep) => !content.includes(dep.toLowerCase()));
    if (missing.length === 0) {
      pass("THIRD_PARTY_NOTICES.md — all key deps listed");
    } else {
      warn(`THIRD_PARTY_NOTICES.md — missing: ${missing.join(", ")}`);
    }
  }
}
console.log();

// ------------------------------------------------------------------
// 5. Cache strategy validation
// ------------------------------------------------------------------
console.log("[5/8] Cache strategy validation...");
{
  const probe = `
from cache_compression import registry
registry.discover()
valid = {'f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'}
nat = registry.get('native')
for bits in (0,):
    flags = nat.llama_cpp_cache_flags(bits)
    for i, f in enumerate(flags):
        if f.startswith('--cache-type-') and i+1 < len(flags):
            if flags[i+1] not in valid:
                print(f'INVALID: Native emits {flags[i+1]}')
tq = registry.get('turboquant')
if tq.required_llama_binary() != 'turbo':
    print('INVALID: TurboQuant not routing to turbo')
# FU-030 (2026-05-10): rotorquant + chaosengine were dropped. Their ids
# must coerce to turboquant via the legacy alias map; assert that here so
# regressions surface in CI rather than at runtime.
for legacy_id in ('rotorquant', 'chaosengine'):
    coerced = registry.resolve_legacy_id(legacy_id)
    if coerced != 'turboquant':
        print(f'INVALID: legacy id {legacy_id} did not coerce to turboquant (got {coerced})')
    if registry.get(legacy_id) is None:
        print(f'INVALID: legacy id {legacy_id} did not resolve via registry.get')
print('OK')
`.trim();
  const result = capture(venvPython(), ["-c", probe]);
  const out = `${result.stdout}\n${result.stderr}`;
  if (out.includes("INVALID")) {
    fail(`Cache strategy validation: ${out.trim()}`);
  } else if (!result.ok) {
    fail(`Cache strategy probe crashed (exit ${result.code}): ${out.trim()}`);
  } else {
    pass("Cache strategy validation");
  }
}
console.log();

// ------------------------------------------------------------------
// 6. Upstream dependency update check
// ------------------------------------------------------------------
console.log("[6/8] Upstream dependency check...");
{
  // Turbo fork: read version file (commit | branch | build_date) +
  // compare to live HEAD via git ls-remote. Mirrors the .sh exactly,
  // including the johndpope fork URL it tracks.
  const turboVersionFile = path.join(homedir(), ".chaosengine", "bin", "llama-server-turbo.version");
  if (existsSync(turboVersionFile)) {
    const localCommit = readFileSync(turboVersionFile, "utf8").split(/\r?\n/)[0]?.trim() ?? "";
    const lsRemote = capture("git", [
      "ls-remote",
      "https://github.com/TheTom/llama-cpp-turboquant.git",
      "refs/heads/feature/turboquant-kv-cache",
    ]);
    const remoteCommit = lsRemote.stdout.split(/\s+/)[0]?.trim() ?? "";
    if (remoteCommit && localCommit !== remoteCommit) {
      warn(`llama-server-turbo update available (local: ${localCommit.slice(0, 12)}, remote: ${remoteCommit.slice(0, 12)})`);
    } else if (remoteCommit) {
      pass("llama-server-turbo — up to date");
    } else {
      warn("llama-server-turbo — could not reach upstream");
    }
  } else {
    const buildScript = platform() === "win32" ? "scripts\\build-llama-turbo.ps1" : "scripts/build-llama-turbo.sh";
    warn(`llama-server-turbo — not installed (run ${buildScript})`);
  }

  // FU-030 dropped vendor/ChaosEngine; the staleness probe that lived
  // here used to walk ``vendor/ChaosEngine/.git`` and warn on commits
  // behind upstream. Removed alongside the vendored package.

  // FU-033: dflash-mlx pin sync. The ``[dflash-mlx]`` extra in
  // pyproject.toml and the ``stageOptionalRuntimePackages`` entry in
  // scripts/stage-runtime.mjs both pin to a specific git commit. They
  // drifted in May 2026 — pyproject was at v0.1.5.1 (8d8545d) while
  // stage-runtime still bundled v0.1.4.1 (f825ffb), shipping an old
  // binary in release builds even when the dev .venv ran new. Catch
  // future drift here rather than at first ``npm run stage:runtime``.
  const pinRe = /dflash-mlx\.git@([a-f0-9]+)/;
  const pyprojectPath = path.join(REPO_ROOT, "pyproject.toml");
  const stageRuntimePath = path.join(REPO_ROOT, "scripts", "stage-runtime.mjs");
  const pyprojectMatch = readFileSync(pyprojectPath, "utf8").match(pinRe);
  const stageMatch = readFileSync(stageRuntimePath, "utf8").match(pinRe);
  if (!pyprojectMatch || !stageMatch) {
    warn("dflash-mlx pin sync — could not extract commit hashes from both files");
  } else if (pyprojectMatch[1] !== stageMatch[1]) {
    fail(
      `dflash-mlx pin drift — pyproject.toml=${pyprojectMatch[1].slice(0, 12)} ` +
        `stage-runtime.mjs=${stageMatch[1].slice(0, 12)}. ` +
        `Sync both to the same commit to avoid release-build regressions.`,
    );
  } else {
    pass(`dflash-mlx pin sync (${pyprojectMatch[1].slice(0, 12)})`);
  }

  // App version sync across the 4 manifests. The v0.9.0 release shipped
  // with pyproject.toml at 0.8.0 because nothing enforced cross-file
  // sync — users saw "Latest release · v0.9.0" on the site but the
  // bundled backend reported appVersion 0.8.0. Pre-build gate now
  // pins all four sources to the same string.
  const versionSources = [
    {
      label: "package.json",
      path: path.join(REPO_ROOT, "package.json"),
      re: /"version"\s*:\s*"([^"]+)"/,
    },
    {
      label: "pyproject.toml",
      path: path.join(REPO_ROOT, "pyproject.toml"),
      re: /^\s*version\s*=\s*"([^"]+)"/m,
    },
    {
      label: "src-tauri/Cargo.toml",
      path: path.join(REPO_ROOT, "src-tauri", "Cargo.toml"),
      re: /^\s*version\s*=\s*"([^"]+)"/m,
    },
    {
      label: "src-tauri/tauri.conf.json",
      path: path.join(REPO_ROOT, "src-tauri", "tauri.conf.json"),
      re: /"version"\s*:\s*"([^"]+)"/,
    },
  ];
  const versions = versionSources.map((s) => {
    const text = readFileSync(s.path, "utf8");
    const m = text.match(s.re);
    return { label: s.label, version: m ? m[1] : null };
  });
  const distinct = [...new Set(versions.map((v) => v.version))];
  if (versions.some((v) => v.version === null)) {
    warn("app version sync — could not extract version from one or more manifests");
  } else if (distinct.length > 1) {
    const detail = versions.map((v) => `${v.label}=${v.version}`).join(" ");
    fail(`app version drift across manifests — ${detail}. Bump all four to the same string.`);
  } else {
    pass(`app version sync (${distinct[0]})`);
  }
}
console.log();

// ------------------------------------------------------------------
// 7. Binary availability
// ------------------------------------------------------------------
console.log("[7/8] Binary availability...");
{
  const isWin = platform() === "win32";
  const exeSuffix = isWin ? ".exe" : "";
  const standardCandidates = isWin
    ? [path.join(homedir(), ".chaosengine", "bin", `llama-server${exeSuffix}`)]
    : [
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
        path.join(homedir(), ".chaosengine", "bin", "llama-server"),
      ];

  // PATH lookup as the fallback so brew/apt installs are still picked up.
  const onPath = capture(isWin ? "where" : "which", ["llama-server"]);
  const standardFound =
    onPath.ok || standardCandidates.some((p) => existsSync(p));
  if (standardFound) pass("llama-server (standard) — found");
  else warn("llama-server (standard) — not found");

  const turboPath = path.join(homedir(), ".chaosengine", "bin", `llama-server-turbo${exeSuffix}`);
  if (existsSync(turboPath)) {
    pass("llama-server-turbo — found");
  } else {
    warn("llama-server-turbo — not found (RotorQuant/TurboQuant GGUF will fall back to f16)");
  }
}
console.log();

// ------------------------------------------------------------------
// 8. i18n locale validation (FU-042)
// ------------------------------------------------------------------
// Warn-only by default; --strict upgrades to fail-on-coverage <95 %.
console.log("[8/8] i18n locale validation...");
{
  const result = run("node", ["scripts/i18n-validate.mjs"]);
  if (result.ok) pass("i18n locale catalogs");
  else fail("i18n locale catalogs — see output above");
}
console.log();

// ------------------------------------------------------------------
// Summary
// ------------------------------------------------------------------
console.log("=== Summary ===");
for (const [tag, label] of results) {
  console.log(`  ${tag.padEnd(4)}  ${label}`);
}
console.log();
console.log(`  ${passCount} passed, ${failCount} failed, ${warnCount} warnings`);
console.log();

if (failCount > 0) {
  console.log("BUILD BLOCKED — fix failures above before shipping.");
  process.exit(1);
} else {
  console.log("All gates passed.");
  process.exit(0);
}
