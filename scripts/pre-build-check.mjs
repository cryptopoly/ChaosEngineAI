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
 *    missing, vendor/ChaosEngine submodule behind upstream)
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
console.log("[1/7] Python tests...");
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
console.log("[2/7] TypeScript tests...");
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
console.log("[3/7] TypeScript type checking...");
{
  const result = run(npxCommand(), ["tsc", "--noEmit"]);
  if (result.ok) pass("TypeScript types");
  else fail("TypeScript type errors — see output above");
}
console.log();

// ------------------------------------------------------------------
// 4. Licence notices
// ------------------------------------------------------------------
console.log("[4/7] Licence notices...");
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
      "chaosengine",
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
console.log("[5/7] Cache strategy validation...");
{
  const probe = `
from cache_compression import registry
registry.discover()
valid = {'f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'}
ce = registry.get('chaosengine')
for bits in (2,3,4,5,6,8):
    flags = ce.llama_cpp_cache_flags(bits)
    for i, f in enumerate(flags):
        if f.startswith('--cache-type-') and i+1 < len(flags):
            if flags[i+1] not in valid:
                print(f'INVALID: ChaosEngine {bits}-bit emits {flags[i+1]}')
rq = registry.get('rotorquant')
tq = registry.get('turboquant')
if rq.required_llama_binary() != 'turbo':
    print('INVALID: RotorQuant not routing to turbo')
if tq.required_llama_binary() != 'turbo':
    print('INVALID: TurboQuant not routing to turbo')
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
console.log("[6/7] Upstream dependency check...");
{
  // Turbo fork: read version file (commit | branch | build_date) +
  // compare to live HEAD via git ls-remote. Mirrors the .sh exactly,
  // including the johndpope fork URL it tracks.
  const turboVersionFile = path.join(homedir(), ".chaosengine", "bin", "llama-server-turbo.version");
  if (existsSync(turboVersionFile)) {
    const localCommit = readFileSync(turboVersionFile, "utf8").split(/\r?\n/)[0]?.trim() ?? "";
    const lsRemote = capture("git", [
      "ls-remote",
      "https://github.com/johndpope/llama-cpp-turboquant.git",
      "refs/heads/feature/planarquant-kv-cache",
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

  // ChaosEngine vendor submodule: check commits behind origin/main.
  const vendorGit = path.join(REPO_ROOT, "vendor", "ChaosEngine", ".git");
  if (existsSync(vendorGit)) {
    const behind = capture("git", ["-C", "vendor/ChaosEngine", "rev-list", "HEAD..origin/main", "--count"]);
    if (behind.ok) {
      const count = behind.stdout.trim();
      if (count === "0") {
        pass("vendor/ChaosEngine — up to date");
      } else {
        warn(`vendor/ChaosEngine — ${count} commits behind upstream`);
      }
    } else {
      warn("vendor/ChaosEngine — could not check (fetch first)");
    }
  }
}
console.log();

// ------------------------------------------------------------------
// 7. Binary availability
// ------------------------------------------------------------------
console.log("[7/7] Binary availability...");
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
