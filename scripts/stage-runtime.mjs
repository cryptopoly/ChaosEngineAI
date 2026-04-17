#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadEnvFiles } from "./load-env.mjs";

const modeArg = process.argv.find((value) => value.startsWith("--mode="));
const mode = modeArg ? modeArg.split("=", 2)[1] : "development";
const strict = mode === "release";

const scriptRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptRoot, "..");
const workspaceRoot = projectRoot;
const tauriRoot = path.join(projectRoot, "src-tauri");
const resourcesRoot = path.join(tauriRoot, "resources");
const embeddedResourcesRoot = path.join(resourcesRoot, "embedded");
const stagingRoot = path.join(projectRoot, ".runtime-stage");

loadEnvFiles([
  path.join(projectRoot, ".env"),
  path.join(projectRoot, ".env.local"),
]);
const signingIdentity = process.env.CHAOSENGINE_APPLE_SIGNING_IDENTITY || process.env.APPLE_SIGNING_IDENTITY || "";

const platformTag = resolvePlatformTag();
const stageRoot = path.join(stagingRoot, platformTag);
const pythonStageRoot = path.join(stageRoot, "python");
const pythonHomeDest = path.join(pythonStageRoot, "home");
const sitePackagesDest = path.join(pythonStageRoot, "site-packages");
const backendDest = path.join(stageRoot, "backend");
const binDest = path.join(stageRoot, "bin");
const manifestDest = path.join(embeddedResourcesRoot, `runtime-${platformTag}.manifest.json`);
const archiveDest = path.join(embeddedResourcesRoot, `runtime-${platformTag}.tar.gz`);

main();

function main() {
  ensureDir(embeddedResourcesRoot);
  cleanupStaleTauriResources();
  fs.rmSync(path.join(resourcesRoot, "runtime"), { recursive: true, force: true });
  fs.rmSync(stageRoot, { recursive: true, force: true });
  cleanupStagedRuntimeArtifacts();
  ensureDir(stageRoot);

  const pythonInfo = inspectPython();
  validateBundledPythonPackages(pythonInfo.executable);
  const pythonHomeSource = resolveExistingPath(
    process.env.CHAOSENGINE_EMBED_PYTHON_HOME || pythonInfo.basePrefix,
    "embedded Python home",
  );
  const sitePackagesSource = resolveExistingPath(
    process.env.CHAOSENGINE_EMBED_SITE_PACKAGES || pythonInfo.sitePackages,
    "embedded site-packages",
  );

  copyTree(pythonHomeSource, pythonHomeDest);
  copyTree(sitePackagesSource, sitePackagesDest);
  pruneBundledProjectArtifacts();

  ensureDir(backendDest);
  for (const relativePath of ["backend_service", "compression"]) {
    copyTree(path.join(workspaceRoot, relativePath), path.join(backendDest, relativePath));
  }
  for (const relativeFile of ["README.md", "pyproject.toml"]) {
    copyFile(path.join(workspaceRoot, relativeFile), path.join(backendDest, relativeFile));
  }

  const chaosEngineBundle = stageVendoredChaosEngine(pythonInfo.executable);
  const bundledOptionalPackages = stageOptionalRuntimePackages(pythonInfo.executable);
  validateBundledProjectImports(pythonInfo.executable);
  const llamaWarnings = stageLlamaBinaries();
  maybeSignEmbeddedRuntime();
  const pythonBinaryRelative = resolveEmbeddedPythonBinary(pythonInfo.versionTag);
  const embeddedRuntime = {
    platformTag,
    mode,
    buildStamp: new Date().toISOString(),
    backendRoot: "backend",
    pythonBinary: path.join("python", "home", pythonBinaryRelative).split(path.sep).join("/"),
    pythonHome: path.join("python", "home").split(path.sep).join("/"),
    pythonPath: ["backend", "python/site-packages"],
    libraryPathEntries: ["bin", "python/home/lib"],
    pathEntries: ["python/home/bin", "bin"],
    llamaServer: fs.existsSync(path.join(binDest, binaryName("llama-server"))) ? `bin/${binaryName("llama-server")}` : null,
    llamaServerTurbo: fs.existsSync(path.join(binDest, binaryName("llama-server-turbo"))) ? `bin/${binaryName("llama-server-turbo")}` : null,
    llamaCli: fs.existsSync(path.join(binDest, binaryName("llama-cli"))) ? `bin/${binaryName("llama-cli")}` : null,
    pythonVersion: pythonInfo.versionTag,
    bundledCacheStrategies: chaosEngineBundle ? ["chaosengine"] : [],
    bundledOptionalPackages: bundledOptionalPackages,
    warnings: llamaWarnings,
  };

  fs.writeFileSync(path.join(stageRoot, "manifest.json"), JSON.stringify(embeddedRuntime, null, 2));
  fs.copyFileSync(path.join(stageRoot, "manifest.json"), manifestDest);
  execFileSync("tar", ["-czf", archiveDest, "-C", stageRoot, "."], {
    cwd: workspaceRoot,
    env: { ...process.env, COPYFILE_DISABLE: "1" },
    stdio: "inherit",
  });
  fs.rmSync(stageRoot, { recursive: true, force: true });

  console.log(`[stage-runtime] mode=${mode}`);
  console.log(`[stage-runtime] platform=${platformTag}`);
  console.log(`[stage-runtime] archive -> ${archiveDest}`);
  console.log(`[stage-runtime] manifest -> ${manifestDest}`);
  if (llamaWarnings.length) {
    for (const warning of llamaWarnings) {
      console.warn(`[stage-runtime] warning: ${warning}`);
    }
  }
}

function resolvePlatformTag() {
  const platform = process.env.TAURI_ENV_PLATFORM || process.platform;
  const arch = normalizeArch(process.env.TAURI_ENV_ARCH || process.arch);
  const normalizedPlatform =
    platform === "macos" || platform === "darwin"
      ? "darwin"
      : platform === "win32"
        ? "windows"
        : platform;
  return `${normalizedPlatform}-${arch}`;
}

function normalizeArch(arch) {
  if (arch === "arm64") {
    return "aarch64";
  }
  if (arch === "x64") {
    return "x86_64";
  }
  return arch;
}

function inspectPython() {
  const pythonBinary =
    process.env.CHAOSENGINE_EMBED_PYTHON_BIN ||
    (process.platform === "win32"
      ? path.join(workspaceRoot, ".venv", "Scripts", "python.exe")
      : path.join(workspaceRoot, ".venv", "bin", "python3"));
  assertPathExists(pythonBinary, "Python executable");
  const script = [
    "import json, sys, sysconfig",
    "payload = {",
    "  'executable': sys.executable,",
    "  'prefix': sys.prefix,",
    "  'basePrefix': sys.base_prefix,",
    "  'sitePackages': sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib'),",
    "  'versionTag': f\"{sys.version_info.major}.{sys.version_info.minor}\",",
    "}",
    "print(json.dumps(payload))",
  ].join("\n");

  const payload = execFileSync(pythonBinary, ["-c", script], {
    cwd: workspaceRoot,
    encoding: "utf8",
  }).trim();
  const parsed = JSON.parse(payload);
  if (!parsed.basePrefix || !parsed.sitePackages || !parsed.versionTag) {
    throw new Error("Could not inspect the embedded Python layout.");
  }
  return parsed;
}

function validateBundledPythonPackages(pythonBinary) {
  const script = [
    "import importlib.util, json",
    "requirements = {",
    "  'desktop': [('fastapi', 'fastapi'), ('huggingface_hub', 'huggingface_hub'), ('psutil', 'psutil'), ('pypdf', 'pypdf'), ('uvicorn', 'uvicorn')],",
    "  'images': [('accelerate', 'accelerate'), ('diffusers', 'diffusers'), ('huggingface_hub', 'huggingface_hub'), ('PIL', 'pillow'), ('safetensors', 'safetensors'), ('torch', 'torch')],",
    "  'inference': [('dflash_mlx', 'dflash-mlx'), ('turboquant', 'turboquant'), ('turboquant_mlx', 'turboquant-mlx-full')],",
    "}",
    "missing = {",
    "  group: [label for module, label in modules if importlib.util.find_spec(module) is None]",
    "  for group, modules in requirements.items()",
    "}",
    "print(json.dumps(missing))",
  ].join("\n");

  const payload = execFileSync(pythonBinary, ["-c", script], {
    cwd: workspaceRoot,
    encoding: "utf8",
  }).trim();
  const missing = JSON.parse(payload);

  // Inference packages are optional — warn but never block the build.
  const optionalGroups = new Set(["inference"]);
  const requiredMissing = Object.entries(missing)
    .filter(([group]) => !optionalGroups.has(group))
    .flatMap(([group, values]) => values.length ? values.map((value) => `${group}:${value}`) : []);
  const optionalMissing = Object.entries(missing)
    .filter(([group]) => optionalGroups.has(group))
    .flatMap(([group, values]) => values.length ? values.map((value) => `${group}:${value}`) : []);

  if (optionalMissing.length) {
    console.warn(
      `[stage-runtime] info: optional inference packages not in build venv (${optionalMissing.join(", ")}). ` +
      `DFlash/TurboQuant will require user install via the Setup page.`,
    );
  }
  if (requiredMissing.length === 0) {
    return;
  }

  const message =
    `Embedded Python is missing required runtime packages (${requiredMissing.join(", ")}). ` +
    `Install them into the build venv with: ${pythonBinary} -m pip install -e ".[desktop,images]"`;
  if (strict) {
    throw new Error(message);
  }
  console.warn(`[stage-runtime] warning: ${message}`);
}

function validateBundledProjectImports(pythonBinary) {
  const script = [
    "import json",
    "from compression import registry",
    "print(json.dumps([entry['id'] for entry in registry.available()]))",
  ].join("\n");

  const env = {
    ...process.env,
    PYTHONPATH: [backendDest, sitePackagesDest].join(path.delimiter),
  };

  const payload = execFileSync(pythonBinary, ["-c", script], {
    cwd: workspaceRoot,
    encoding: "utf8",
    env,
  }).trim();
  const ids = JSON.parse(payload);
  const expected = ["native", "rotorquant", "triattention", "turboquant", "chaosengine"];
  const missing = expected.filter((id) => !ids.includes(id));
  if (missing.length === 0) {
    return;
  }

  const message =
    `Bundled runtime is missing expected cache strategies (${missing.join(", ")}). ` +
    `This usually means an import failure or path-shadowing issue in the staged backend.`;
  if (strict) {
    throw new Error(message);
  }
  console.warn(`[stage-runtime] warning: ${message}`);
}

function stageVendoredChaosEngine(pythonBinary) {
  const vendor = resolveChaosEngineVendor();
  if (!vendor) {
    return null;
  }

  console.log(`[stage-runtime] bundling ChaosEngine (${vendor.source})`);
  execFileSync(
    pythonBinary,
    [
      "-m",
      "pip",
      "install",
      "--disable-pip-version-check",
      "--no-deps",
      "--no-compile",
      "--upgrade",
      "--target",
      sitePackagesDest,
      vendor.path,
    ],
    {
      cwd: workspaceRoot,
      stdio: "inherit",
    },
  );
  return vendor;
}

function resolveChaosEngineVendor() {
  const override = process.env.CHAOSENGINE_VENDOR_PATH;
  if (override) {
    return {
      path: resolveExistingPath(override, "ChaosEngine vendor path"),
      source: "env-override",
    };
  }

  const vendoredPath = path.join(workspaceRoot, "vendor", "ChaosEngine");
  if (!fs.existsSync(vendoredPath)) {
    return null;
  }
  return {
    path: fs.realpathSync(vendoredPath),
    source: "vendor/ChaosEngine",
  };
}

function stageOptionalRuntimePackages(pythonBinary) {
  // Pre-install optional runtime packages into the staged site-packages
  // so that DFlash, TurboQuant, and RotorQuant work out of the box for
  // new users without requiring manual pip installs via the Setup page.
  //
  // Each entry: [pip package name, import name used for verification]
  const optionalPackages = [
    ["dflash-mlx", "dflash_mlx"],
    ["turboquant", "turboquant"],
    ["turboquant-mlx-full", "turboquant_mlx"],
  ];

  const installed = [];
  const skipped = [];

  for (const [pipName, importName] of optionalPackages) {
    // Check if already available in the build venv
    const checkScript = `import importlib.util; exit(0 if importlib.util.find_spec("${importName}") else 1)`;
    let available = false;
    try {
      execFileSync(pythonBinary, ["-c", checkScript], {
        cwd: workspaceRoot,
        stdio: "ignore",
      });
      available = true;
    } catch {
      available = false;
    }

    if (!available) {
      const message = `Optional package "${pipName}" not found in build venv — skipping bundle`;
      if (strict) {
        console.warn(`[stage-runtime] warning: ${message}. Install with: ${pythonBinary} -m pip install ${pipName}`);
      }
      skipped.push(pipName);
      continue;
    }

    try {
      console.log(`[stage-runtime] bundling optional package: ${pipName}`);
      execFileSync(
        pythonBinary,
        [
          "-m", "pip", "install",
          "--disable-pip-version-check",
          "--no-deps",
          "--no-compile",
          "--upgrade",
          "--target", sitePackagesDest,
          pipName,
        ],
        {
          cwd: workspaceRoot,
          stdio: "inherit",
        },
      );
      installed.push(pipName);
    } catch (err) {
      const message = `Failed to bundle optional package "${pipName}": ${err.message}`;
      if (strict) {
        throw new Error(message);
      }
      console.warn(`[stage-runtime] warning: ${message}`);
      skipped.push(pipName);
    }
  }

  if (installed.length) {
    console.log(`[stage-runtime] bundled optional packages: ${installed.join(", ")}`);
  }
  if (skipped.length) {
    console.log(`[stage-runtime] skipped optional packages (not in build venv): ${skipped.join(", ")}`);
  }
  return installed;
}

function stageLlamaBinaries() {
  const warnings = [];
  const sourceDir = process.env.CHAOSENGINE_LLAMA_BIN_DIR || defaultLlamaBinDir();
  if (!fs.existsSync(sourceDir)) {
    if (strict) {
      throw new Error("llama.cpp binary directory not found.");
    }
    return ["llama.cpp binary directory not found."];
  }

  ensureDir(binDest);
  const entries = fs.readdirSync(sourceDir);
  const selected = entries.filter((entry) => shouldCopyLlamaEntry(entry));

  if (!selected.includes(binaryName("llama-server"))) {
    const message = `Missing ${binaryName("llama-server")} in the configured llama.cpp binary directory`;
    if (strict) {
      throw new Error(message);
    }
    warnings.push(message);
  }

  for (const entry of selected) {
    const sourcePath = path.join(sourceDir, entry);
    const destinationPath = path.join(binDest, entry);
    copyPath(sourcePath, destinationPath);
  }

  // Also pick up llama-server-turbo from ~/.chaosengine/bin/ if it was
  // not already included from the primary llama.cpp build directory.
  const turboName = binaryName("llama-server-turbo");
  if (!fs.existsSync(path.join(binDest, turboName))) {
    const chaosEngineBinDir = path.join(os.homedir(), ".chaosengine", "bin");
    const turboCandidatePath = path.join(chaosEngineBinDir, turboName);
    if (fs.existsSync(turboCandidatePath)) {
      copyPath(turboCandidatePath, path.join(binDest, turboName));
    }
  }

  return warnings;
}

function defaultLlamaBinDir() {
  return path.resolve(workspaceRoot, "..", "llama.cpp", "build", "bin");
}

function shouldCopyLlamaEntry(entry) {
  if (entry.startsWith("llama-server") || entry.startsWith("llama-cli")) {
    return true;
  }
  if (process.platform === "darwin" && entry.endsWith(".dylib")) {
    return true;
  }
  if (process.platform === "linux" && entry.includes(".so")) {
    return true;
  }
  if (process.platform === "win32" && entry.endsWith(".dll")) {
    return true;
  }
  return false;
}

function binaryName(base) {
  return process.platform === "win32" ? `${base}.exe` : base;
}

function resolveEmbeddedPythonBinary(versionTag) {
  if (process.platform === "win32") {
    return "python.exe";
  }
  const candidates = [
    path.join("bin", "python3"),
    path.join("bin", `python${versionTag}`),
    path.join("bin", "python"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(pythonHomeDest, candidate))) {
      return candidate;
    }
  }
  throw new Error(`No embedded Python binary was staged for version ${versionTag}.`);
}

function resolveExistingPath(targetPath, label) {
  if (!targetPath || !fs.existsSync(targetPath)) {
    throw new Error(`Missing ${label}: ${targetPath || "(empty)"}`);
  }
  return fs.realpathSync(targetPath);
}

function assertPathExists(targetPath, label) {
  if (!targetPath || !fs.existsSync(targetPath)) {
    throw new Error(`Missing ${label}: ${targetPath || "(empty)"}`);
  }
}

function cleanupStaleTauriResources() {
  const targetRoot = path.join(tauriRoot, "target");
  const staleRoots = [
    path.join(targetRoot, "debug", "python"),
    path.join(targetRoot, "debug", "runtime"),
    path.join(targetRoot, "release", "python"),
    path.join(targetRoot, "release", "runtime"),
  ];

  for (const staleRoot of staleRoots) {
    fs.rmSync(staleRoot, { recursive: true, force: true });
  }
}

function cleanupStagedRuntimeArtifacts() {
  if (!fs.existsSync(embeddedResourcesRoot)) {
    return;
  }

  for (const entry of fs.readdirSync(embeddedResourcesRoot)) {
    if (!entry.startsWith("runtime-")) {
      continue;
    }
    fs.rmSync(path.join(embeddedResourcesRoot, entry), { recursive: true, force: true });
  }
}

function maybeSignEmbeddedRuntime() {
  // Sign whenever we're on macOS and have a signing identity available.
  // We deliberately do NOT gate this on `strict` (release) mode, because
  // the release workflow currently uses dev-mode staging to skip the
  // llama.cpp binary requirement — but still needs the embedded Python
  // runtime signed for notarization to pass.
  if (process.platform !== "darwin" || !signingIdentity) {
    return;
  }

  const entitlements = path.join(tauriRoot, "macos", "ChaosEngineAI.entitlements");
  const hasEntitlements = fs.existsSync(entitlements);

  const signTargets = collectSignTargets(stageRoot).sort((left, right) => {
    const depthDelta = pathDepth(right) - pathDepth(left);
    if (depthDelta !== 0) {
      return depthDelta;
    }
    return left.localeCompare(right);
  });

  console.log(`[stage-runtime] codesigning ${signTargets.length} embedded Mach-O targets with identity "${signingIdentity}"`);

  for (const target of signTargets) {
    const args = [
      "--force",
      "--sign",
      signingIdentity,
      "--timestamp",
      "--options",
      "runtime",
    ];
    if (hasEntitlements) {
      args.push("--entitlements", entitlements);
    }
    args.push(target);
    execFileSync("codesign", args, {
      cwd: workspaceRoot,
      stdio: "inherit",
    });
  }
}

function collectSignTargets(rootPath) {
  const targets = [];
  if (!fs.existsSync(rootPath)) {
    return targets;
  }

  walk(rootPath, (entryPath, stat) => {
    if (stat.isSymbolicLink()) {
      return false;
    }
    if (stat.isDirectory()) {
      if (looksLikeBundle(entryPath)) {
        targets.push(entryPath);
        return false;
      }
      return true;
    }
    if (isRuntimeBinary(entryPath, stat)) {
      targets.push(entryPath);
    }
    return false;
  });
  return targets;
}

function walk(currentPath, visitor) {
  const stat = fs.lstatSync(currentPath);
  const descend = visitor(currentPath, stat);
  if (!descend || !stat.isDirectory()) {
    return;
  }

  for (const entry of fs.readdirSync(currentPath)) {
    walk(path.join(currentPath, entry), visitor);
  }
}

function looksLikeBundle(targetPath) {
  return [".app", ".framework", ".appex", ".xpc"].includes(path.extname(targetPath));
}

function isRuntimeBinary(targetPath, stat) {
  const extension = path.extname(targetPath).toLowerCase();
  if ([".dylib", ".so", ".node"].includes(extension)) {
    return true;
  }

  if ((stat.mode & 0o111) !== 0) {
    return true;
  }

  return path.basename(targetPath) === "Python";
}

function pathDepth(targetPath) {
  return targetPath.split(path.sep).length;
}

function copyTree(source, destination) {
  ensureDir(path.dirname(destination));
  fs.cpSync(source, destination, {
    recursive: true,
    force: true,
    verbatimSymlinks: true,
    filter: (currentPath) => !shouldIgnorePath(currentPath),
  });
}

function copyFile(source, destination) {
  ensureDir(path.dirname(destination));
  fs.copyFileSync(source, destination);
}

function copyPath(source, destination) {
  const stat = fs.lstatSync(source);
  if (stat.isDirectory()) {
    copyTree(source, destination);
    return;
  }
  ensureDir(path.dirname(destination));
  if (stat.isSymbolicLink()) {
    const target = fs.readlinkSync(source);
    safeUnlink(destination);
    fs.symlinkSync(target, destination);
    return;
  }
  fs.copyFileSync(source, destination);
}

function safeUnlink(targetPath) {
  try {
    fs.unlinkSync(targetPath);
  } catch {
    // Ignore existing path cleanup failures.
  }
}

function shouldIgnorePath(currentPath) {
  const base = path.basename(currentPath);
  // Skip the CPython build-artifact directory (Makefile, config.c,
  // python.o) — only used when rebuilding Python from source, and it
  // contains unsigned object files that trip macOS notarization.
  if (base.startsWith("config-") && base.includes("-darwin")) {
    return true;
  }
  return (
    base === "__pycache__" ||
    base === ".DS_Store" ||
    base.startsWith("._") ||
    base === "_CodeSignature" ||
    base === "Headers" ||
    base === "share" ||
    base === "pkgconfig" ||
    base === "Python.app" ||
    base === "Documentation" ||
    base.endsWith(".pyc") ||
    base.endsWith(".o") ||
    base.endsWith(".a")
  );
}

function pruneBundledProjectArtifacts() {
  if (!fs.existsSync(sitePackagesDest)) {
    return;
  }

  for (const entry of fs.readdirSync(sitePackagesDest)) {
    const fullPath = path.join(sitePackagesDest, entry);

    if (
      entry.startsWith("__editable__") ||
      entry.endsWith(".egg-link") ||
      /^chaosengine_ai-.*\.(dist-info|egg-info)$/.test(entry)
    ) {
      fs.rmSync(fullPath, { recursive: true, force: true });
      continue;
    }

    if (!entry.endsWith(".pth")) {
      continue;
    }

    const original = fs.readFileSync(fullPath, "utf8");
    const filtered = original
      .split(/\r?\n/)
      .filter((line) => {
        const trimmed = line.trim();
        if (!trimmed) return false;
        if (trimmed.includes(workspaceRoot)) return false;
        if (trimmed.includes("__editable__")) return false;
        if (trimmed.includes("chaosengine_ai")) return false;
        return true;
      })
      .join("\n");

    if (filtered.trim()) {
      fs.writeFileSync(fullPath, `${filtered}\n`);
    } else {
      fs.rmSync(fullPath, { force: true });
    }
  }
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}
