#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadEnvFiles } from "./load-env.mjs";

const scriptRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptRoot, "..");
const tauriRoot = path.join(projectRoot, "src-tauri");
const tauriConfigPath = path.join(tauriRoot, "tauri.conf.json");
const tauriConfig = JSON.parse(fs.readFileSync(tauriConfigPath, "utf8"));
loadEnvFiles([
  path.join(projectRoot, ".env"),
  path.join(projectRoot, ".env.local"),
]);
const productName = tauriConfig.productName || "ChaosEngineAI";
const version = tauriConfig.version || "0.1.0";
const arch = normalizeArch(process.arch);
const releaseRoot = path.join(projectRoot, "releases", "macos");
const builtAppPath = path.join(tauriRoot, "target", "release", "bundle", "macos", `${productName}.app`);
const releaseAppPath = path.join(releaseRoot, `${productName}.app`);
const releaseDmgPath = path.join(releaseRoot, `${productName}_${version}_${arch}.dmg`);
const entitlementsPath = path.join(tauriRoot, "macos", "ChaosEngineAI.entitlements");

const args = new Set(process.argv.slice(2));
const signingIdentity = args.has("--skip-sign")
  ? ""
  : process.env.CHAOSENGINE_APPLE_SIGNING_IDENTITY || process.env.APPLE_SIGNING_IDENTITY || "";
const skipNotarize = args.has("--skip-notarize") || process.env.CHAOSENGINE_SKIP_NOTARIZE === "1";
const notaryAuth = resolveNotaryAuthArgs();
const tauriUpdaterPrivateKey = process.env.TAURI_SIGNING_PRIVATE_KEY || "";

main();

function main() {
  if (process.platform !== "darwin") {
    throw new Error("`release-macos` only runs on macOS.");
  }

  ensureDir(releaseRoot);
  fs.rmSync(releaseAppPath, { recursive: true, force: true });
  fs.rmSync(releaseDmgPath, { force: true });

  const { buildArgs, cleanupConfigPath } = prepareTauriBuildArgs();
  try {
    run("npx", buildArgs, {
      cwd: projectRoot,
      env: process.env,
    });
  } finally {
    if (cleanupConfigPath) {
      fs.rmSync(cleanupConfigPath, { force: true });
    }
  }

  assertPathExists(builtAppPath, "Tauri app bundle");
  fs.cpSync(builtAppPath, releaseAppPath, { recursive: true, force: true });

  if (signingIdentity) {
    signAppBundle(releaseAppPath);
    verifySignedApp(releaseAppPath);
  } else {
    console.warn("[release-macos] APPLE_SIGNING_IDENTITY not set; building an unsigned app bundle.");
  }

  createDistributionDmg(releaseAppPath, releaseDmgPath);

  if (signingIdentity) {
    signFlatArtifact(releaseDmgPath);
  }

  if (signingIdentity && !skipNotarize && notaryAuth) {
    notarizeArtifact(releaseDmgPath, notaryAuth);
  } else if (!signingIdentity) {
    console.warn("[release-macos] Notarization skipped because the app was not code-signed.");
  } else if (skipNotarize) {
    console.warn("[release-macos] Notarization skipped by request.");
  } else {
    console.warn("[release-macos] Notarization skipped because no Apple notary credentials were configured.");
  }

  publishToAssets(releaseAppPath, releaseDmgPath);

  console.log(`[release-macos] app -> ${releaseAppPath}`);
  console.log(`[release-macos] dmg -> ${releaseDmgPath}`);
}

function publishToAssets(appPath, dmgPath) {
  // Mirror the signed app + dmg into the shared ``assets/`` folder at the
  // repo root so macOS, Windows, and Linux builds all deposit installers
  // in one predictable location. We copy rather than move so the existing
  // releases/macos/ layout stays intact for scripts that still reference it.
  const assetsRoot = path.join(projectRoot, "assets");
  fs.mkdirSync(assetsRoot, { recursive: true });

  if (fs.existsSync(appPath)) {
    const destApp = path.join(assetsRoot, path.basename(appPath));
    fs.rmSync(destApp, { recursive: true, force: true });
    fs.cpSync(appPath, destApp, { recursive: true, force: true, verbatimSymlinks: true });
  }
  if (fs.existsSync(dmgPath)) {
    const destDmg = path.join(assetsRoot, path.basename(dmgPath));
    fs.rmSync(destDmg, { force: true });
    fs.copyFileSync(dmgPath, destDmg);
  }
  console.log(`[release-macos] published artifacts -> ${assetsRoot}`);
}

function prepareTauriBuildArgs() {
  const buildArgs = ["tauri", "build", "--bundles", "app", "--ci"];
  let cleanupConfigPath = null;

  if (!signingIdentity) {
    buildArgs.push("--no-sign");
  }

  if (!tauriUpdaterPrivateKey) {
    const overrideConfigPath = path.join(os.tmpdir(), `chaosengine-tauri-local-${Date.now()}.json`);
    fs.writeFileSync(
      overrideConfigPath,
      `${JSON.stringify({ bundle: { createUpdaterArtifacts: false } })}\n`,
      "utf8",
    );
    buildArgs.push("--config", overrideConfigPath);
    cleanupConfigPath = overrideConfigPath;
    console.warn(
      "[release-macos] TAURI_SIGNING_PRIVATE_KEY not set; disabling updater artifacts for this local build.",
    );
  }

  return { buildArgs, cleanupConfigPath };
}

function normalizeArch(value) {
  if (value === "arm64") {
    return "aarch64";
  }
  if (value === "x64") {
    return "x86_64";
  }
  return value;
}

function resolveNotaryAuthArgs() {
  const keychainProfile =
    process.env.CHAOSENGINE_NOTARY_KEYCHAIN_PROFILE ||
    process.env.APPLE_KEYCHAIN_PROFILE ||
    process.env.APPLE_NOTARYTOOL_PROFILE ||
    "";
  if (keychainProfile) {
    return ["--keychain-profile", keychainProfile];
  }

  const apiKeyId =
    process.env.CHAOSENGINE_APPLE_API_KEY_ID ||
    process.env.CHAOSENGINE_APPLE_API_KEY ||
    process.env.APPLE_API_KEY ||
    "";
  const apiKeyPath = process.env.CHAOSENGINE_APPLE_API_KEY_PATH || process.env.APPLE_API_KEY_PATH || "";
  const apiIssuer = process.env.CHAOSENGINE_APPLE_API_ISSUER || process.env.APPLE_API_ISSUER || "";
  if (apiKeyId && apiKeyPath) {
    const args = ["--key", apiKeyPath, "--key-id", apiKeyId];
    if (apiIssuer) {
      args.push("--issuer", apiIssuer);
    }
    return args;
  }

  const appleId = process.env.CHAOSENGINE_APPLE_ID || process.env.APPLE_ID || "";
  const applePassword = process.env.CHAOSENGINE_APPLE_PASSWORD || process.env.APPLE_PASSWORD || "";
  const appleTeamId = process.env.CHAOSENGINE_APPLE_TEAM_ID || process.env.APPLE_TEAM_ID || "";
  if (appleId && applePassword && appleTeamId) {
    return ["--apple-id", appleId, "--password", applePassword, "--team-id", appleTeamId];
  }

  return null;
}

function signAppBundle(appPath) {
  const command = [
    "--force",
    "--deep",
    "--sign",
    signingIdentity,
    "--timestamp",
    "--options",
    "runtime",
  ];
  if (fs.existsSync(entitlementsPath)) {
    command.push("--entitlements", entitlementsPath);
  }
  command.push(appPath);
  run("codesign", command);
}

function signFlatArtifact(targetPath) {
  run("codesign", ["--force", "--sign", signingIdentity, "--timestamp", targetPath]);
}

function verifySignedApp(appPath) {
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", appPath]);
  run("spctl", ["--assess", "--type", "execute", "-vv", appPath]);
}

function createDistributionDmg(appPath, dmgPath) {
  const dmgStageRoot = fs.mkdtempSync(path.join(os.tmpdir(), "chaosengine-dmg-"));
  const stageAppPath = path.join(dmgStageRoot, path.basename(appPath));
  try {
    fs.cpSync(appPath, stageAppPath, { recursive: true, force: true });
    fs.symlinkSync("/Applications", path.join(dmgStageRoot, "Applications"));
    run("hdiutil", [
      "create",
      "-fs",
      "HFS+",
      "-volname",
      productName,
      "-srcfolder",
      dmgStageRoot,
      "-ov",
      "-format",
      "UDZO",
      dmgPath,
    ]);
  } finally {
    fs.rmSync(dmgStageRoot, { recursive: true, force: true });
  }
}

function notarizeArtifact(targetPath, authArgs) {
  const submissionPayload = execFileSync(
    "xcrun",
    ["notarytool", "submit", targetPath, "--wait", "--output-format", "json", ...authArgs],
    {
      cwd: projectRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
    },
  ).trim();

  const submission = submissionPayload ? JSON.parse(submissionPayload) : {};
  if (submission.status && submission.status !== "Accepted") {
    throw new Error(`Notarization failed with status ${submission.status}.`);
  }

  run("xcrun", ["stapler", "staple", "-v", targetPath]);
  run("xcrun", ["stapler", "validate", "-v", targetPath]);
}

function run(command, commandArgs, options = {}) {
  execFileSync(command, commandArgs, {
    cwd: projectRoot,
    stdio: "inherit",
    ...options,
  });
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}

function assertPathExists(targetPath, label) {
  if (!fs.existsSync(targetPath)) {
    throw new Error(`Missing ${label}: ${targetPath}`);
  }
}
