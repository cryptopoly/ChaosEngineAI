#!/usr/bin/env node

// Copy the freshly-built desktop installers to ``<repo>/assets`` so the
// user doesn't have to hunt through ``src-tauri/target/release/bundle/**``
// (or ``releases/macos``) every release. Flat output, one directory, every
// platform's deliverables in the same place.
//
// Usage:
//   node scripts/publish-artifacts.mjs                 # copy every bundle
//   node scripts/publish-artifacts.mjs --bundles nsis  # limit to NSIS
//
// Called from build.ps1 / build.sh / release-macos.mjs. Safe to run more
// than once — existing files in ``assets/`` are overwritten.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const bundleRoot = path.join(projectRoot, "src-tauri", "target", "release", "bundle");
const releaseMacRoot = path.join(projectRoot, "releases", "macos");
const assetsRoot = path.join(projectRoot, "assets");

// File extensions we treat as shippable installer artifacts. ``.app`` is a
// directory bundle on macOS; everything else is a single file.
const INSTALLER_EXTS = new Set([
  ".dmg",
  ".exe",
  ".msi",
  ".deb",
  ".rpm",
  ".AppImage",
  ".appimage",
  ".app",
]);

// Parse --bundles (comma-separated) so build.sh can pass through the same
// filter it handed to ``tauri build``. When omitted we sweep every subdir.
const bundlesArg = process.argv.find((value) => value.startsWith("--bundles="));
const bundleFilter = bundlesArg
  ? new Set(bundlesArg.split("=", 2)[1].split(",").map((value) => value.trim()).filter(Boolean))
  : null;

main();

function main() {
  fs.mkdirSync(assetsRoot, { recursive: true });

  const copied = [];
  collectFromBundleTree(bundleRoot, copied);
  collectFromDir(releaseMacRoot, copied);

  if (!copied.length) {
    console.warn(`[publish-artifacts] no installer artifacts found under ${bundleRoot}`);
    return;
  }
  console.log(`[publish-artifacts] copied ${copied.length} artifact(s) to ${assetsRoot}`);
  for (const name of copied) {
    console.log(`  - ${name}`);
  }
}

function collectFromBundleTree(root, copied) {
  if (!fs.existsSync(root)) return;
  for (const subdir of fs.readdirSync(root)) {
    if (bundleFilter && !bundleFilter.has(subdir)) {
      continue;
    }
    const subPath = path.join(root, subdir);
    // ``bundle/`` can contain stray files like ``.DS_Store`` — skip anything
    // that isn't itself a directory before recursing.
    let stat;
    try { stat = fs.statSync(subPath); } catch { continue; }
    if (!stat.isDirectory()) continue;
    collectFromDir(subPath, copied);
  }
}

function collectFromDir(dir, copied) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir)) {
    if (entry === ".DS_Store") continue;
    const entryPath = path.join(dir, entry);
    if (!isInstallerArtifact(entry, entryPath)) {
      continue;
    }
    const dest = path.join(assetsRoot, entry);
    copyArtifact(entryPath, dest);
    copied.push(entry);
  }
}

function isInstallerArtifact(entryName, entryPath) {
  // Tauri's bundle_dmg writes an intermediate ``rw.<pid>.<name>.dmg`` as
  // it fills the disk image before converting to the final compressed
  // read-only DMG. Those files are tempdata the cleanup pass deletes —
  // they should never be published.
  if (/^rw\.\d+\./.test(entryName)) return false;
  if (entryName.endsWith(".temp") || entryName.includes(".tmp.")) return false;

  const ext = path.extname(entryName);
  if (!ext) return false;
  if (!INSTALLER_EXTS.has(ext)) {
    // Case-insensitive fallback (``.AppImage`` vs ``.appimage``).
    if (!INSTALLER_EXTS.has(ext.toLowerCase())) {
      return false;
    }
  }
  // ``.app`` is only an installer when it's a directory bundle.
  if (ext === ".app") {
    let stat;
    try {
      stat = fs.statSync(entryPath);
    } catch {
      return false;
    }
    return stat.isDirectory();
  }
  return true;
}

function copyArtifact(source, destination) {
  const stat = fs.lstatSync(source);
  if (stat.isDirectory()) {
    fs.rmSync(destination, { recursive: true, force: true });
    fs.cpSync(source, destination, { recursive: true, force: true, verbatimSymlinks: true });
    return;
  }
  fs.rmSync(destination, { force: true });
  fs.copyFileSync(source, destination);
}
