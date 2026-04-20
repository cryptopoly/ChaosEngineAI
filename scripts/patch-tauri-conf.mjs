#!/usr/bin/env node

// Patch or restore src-tauri/tauri.conf.json for local release builds.
//
// Usage:
//   node scripts/patch-tauri-conf.mjs patch
//   node scripts/patch-tauri-conf.mjs restore
//
// Rationale: the previous implementation embedded this logic as an inline
// `node -e "..."` call from build.ps1. On Windows PowerShell, the mix of
// single quotes inside the double-quoted argument made the script unreliable
// — one misparse left the JSON file empty, which then cascaded into a
// confusing "EOF while parsing a value at line 1 column 0" from `tauri
// build`. A standalone .mjs file has no quoting surface area, and we can
// self-heal an empty/corrupt tauri.conf.json by restoring it from git
// before patching.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const confPath = path.join(projectRoot, "src-tauri", "tauri.conf.json");

const action = process.argv[2];
if (action !== "patch" && action !== "restore") {
  console.error(`Usage: node ${path.basename(fileURLToPath(import.meta.url))} <patch|restore>`);
  process.exit(2);
}

function readConf() {
  let raw;
  try {
    raw = fs.readFileSync(confPath, "utf8");
  } catch (err) {
    throw new Error(`Cannot read ${confPath}: ${err.message}`);
  }
  if (!raw.trim()) {
    // A prior failed build may have left the file empty. Restore it from
    // git before continuing so we have something valid to patch.
    console.log(`==> tauri.conf.json is empty — restoring from git`);
    try {
      execFileSync("git", ["checkout", "--", "src-tauri/tauri.conf.json"], {
        cwd: projectRoot,
        stdio: "inherit",
      });
    } catch (err) {
      throw new Error(`tauri.conf.json is empty and git checkout failed: ${err.message}`);
    }
    raw = fs.readFileSync(confPath, "utf8");
    if (!raw.trim()) {
      throw new Error(`tauri.conf.json is still empty after git checkout`);
    }
  }
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(`tauri.conf.json is not valid JSON: ${err.message}`);
  }
}

function writeConf(conf) {
  fs.writeFileSync(confPath, `${JSON.stringify(conf, null, 2)}\n`);
}

if (action === "patch") {
  const conf = readConf();
  // stage:runtime:release (not stage:runtime) — the dev variant writes
  // mode=development into the manifest AND skips the runtime tar.gz, both
  // of which produce a broken installer.
  conf.build = conf.build || {};
  conf.build.beforeBundleCommand = "npm run stage:runtime:release";
  conf.bundle = conf.bundle || {};
  conf.bundle.createUpdaterArtifacts = false;
  writeConf(conf);
  console.log(`==> Patched ${path.relative(projectRoot, confPath)} for local release build`);
} else {
  // restore: let git put back the committed version
  execFileSync("git", ["checkout", "--quiet", "--", "src-tauri/tauri.conf.json"], {
    cwd: projectRoot,
    stdio: "inherit",
  });
  console.log(`==> Restored ${path.relative(projectRoot, confPath)} from git`);
}
