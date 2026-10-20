#!/usr/bin/env node
/**
 * Fail unless every place the app version lives agrees, and (with --tag) the
 * release tag agrees too.
 *
 *   node scripts/check-version-sync.mjs            # manifests + lock files
 *   node scripts/check-version-sync.mjs --tag v0.9.8
 *
 * The in-app updater compares the running app's version (tauri.conf.json) with
 * the version in latest.json, which the release workflow takes from the tag. A
 * tag that does not match tauri.conf.json makes every install look out of date
 * forever: v0.9.4 shipped files named 0.9.3, so an up-to-date install kept
 * offering the same release. ``--root <dir>`` points the check at another
 * checkout (the tests use it).
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const root = flag("--root") ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tag = flag("--tag");

const read = (rel) => readFileSync(path.join(root, rel), "utf8");
const sources = [
  ["package.json", () => JSON.parse(read("package.json")).version],
  ["package-lock.json", () => JSON.parse(read("package-lock.json")).packages[""].version],
  ["pyproject.toml", () => read("pyproject.toml").match(/^version\s*=\s*"([^"]+)"/m)?.[1]],
  ["src-tauri/Cargo.toml", () => read("src-tauri/Cargo.toml").match(/^version\s*=\s*"([^"]+)"/m)?.[1]],
  [
    "src-tauri/Cargo.lock",
    () => {
      const name = read("src-tauri/Cargo.toml").match(/^name\s*=\s*"([^"]+)"/m)?.[1];
      const block = read("src-tauri/Cargo.lock").match(new RegExp(`name = "${name}"\\nversion = "([^"]+)"`));
      return block?.[1];
    },
  ],
  ["src-tauri/tauri.conf.json", () => JSON.parse(read("src-tauri/tauri.conf.json")).version],
];

const found = sources.map(([label, get]) => {
  try {
    return { label, version: get() ?? null };
  } catch {
    return { label, version: null };
  }
});
if (tag !== undefined) found.push({ label: `release tag ${tag}`, version: tag.replace(/^v/, "") });

const detail = found.map((f) => `${f.label}=${f.version}`).join("  ");
if (found.some((f) => !f.version) || new Set(found.map((f) => f.version)).size > 1) {
  console.error(`App version mismatch: ${detail}`);
  process.exit(1);
}
console.log(`App version ${found[0].version} agrees everywhere (${found.length} sources)`);
