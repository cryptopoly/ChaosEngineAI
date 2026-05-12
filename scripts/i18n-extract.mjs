#!/usr/bin/env node
/**
 * FU-042 — frontend i18n key extractor.
 *
 * Walks ``src/`` for ``t("...")`` / ``i18n.t("...")`` / ``<Trans i18nKey="...">``
 * call sites, deduplicates the discovered keys per namespace, and:
 *
 *   1. Reports any keys present in source code but missing from
 *      ``src/locales/en/<ns>.json`` (the source-of-truth catalog).
 *   2. Reports any keys present in ``en/*.json`` but unreferenced in
 *      the code (orphans — candidates for deletion).
 *   3. Writes a ``src/locales/.extract-report.json`` snapshot for the
 *      CI gate (``i18n-validate.mjs``) to compare against.
 *
 * Conservative scan — only matches literal string keys; dynamic
 * ``t(variable)`` calls show up as ``<dynamic>`` in the report and are
 * filtered out before parity checks (they're un-extractable).  Folks
 * who need dynamic keys add explicit "extract hints":
 *
 *     // i18n-extract: chat.tokensUsed
 *     t(messageId);
 *
 * The script reads those ``// i18n-extract:`` comments and treats the
 * declared key as if it were a literal call site.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.join(ROOT, "src");
const LOCALES = path.join(SRC, "locales");
const REPORT = path.join(LOCALES, ".extract-report.json");

const NAMESPACES = new Set([
  "common",
  "chat",
  "studio",
  "library",
  "setup",
  "diagnostics",
  "runtime",
  "errors",
  "dashboard",
  "meta",
]);

const FILE_EXTS = new Set([".ts", ".tsx"]);

// Match literal ``t("..." | '...' | `...`)`` calls. Also tolerant of
// ``useTranslation("ns").t("key")`` and ``i18n.t("key")``. The first
// capture is the literal key.  Excludes template literals containing
// ``${...}`` — those are dynamic and we surface them separately.
const T_CALL = /(?:^|[^A-Za-z0-9_$])t\(\s*(["'])([^"'`]+?)\1\s*(?:,|\))/g;
const TRANS_KEY = /<Trans\b[^>]*\bi18nKey\s*=\s*(["'])([^"'`]+?)\1/g;
const EXTRACT_HINT = /\/\/\s*i18n-extract:\s*([\w.-]+)/g;

async function walk(dir) {
  const out = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    if (entry.name === "node_modules") continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await walk(fullPath)));
      continue;
    }
    if (FILE_EXTS.has(path.extname(entry.name))) out.push(fullPath);
  }
  return out;
}

function classify(rawKey) {
  // Keys like ``chat.tokensUsed`` → namespace ``chat``, leaf ``tokensUsed``.
  // Bare keys default to ``common``.
  if (rawKey.includes(":")) {
    const [ns, ...rest] = rawKey.split(":");
    return { namespace: ns, path: rest.join(":") };
  }
  const first = rawKey.split(".")[0];
  if (NAMESPACES.has(first)) {
    return { namespace: first, path: rawKey.slice(first.length + 1) };
  }
  return { namespace: "common", path: rawKey };
}

function leafPaths(node, prefix = []) {
  if (typeof node === "string") return [prefix.join(".")];
  if (!node || typeof node !== "object") return [];
  if (Array.isArray(node)) return prefix.length ? [prefix.join(".")] : [];
  const acc = [];
  for (const [key, value] of Object.entries(node)) {
    acc.push(...leafPaths(value, [...prefix, key]));
  }
  return acc;
}

async function loadEnCatalog() {
  const catalog = {};
  for (const ns of NAMESPACES) {
    const file = path.join(LOCALES, "en", `${ns}.json`);
    try {
      const raw = await fs.readFile(file, "utf-8");
      catalog[ns] = new Set(leafPaths(JSON.parse(raw)));
    } catch (err) {
      console.warn(`[extract] failed to read en/${ns}.json: ${err.message}`);
      catalog[ns] = new Set();
    }
  }
  return catalog;
}

async function main() {
  const files = await walk(SRC);
  const used = Object.fromEntries(Array.from(NAMESPACES, (ns) => [ns, new Set()]));
  const dynamic = [];

  for (const file of files) {
    const rel = path.relative(ROOT, file);
    const content = await fs.readFile(file, "utf-8");
    for (const match of content.matchAll(T_CALL)) {
      const raw = match[2];
      const { namespace, path: leaf } = classify(raw);
      if (used[namespace]) used[namespace].add(leaf);
    }
    for (const match of content.matchAll(TRANS_KEY)) {
      const raw = match[2];
      const { namespace, path: leaf } = classify(raw);
      if (used[namespace]) used[namespace].add(leaf);
    }
    for (const match of content.matchAll(EXTRACT_HINT)) {
      const raw = match[1];
      const { namespace, path: leaf } = classify(raw);
      if (used[namespace]) used[namespace].add(leaf);
    }
    // Surface dynamic call sites separately so reviewers can spot them.
    const dyn = content.match(/\bt\(\s*\$\{[^}]+\}/g);
    if (dyn) dynamic.push(rel);
  }

  const en = await loadEnCatalog();

  const missing = {};
  const orphans = {};
  for (const ns of NAMESPACES) {
    const usedSet = used[ns];
    const enSet = en[ns];
    const miss = [...usedSet].filter((leaf) => leaf && !enSet.has(leaf));
    const orph = [...enSet].filter((leaf) => leaf && !usedSet.has(leaf));
    if (miss.length) missing[ns] = miss.sort();
    if (orph.length) orphans[ns] = orph.sort();
  }

  const report = {
    generatedAt: new Date().toISOString(),
    namespaces: Object.fromEntries(
      Array.from(NAMESPACES, (ns) => [
        ns,
        { used: [...used[ns]].sort(), enKeys: [...en[ns]].sort() },
      ]),
    ),
    missing,
    orphans,
    dynamicCallSites: dynamic.sort(),
  };

  await fs.writeFile(REPORT, JSON.stringify(report, null, 2) + "\n", "utf-8");

  const missingCount = Object.values(missing).reduce((sum, arr) => sum + arr.length, 0);
  const orphanCount = Object.values(orphans).reduce((sum, arr) => sum + arr.length, 0);
  console.log(`[i18n-extract] scanned ${files.length} files`);
  console.log(`[i18n-extract] used keys (per ns):`);
  for (const ns of NAMESPACES) {
    console.log(`  ${ns}: ${used[ns].size} used, ${en[ns].size} catalog`);
  }
  if (missingCount) {
    console.log(`[i18n-extract] ${missingCount} key(s) used but NOT in en catalog:`);
    for (const [ns, arr] of Object.entries(missing)) {
      console.log(`  ${ns}:`);
      for (const leaf of arr) console.log(`    + ${leaf}`);
    }
  }
  if (orphanCount) {
    console.log(`[i18n-extract] ${orphanCount} key(s) in en catalog but unreferenced:`);
    for (const [ns, arr] of Object.entries(orphans)) {
      console.log(`  ${ns}:`);
      for (const leaf of arr) console.log(`    - ${leaf}`);
    }
  }
  if (dynamic.length) {
    console.log(`[i18n-extract] ${dynamic.length} file(s) with dynamic t() — add // i18n-extract: hints:`);
    for (const rel of dynamic) console.log(`  ${rel}`);
  }
  console.log(`[i18n-extract] report written to ${path.relative(ROOT, REPORT)}`);
}

main().catch((err) => {
  console.error("[i18n-extract] failed:", err);
  process.exit(1);
});
