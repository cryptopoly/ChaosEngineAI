#!/usr/bin/env node
/**
 * FU-042 — i18n validator.
 *
 * Three independent checks, each emitting a non-zero exit code on
 * failure so CI can gate releases on the i18n surface:
 *
 *   1. **Parity** — every key in ``src/locales/en/<ns>.json`` exists
 *      in every other locale's ``<ns>.json``.  Per FU-042 §"Coverage
 *      Gate" the v1 policy is warn-only at <95 %; orchestrate via the
 *      ``--strict`` flag for the future block-on-merge upgrade.
 *
 *   2. **ICU syntax** — each leaf value is parsed by
 *      ``intl-messageformat`` so we catch malformed plurals like
 *      ``{n, plural, one {x} other}`` (missing brace) at build time
 *      rather than at runtime when a user actually hits the surface.
 *
 *   3. **Orphans** — keys present in *any* locale but missing from
 *      ``en/`` are flagged as definite mistakes (en is the source of
 *      truth; any other-locale extra is a translator error or a
 *      stale leftover).
 *
 *   4. **Source keys** — every literal ``t("key", ...)`` call under
 *      ``src/`` must name a key that ``en/<ns>.json`` has.  The namespace
 *      comes from the code itself (see ``i18n-source-keys.mjs``), so a key
 *      that only exists as an inline ``defaultValue`` fails the gate
 *      instead of silently showing English in every other locale.
 *
 * Exit codes:
 *   0   all checks pass
 *   1   structural error (malformed JSON, ICU parse fail, missing file)
 *   2   coverage below threshold (only when --strict is passed)
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { collectSourceKeys, inferNamespaces, loadTypeScript } from "./i18n-source-keys.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// I18N_LOCALES_DIR lets the unit test point the validator at a fixture tree.
const LOCALES = process.env.I18N_LOCALES_DIR
  ? path.resolve(process.env.I18N_LOCALES_DIR)
  : path.join(ROOT, "src", "locales");

// The source check reads ``src/`` next to the real catalogs.  A test that
// points I18N_LOCALES_DIR at a fixture tree has no matching source, so it
// opts in with I18N_SRC_DIR.
const SRC_DIR = process.env.I18N_SRC_DIR
  ? path.resolve(process.env.I18N_SRC_DIR)
  : process.env.I18N_LOCALES_DIR
    ? null
    : path.join(ROOT, "src");

const ARGS = new Set(process.argv.slice(2));
const STRICT = ARGS.has("--strict");
const MIN_COVERAGE = STRICT ? 95 : 0;

const NAMESPACES = [
  "common",
  "chat",
  "studio",
  "library",
  "setup",
  "diagnostics",
  "runtime",
  "errors",
  "meta",
  "dashboard",
];

const LOCALES_LIST = [
  "en",
  "zh-CN",
  "zh-TW",
  "ja",
  "de",
  "ru",
  "ko",
  "fr",
  "es",
  "pt-BR",
  "tlh",
];

// D-69: community / fan translations.  Parity, ICU and orphan checks
// still run for them; only the --strict 95 % coverage threshold skips
// them, and the coverage table marks them.
export const EXPERIMENTAL_LOCALES = new Set(["tlh"]);

// ``intl-messageformat`` is a transitive dep of ``i18next-icu``; load
// via createRequire so this CLI script works whether or not the user
// has run ``npm install`` yet (we fall back to a lighter regex check
// when the module isn't on disk).
const require = createRequire(import.meta.url);
let IntlMessageFormat = null;
try {
  ({ IntlMessageFormat } = require("intl-messageformat"));
} catch {
  // Fallback below.
}

function leafEntries(node, prefix = []) {
  if (typeof node === "string") return [[prefix.join("."), node]];
  if (!node || typeof node !== "object") return [];
  const acc = [];
  for (const [k, v] of Object.entries(node)) acc.push(...leafEntries(v, [...prefix, k]));
  return acc;
}

async function loadCatalog(locale) {
  const out = {};
  for (const ns of NAMESPACES) {
    const file = path.join(LOCALES, locale, `${ns}.json`);
    try {
      const raw = await fs.readFile(file, "utf-8");
      out[ns] = JSON.parse(raw);
    } catch (err) {
      throw new Error(`failed to read ${path.relative(ROOT, file)}: ${err.message}`);
    }
  }
  return out;
}

function validateIcu(value, locale, fullKey) {
  if (!/[{}]/.test(value)) return null;
  if (IntlMessageFormat) {
    try {
      new IntlMessageFormat(value, locale);
      return null;
    } catch (err) {
      return `ICU parse failed at ${locale}:${fullKey} — ${err.message}`;
    }
  }
  // Lightweight fallback when intl-messageformat isn't installed.
  // Balance-check the braces and reject obvious typos like a single
  // ``{`` with no closer.
  let depth = 0;
  for (const ch of value) {
    if (ch === "{") depth++;
    else if (ch === "}") depth--;
    if (depth < 0) return `Unbalanced '}' at ${locale}:${fullKey}`;
  }
  if (depth !== 0) return `Unbalanced '{' at ${locale}:${fullKey}`;
  return null;
}

/**
 * Source keys with no English catalog entry.  Returns the error lines.
 *
 * ``dynamic`` calls (``t(`voice.x.${id}`)``) cannot be enumerated, so only
 * their static prefix is checked: some English key must start with it.
 */
async function checkSourceKeys(enEntries) {
  if (!SRC_DIR) return { errors: [], note: "source check skipped (fixture catalogs)" };
  if (!loadTypeScript()) {
    return { errors: [], note: "typescript not installed - source check skipped (run npm install)" };
  }
  const { calls, dynamic, hints } = await collectSourceKeys(SRC_DIR, path.dirname(SRC_DIR));
  inferNamespaces(calls, enEntries);
  const errors = [];
  const missing = new Map(); // "ns: key" -> first "file:line" (+ count)
  for (const c of calls) {
    const where = `${c.file}:${c.line}`;
    if (c.ns === null) {
      errors.push(`source ${where} - t("${c.key}") has no namespace; write t("<ns>:${c.key}")`);
      continue;
    }
    if (!enEntries[c.ns]) {
      errors.push(`source ${where} - t("${c.key}") names the unknown namespace "${c.ns}"`);
      continue;
    }
    if (!enEntries[c.ns].has(c.key)) {
      const id = `${c.ns}: ${c.key}`;
      const prev = missing.get(id);
      if (prev) prev.count++;
      else missing.set(id, { where, count: 1, defaultValue: c.defaultValue });
    }
  }
  for (const [id, m] of missing) {
    const text = m.defaultValue === null ? "" : `  = ${JSON.stringify(m.defaultValue.length > 70 ? m.defaultValue.slice(0, 67) + "..." : m.defaultValue)}`;
    errors.push(`missing from en catalog: ${id}${text}  (${m.where}${m.count > 1 ? `, +${m.count - 1} more` : ""})`);
  }
  for (const d of dynamic) {
    if (!d.ns || !enEntries[d.ns] || !d.prefix.endsWith(".")) continue;
    let found = false;
    for (const leaf of enEntries[d.ns].keys()) {
      if (leaf.startsWith(d.prefix)) {
        found = true;
        break;
      }
    }
    if (!found) errors.push(`source ${d.file}:${d.line} - no en key starts with ${d.ns}:${d.prefix} (dynamic key)`);
  }
  for (const h of hints) {
    const bare = h.key.includes(":") ? h.key.split(":").slice(1).join(":") : h.key;
    const ns = h.key.includes(":") ? h.key.split(":")[0] : null;
    const spaces = ns ? [ns] : NAMESPACES;
    if (!spaces.some((n) => enEntries[n]?.has(bare))) {
      errors.push(`source ${h.file}:${h.line} - i18n-extract hint ${h.key} is not in the en catalog`);
    }
  }
  return { errors, note: null };
}

async function main() {
  console.log(`[i18n-validate] mode = ${STRICT ? "strict (CI gate)" : "warn-only"}`);
  if (!IntlMessageFormat) {
    console.log(`[i18n-validate] intl-messageformat not installed — using regex fallback ICU check`);
  }

  const catalogs = {};
  for (const locale of LOCALES_LIST) {
    catalogs[locale] = await loadCatalog(locale);
  }

  const enEntries = {};
  for (const ns of NAMESPACES) {
    enEntries[ns] = new Map(leafEntries(catalogs.en[ns]));
  }

  const errors = [];
  const warnings = [];
  const coverage = {};

  // ICU + parity passes
  for (const locale of LOCALES_LIST) {
    let totalKeys = 0;
    let presentKeys = 0;
    for (const ns of NAMESPACES) {
      const localeMap = new Map(leafEntries(catalogs[locale][ns]));
      for (const [leaf, value] of localeMap) {
        const icuErr = validateIcu(value, locale, `${ns}.${leaf}`);
        if (icuErr) errors.push(icuErr);
      }
      // Orphans (key in locale but missing in en)
      if (locale !== "en") {
        for (const leaf of localeMap.keys()) {
          if (!enEntries[ns].has(leaf)) {
            warnings.push(`orphan ${locale}:${ns}.${leaf} — not present in en`);
          }
        }
      }
      // Coverage
      const enLeaves = enEntries[ns];
      totalKeys += enLeaves.size;
      for (const leaf of enLeaves.keys()) if (localeMap.has(leaf)) presentKeys++;
    }
    const pct = totalKeys === 0 ? 100 : Math.round((presentKeys / totalKeys) * 1000) / 10;
    coverage[locale] = { presentKeys, totalKeys, pct };
  }

  // Source keys
  const source = await checkSourceKeys(enEntries);
  if (source.note) console.log(`[i18n-validate] ${source.note}`);
  errors.push(...source.errors);

  // Print coverage table
  console.log("[i18n-validate] coverage:");
  for (const locale of LOCALES_LIST) {
    const { presentKeys, totalKeys, pct } = coverage[locale];
    const bar = "█".repeat(Math.round(pct / 5));
    const mark = EXPERIMENTAL_LOCALES.has(locale) ? "  (experimental, outside the strict gate)" : "";
    console.log(
      `  ${locale.padEnd(6)} ${pct.toString().padStart(5)}%  ${bar.padEnd(20)}  (${presentKeys}/${totalKeys})${mark}`,
    );
  }

  if (warnings.length) {
    console.log(`[i18n-validate] ${warnings.length} warning(s):`);
    for (const w of warnings) console.log(`  ! ${w}`);
  }

  if (errors.length) {
    console.error(`[i18n-validate] ${errors.length} error(s):`);
    for (const e of errors) console.error(`  ✗ ${e}`);
    process.exit(1);
  }

  // Coverage gate (strict only)
  const lowCoverage = Object.entries(coverage)
    .filter(([locale]) => locale !== "en" && !EXPERIMENTAL_LOCALES.has(locale))
    .filter(([, c]) => c.pct < MIN_COVERAGE);
  if (STRICT && lowCoverage.length) {
    console.error(`[i18n-validate] ${lowCoverage.length} locale(s) below ${MIN_COVERAGE}% threshold:`);
    for (const [locale, c] of lowCoverage) {
      console.error(`  ${locale}: ${c.pct}%`);
    }
    process.exit(2);
  }

  console.log("[i18n-validate] OK");
}

main().catch((err) => {
  console.error("[i18n-validate] failed:", err.message);
  process.exit(1);
});
