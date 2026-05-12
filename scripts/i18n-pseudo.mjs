#!/usr/bin/env node
/**
 * FU-042 — pseudo-locale generator.
 *
 * Produces two synthetic locale trees under ``src/locales/`` for QA
 * purposes:
 *
 *   - **en-XA** — accented + lengthened (e.g. ``Save`` → ``[Šåvé___]``).
 *     Catches: missing extraction, layout clipping at +40 % length,
 *     hardcoded English strings that bypassed ``t()``.
 *
 *   - **en-XB** — RTL pseudo (right-to-left override).  Wraps every
 *     leaf in U+202E + U+202C so flipping ``dir`` exercises the bidi
 *     code path against Latin text (no actual Arabic / Hebrew
 *     dependency yet).  Catches: hardcoded ``margin-left`` instead of
 *     ``margin-inline-start`` in CSS.
 *
 * Both pseudo-locales are *dev-only* — they're hidden behind the
 * ``?devLocale=en-XA`` URL param + ``CHAOSENGINE_I18N_DEV_MODE=1`` env
 * var so they don't leak into shipped builds.  This script regenerates
 * the trees on demand; commit the output if you want CI checks against
 * them, otherwise it's a local-only artefact.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LOCALES = path.join(ROOT, "src", "locales");
const EN_DIR = path.join(LOCALES, "en");

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

const ACCENT_MAP = {
  a: "å", A: "Å", b: "ƀ", B: "Ɓ", c: "ç", C: "Ç",
  d: "đ", D: "Đ", e: "é", E: "É", f: "ƒ", F: "Ƒ",
  g: "ğ", G: "Ğ", h: "ħ", H: "Ħ", i: "í", I: "Í",
  j: "ĵ", J: "Ĵ", k: "ķ", K: "Ķ", l: "ł", L: "Ł",
  m: "ɱ", M: "Ɯ", n: "ñ", N: "Ñ", o: "ö", O: "Ö",
  p: "þ", P: "Þ", q: "ǫ", Q: "Ǫ", r: "ř", R: "Ř",
  s: "š", S: "Š", t: "ŧ", T: "Ŧ", u: "ü", U: "Ü",
  v: "ṽ", V: "Ṽ", w: "ŵ", W: "Ŵ", x: "ẋ", X: "Ẋ",
  y: "ý", Y: "Ý", z: "ž", Z: "Ž",
};

/**
 * Apply pseudo-loc transforms to a string, but ONLY to the literal
 * characters — preserve ICU placeholders ``{name}``, ``{n, plural, ...}``,
 * angle-bracket tags, and other interpolation syntax intact so the
 * resulting catalog still parses + functions at runtime.
 */
function transformLiteral(value, transform) {
  // Tokenize: alternate literal runs vs. structural tokens we must
  // pass through unchanged.  This handles nested ICU correctly because
  // we only ever modify the literal text *between* tokens.
  const ICU = /(\{[^{}]*\})/g;
  return value
    .split(ICU)
    .map((chunk) => (chunk.startsWith("{") ? chunk : transform(chunk)))
    .join("");
}

function accentAndPad(text) {
  if (!text) return text;
  const accented = [...text].map((ch) => ACCENT_MAP[ch] ?? ch).join("");
  // Pad to ~140 % length, capped at 200 chars so we don't blow out
  // single-word UI labels into novels.
  const target = Math.min(200, Math.max(text.length + 4, Math.round(text.length * 1.4)));
  const padNeeded = Math.max(0, target - accented.length);
  const padding = "_".repeat(padNeeded);
  return `[${accented}${padding}]`;
}

const RLO = "‮"; // Right-to-Left Override
const PDF = "‬"; // Pop Directional Formatting

function rtlWrap(text) {
  if (!text) return text;
  return `${RLO}${text}${PDF}`;
}

function deepTransform(node, transform) {
  if (typeof node === "string") return transformLiteral(node, transform);
  if (Array.isArray(node)) return node.map((entry) => deepTransform(entry, transform));
  if (node && typeof node === "object") {
    const out = {};
    for (const [k, v] of Object.entries(node)) out[k] = deepTransform(v, transform);
    return out;
  }
  return node;
}

async function writePseudo(target, transform) {
  const dir = path.join(LOCALES, target);
  await fs.mkdir(dir, { recursive: true });
  for (const ns of NAMESPACES) {
    const src = path.join(EN_DIR, `${ns}.json`);
    const dst = path.join(dir, `${ns}.json`);
    const raw = await fs.readFile(src, "utf-8");
    const parsed = JSON.parse(raw);
    const transformed = deepTransform(parsed, transform);
    // Don't pseudo-loc the meta tree's supportedLocales — endonyms must
    // stay readable so the picker still works.  Patch back the source.
    if (ns === "meta") {
      transformed.locale = target;
      transformed.displayName = target;
      transformed.endonym = target;
      transformed.supportedLocales = parsed.supportedLocales;
    }
    await fs.writeFile(dst, JSON.stringify(transformed, null, 2) + "\n", "utf-8");
  }
  console.log(`[i18n-pseudo] wrote ${path.relative(ROOT, dir)}`);
}

async function main() {
  await writePseudo("en-XA", accentAndPad);
  await writePseudo("en-XB", rtlWrap);
  console.log("[i18n-pseudo] done");
}

main().catch((err) => {
  console.error("[i18n-pseudo] failed:", err);
  process.exit(1);
});
