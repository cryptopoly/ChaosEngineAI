/**
 * D-69: the i18n validator's strict coverage gate skips experimental
 * (community) locales but still checks their structure.
 *
 * Runs ``scripts/i18n-validate.mjs`` against a throwaway fixture tree
 * through ``I18N_LOCALES_DIR`` so the real catalogs are not involved.
 */

// The app tsconfig has no Node type definitions (the UI never touches
// Node APIs); this test drives a Node CLI script, so it declares the few
// Node bits it uses itself and loads the modules at run time.
declare const process: { execPath: string; env: Record<string, string | undefined> };
declare function require(id: string): any;

import { describe, it, expect, beforeAll, afterAll } from "vitest";

interface RunResult { status: number | null; stdout: string; stderr: string }
const { spawnSync } = require("node:child_process") as {
  spawnSync: (cmd: string, args: string[], opts: { env: Record<string, string | undefined>; encoding: string }) => RunResult;
};
const { mkdtempSync, mkdirSync, rmSync, writeFileSync } = require("node:fs") as {
  mkdtempSync: (prefix: string) => string;
  mkdirSync: (dir: string, opts: { recursive: boolean }) => void;
  rmSync: (dir: string, opts: { recursive: boolean; force: boolean }) => void;
  writeFileSync: (file: string, data: string) => void;
};
const { tmpdir } = require("node:os") as { tmpdir: () => string };
const path = require("node:path") as { resolve: (...parts: string[]) => string; join: (...parts: string[]) => string };

const { fileURLToPath } = require("node:url") as { fileURLToPath: (url: URL) => string };
// fileURLToPath, not ``.pathname``: on Windows the latter is "/C:/..." and resolves to "C:\C:\...".
const SCRIPT = path.resolve(fileURLToPath(new URL(".", import.meta.url)), "../../../scripts/i18n-validate.mjs");
const NAMESPACES = ["common", "chat", "studio", "library", "setup", "diagnostics", "runtime", "errors", "meta", "dashboard"];
const LOCALES = ["en", "zh-CN", "zh-TW", "ja", "de", "ru", "ko", "fr", "es", "pt-BR", "tlh"];

let root: string;

function writeTree(halfCoverageFor: string[]) {
  for (const locale of LOCALES) {
    mkdirSync(path.join(root, locale), { recursive: true });
    for (const ns of NAMESPACES) {
      const full = { a: "one", b: "two {n}", c: "three", d: "four" };
      const half = { a: "one", b: "two {n}" };
      const body = halfCoverageFor.includes(locale) ? half : full;
      writeFileSync(path.join(root, locale, `${ns}.json`), JSON.stringify(body));
    }
  }
}

function run(args: string[]) {
  return spawnSync(process.execPath, [SCRIPT, ...args], {
    env: { ...process.env, I18N_LOCALES_DIR: root },
    encoding: "utf-8",
  });
}

beforeAll(() => {
  root = mkdtempSync(path.join(tmpdir(), "i18n-gate-"));
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("i18n-validate strict gate", () => {
  it("passes when only the experimental locale is below 95 %", () => {
    writeTree(["tlh"]);
    const res = run(["--strict"]);
    expect(res.stdout).toMatch(/tlh\s+50%/);
    expect(res.stdout).toContain("experimental, outside the strict gate");
    expect(res.status).toBe(0);
  });

  it("still fails when a shipping locale is below 95 %", () => {
    writeTree(["tlh", "de"]);
    const res = run(["--strict"]);
    expect(res.status).toBe(2);
    expect(res.stderr).toContain("de: 50%");
    expect(res.stderr).not.toContain("tlh: 50%");
  });

  it("still reports an ICU error in the experimental locale", () => {
    writeTree([]);
    writeFileSync(path.join(root, "tlh", "common.json"), JSON.stringify({ a: "one", b: "{n, plural, other", c: "three", d: "four" }));
    const res = run([]);
    expect(res.status).toBe(1);
    expect(res.stderr).toContain("tlh:common.b");
  });
});

describe("i18n-validate source key check", () => {
  let srcDir: string;

  /** A fixture catalog tree where studio/chat hold the given extra English keys. */
  function writeCatalogs(extra: { studio?: Record<string, unknown>; chat?: Record<string, unknown> }) {
    writeTree([]);
    for (const [ns, body] of Object.entries(extra)) {
      const file = path.join(root, "en", `${ns}.json`);
      writeFileSync(file, JSON.stringify({ a: "one", b: "two {n}", c: "three", d: "four", ...body }));
      for (const locale of LOCALES.filter((l) => l !== "en")) {
        writeFileSync(path.join(root, locale, `${ns}.json`), JSON.stringify({ a: "one", b: "two {n}", c: "three", d: "four", ...body }));
      }
    }
  }

  function writeSource(body: string) {
    mkdirSync(path.join(srcDir, "features"), { recursive: true });
    writeFileSync(path.join(srcDir, "features", "Panel.tsx"), body);
  }

  function runWithSource(args: string[] = []) {
    return spawnSync(process.execPath, [SCRIPT, ...args], {
      env: { ...process.env, I18N_LOCALES_DIR: root, I18N_SRC_DIR: srcDir },
      encoding: "utf-8",
    });
  }

  beforeAll(() => {
    srcDir = path.join(mkdtempSync(path.join(tmpdir(), "i18n-src-")), "src");
    mkdirSync(srcDir, { recursive: true });
  });

  it("fails when a literal key with a defaultValue is not in the English catalog", () => {
    writeCatalogs({ studio: { voice: { known: "Known" } } });
    writeSource(`
      import { useTranslation } from "react-i18next";
      export function Panel() {
        const { t } = useTranslation("studio");
        return <p>{t("voice.known", { defaultValue: "Known" })}{t("voice.fresh", { defaultValue: "Fresh text" })}</p>;
      }
    `);
    const res = runWithSource(["--strict"]);
    expect(res.status).toBe(1);
    expect(res.stderr).toContain("studio: voice.fresh");
    expect(res.stderr).toContain("Fresh text");
    expect(res.stderr).not.toContain("voice.known");
  });

  it("resolves the namespace per component, so a key in another namespace does not count", () => {
    writeCatalogs({ chat: { voice: { onlyInChat: "Chat" } } });
    writeSource(`
      import { useTranslation } from "react-i18next";
      export function Panel() {
        const { t } = useTranslation("studio");
        return <p>{t("voice.onlyInChat", { defaultValue: "Chat" })}{t("chat:voice.onlyInChat", { defaultValue: "Chat" })}</p>;
      }
    `);
    const res = runWithSource();
    expect(res.status).toBe(1);
    expect(res.stderr).toContain("studio: voice.onlyInChat");
    expect(res.stderr).not.toContain("chat: voice.onlyInChat");
  });

  it("follows aliases, local wrappers and key descriptors", () => {
    writeCatalogs({ studio: { voice: { task: { quick: "Quick" } }, board: { fix: "Fix" } } });
    writeSource(`
      import i18next from "i18next";
      import { useTranslation } from "react-i18next";
      const label = (key: string, defaultValue: string) => i18next.t(\`studio:voice.task.\${key}\`, { defaultValue });
      const FIX = { key: "board.fix", defaultValue: "Fix" };
      const GONE = { key: "board.gone", defaultValue: "Gone" };
      export function Panel() {
        const { t: tStudio } = useTranslation("studio");
        return <p>{label("quick", "Quick")}{label("slow", "Slow")}{tStudio(FIX.key, { defaultValue: FIX.defaultValue })}{GONE.key}</p>;
      }
    `);
    const res = runWithSource();
    expect(res.status).toBe(1);
    expect(res.stderr).toContain("studio: voice.task.slow");
    expect(res.stderr).toContain("studio: board.gone");
    expect(res.stderr).not.toContain("voice.task.quick");
    expect(res.stderr).not.toContain("board.fix");
  });

  it("passes when every literal key is in the catalog", () => {
    writeCatalogs({ studio: { voice: { known: "Known" } } });
    writeSource(`
      import { useTranslation } from "react-i18next";
      export function Panel() {
        const { t } = useTranslation("studio");
        return <p>{t("voice.known", { defaultValue: "Known" })}</p>;
      }
    `);
    const res = runWithSource(["--strict"]);
    expect(res.stderr).toBe("");
    expect(res.status).toBe(0);
  });

  afterAll(() => {
    rmSync(path.resolve(srcDir, ".."), { recursive: true, force: true });
  });
});

