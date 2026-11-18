/**
 * Review L-2: Settings > Diagnostics showed about 100 English strings in
 * every language because its labels never went through t().
 *
 * Three guards, all against the key list in ``diagnostics.json`` and not
 * against screenshots:
 *
 *   1. Marker catalogue. A made-up language whose every diagnostics
 *      string is wrapped in a marker is rendered; any text on the panel
 *      that is neither marked, nor fixture data, nor a name kept verbatim
 *      on purpose (package, file and variable names) is a hard-coded
 *      string and fails.
 *   2. Every key the panel uses has a real string in the nine shipping
 *      languages (a ``null`` placeholder would fall back to English).
 *   3. Rendered in each shipping language, every section title and row
 *      label is that language's catalogue value, and no English value
 *      that the language translates differently is left on screen.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, describe, expect, it, vi } from "vitest";

vi.mock("../../api", () => ({
  fetchDiagnosticsSnapshot: vi.fn(),
  deleteHfRepo: vi.fn(),
  deleteModelPath: vi.fn(),
  fetchStorageChildren: vi.fn(),
  fetchStorageTop: vi.fn(),
  installCudaTorch: vi.fn(),
  installPipPackage: vi.fn(),
  reextractRuntime: vi.fn(),
}));
// The Accelerators panel lives in the same file as the Diagnostics panel; its children are not under test.
vi.mock("../videotools/VideoToolsRow", () => ({ VideoToolsRow: () => null }));
vi.mock("./AcceleratorsBoostPack", () => ({ AcceleratorsBoostPack: () => null }));
vi.mock("./LlamaCudaPanel", () => ({ LlamaCudaPanel: () => null }));
vi.mock("./WslBridgePanel", () => ({ WslBridgePanel: () => null }));

import type { DiagnosticsSnapshot } from "../../api";
import { changeLocale, i18n, type SupportedLocale } from "../../i18n";
import enDiagnostics from "../../locales/en/diagnostics.json";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { DiagnosticsRecoveryActions } from "./DiagnosticsRecoveryActions";
import { DiagnosticsSnapshotView, diagnosticsSections } from "./DiagnosticsSnapshotView";

const SHIPPING: SupportedLocale[] = ["zh-CN", "zh-TW", "ja", "de", "ru", "ko", "fr", "es", "pt-BR"];

/** Names shown as they are in every language: files, packages and the product. */
const VERBATIM = new Set([
  "cwd", "llama-server", "llama-server-turbo", "llama-cli",
  "torch", "diffusers", "accelerate", "transformers", "imageio", "imageio-ffmpeg",
  "sentencepiece", "tiktoken", "protobuf", "ftfy",
  "—", "·", "/",
]);

/** Keys of the dynamic sections (torch probe, environment): they come from the backend, not from the catalogue. */
const DYNAMIC_KEYS = new Set(["cuda_available", "torch_version", "CHAOSENGINE_PORT", "HF_HOME"]);

// Every value is marked "fx-" so data is told apart from labels.
const SNAPSHOT: DiagnosticsSnapshot = {
  generatedAt: 1_800_000_000,
  app: { appVersion: "9.9.9", workspaceRoot: "fx-root", logCount: 1, activeRequests: 0, requestsServed: 3 },
  os: { system: "fx-Darwin", release: "fx-27.3.0", version: "fx-kernel", machine: "fx-arm64", platform: "fx-macOS" },
  hardware: {
    cpu: { logicalCount: 16, physicalCount: 12 },
    memory: { totalGb: 64, availableGb: 31.5 },
    swap: { usedGb: 1, totalGb: 8 },
    disks: [],
    gpu: { gpuName: "fx-gpu", vramTotalGb: 24, driverVersion: "fx-driver", systemCudaVersion: "fx-cuda" },
  },
  python: {
    executable: "fx-python", version: "fx-3.13.1", versionTuple: [3, 13, 1], implementation: "fx-CPython",
    prefix: "fx-prefix", basePrefix: "fx-base", platform: "fx-platform", sysPath: [], cwd: null,
  },
  runtime: {
    engineName: "mlx", engineLabel: "fx-engine", loadedModel: null, warmPoolCount: 2,
    llamaServerPath: "fx-llama-server", llamaServerTurboPath: null, llamaCliPath: null,
  },
  gpu: {
    torchFindSpec: true, diffusersFindSpec: false, accelerateFindSpec: true, transformersFindSpec: true,
    imageioFindSpec: true, ffmpegFindSpec: true, sentencepieceFindSpec: true, tiktokenFindSpec: true,
    protobufFindSpec: true, ftfyFindSpec: false,
    torchSubprocess: { cuda_available: "fx-false", torch_version: "fx-2.9" },
  },
  extras: { path: "fx-extras", exists: false, freeBytes: 5 * 1024 ** 3, sizeBytes: null, topLevelEntries: [] },
  environment: { CHAOSENGINE_PORT: "fx-8876", HF_HOME: null },
  logs: { path: "fx-log-path", tailLines: [] },
};

type Tree = { [key: string]: string | Tree };

function leaves(node: unknown, prefix: string[] = []): Array<[string, unknown]> {
  if (node && typeof node === "object" && !Array.isArray(node)) {
    return Object.entries(node).flatMap(([key, value]) => leaves(value, [...prefix, key]));
  }
  return [[prefix.join("."), node]];
}

function mapStrings(node: unknown, fn: (text: string) => string): unknown {
  if (typeof node === "string") return fn(node);
  if (node && typeof node === "object") {
    return Object.fromEntries(Object.entries(node).map(([key, value]) => [key, mapStrings(value, fn)]));
  }
  return node;
}

/** The keys the Diagnostics panel reads: its title, its subtitle and everything under "panel". */
const PANEL_KEYS = leaves(enDiagnostics)
  .map(([key]) => key)
  .filter((key) => key === "title" || key === "subtitleSnapshot" || key.startsWith("panel."));

function decode(text: string): string {
  return text
    .replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
}

/** Text a user reads: text nodes plus aria-label / title / placeholder values. */
function visibleText(html: string): string[] {
  const attrs = [...html.matchAll(/\s(?:aria-label|title|placeholder)="([^"]*)"/g)].map((m) => m[1]);
  const nodes = html.replace(/<!--.*?-->/g, "\u0000").replace(/<[^>]+>/g, "\u0000").split("\u0000");
  return [...attrs, ...nodes].map((text) => decode(text).trim()).filter(Boolean);
}

function isData(text: string): boolean {
  return text.startsWith("fx-")
    || VERBATIM.has(text)
    || DYNAMIC_KEYS.has(text)
    || /^(ChaosEngineAI v)?\d+(\.\d+)*$/.test(text) // counts, and the product name with its version
    || /^[\d.]+ (B|KB|MB|GB|TB)( \/ [\d.]+ GB)?$/.test(text); // sizes
}

const OPEN = "[[";
const CLOSE = "]]";

/** Text that is not a marked catalogue string, fixture data or a verbatim name: a hard-coded string. */
function hardCoded(html: string): string[] {
  return visibleText(html)
    .map((text) => text.replace(/\[\[.*?\]\]/gs, "").trim())
    .filter((rest) => rest && !isData(rest));
}

function renderAll(): string {
  const noop = () => undefined;
  const actions = (os: Record<string, unknown>) => (
    <DiagnosticsRecoveryActions os={os} backendOnline onRestartServer={noop} busyAction={null} onRefresh={async () => undefined} />
  );
  return [
    <DiagnosticsPanel key="offline" backendOnline={false} onRestartServer={noop} busyAction={null} />,
    <DiagnosticsPanel key="online" backendOnline onRestartServer={noop} busyAction={null} />,
    <DiagnosticsSnapshotView key="view" snapshot={SNAPSHOT} lastFetchedAt={null} />,
    <div key="mac">{actions({ system: "Darwin", machine: "arm64" })}</div>,
    <div key="cuda">{actions({ system: "Linux", machine: "x86_64" })}</div>,
  ].map((element) => renderToStaticMarkup(element)).join("\n");
}

afterAll(async () => {
  await changeLocale("en");
});

describe("Settings > Diagnostics is translated (review L-2)", () => {
  it("has about a hundred strings to look after", () => {
    expect(PANEL_KEYS.length).toBeGreaterThanOrEqual(80);
  });

  it("renders no hard-coded text: everything is a catalogue string, data or a name kept verbatim", async () => {
    i18n.addResourceBundle("xx", "diagnostics", mapStrings(enDiagnostics, (text) => `${OPEN}${text}${CLOSE}`), true, true);
    await i18n.changeLanguage("xx");
    const html = renderAll();
    await i18n.changeLanguage("en");
    expect(html).toContain(`${OPEN}Hardware${CLOSE}`);
    expect(hardCoded(html)).toEqual([]);
  });

  it("a hard-coded label would be caught by that check", () => {
    const html = `<td>System</td><td>fx-Darwin</td><button aria-label="Refresh">${OPEN}ok${CLOSE}</button><p>${OPEN}ok${CLOSE} lines</p>`;
    expect(hardCoded(html)).toEqual(["Refresh", "System", "lines"]);
  });

  it.each(SHIPPING)("%s: every key has a real string and the panel shows it", async (locale) => {
    await changeLocale(locale);
    const catalogue = i18n.getResourceBundle(locale, "diagnostics") as Tree;
    const local = new Map(leaves(catalogue));
    const english = new Map(leaves(enDiagnostics));

    // 2. No key left as a null placeholder or dropped.
    const untranslated = PANEL_KEYS.filter((key) => typeof local.get(key) !== "string" || !(local.get(key) as string).trim());
    expect(untranslated, `${locale}: keys with no translation`).toEqual([]);

    // 3a. Section titles and row labels are this language's catalogue values.
    const labels = new Set(
      [...local].filter(([key]) => /^panel\.(sections|rows)\./.test(key)).map(([, value]) => value as string),
    );
    const sections = diagnosticsSections(SNAPSHOT, i18n.getFixedT(locale, "diagnostics"));
    expect(sections.map((section) => section.id)).toEqual([
      "os", "hardware", "python", "runtime", "gpuPackages", "torchRuntime", "extras", "environment",
    ]);
    const wrongTitles = sections.map((section) => section.title).filter((title) => !labels.has(title));
    const wrongLabels = sections
      .filter((section) => section.id !== "torchRuntime" && section.id !== "environment")
      .flatMap((section) => section.rows.map(([label]) => label))
      .filter((label) => !labels.has(label) && !VERBATIM.has(label));
    expect({ wrongTitles, wrongLabels }, `${locale}: labels that are not catalogue values`).toEqual({ wrongTitles: [], wrongLabels: [] });

    // 3b. No English string is left where this language says it differently.
    const englishOnly = new Set(
      PANEL_KEYS.filter((key) => english.get(key) !== local.get(key)).map((key) => english.get(key) as string),
    );
    const sameInBoth = new Set(PANEL_KEYS.filter((key) => english.get(key) === local.get(key)).map((key) => english.get(key) as string));
    const leftInEnglish = visibleText(renderAll()).filter((text) => englishOnly.has(text) && !sameInBoth.has(text));
    expect(leftInEnglish, `${locale}: English left on the panel`).toEqual([]);
  });
});
