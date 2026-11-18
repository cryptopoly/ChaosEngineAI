import { readFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext, Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, openApp, openMenuIfCompact } from "./helpers";

// Review stream I18NFIX.
// L-2: Settings > Diagnostics showed about 100 English strings in every language. In German and Japanese
//      every label on the panel must now be that language's catalogue string. The check is against the key
//      list in src/locales/<lang>/diagnostics.json ("panel"), not against a screenshot.
// L-7: the Language panel is titled "Language and time" and its clock-format row, like the UI-scale row
//      above it, has a visible label.
// L-4: the Chat welcome names no model family (the catalogue moves on; the copy went stale).
// L-5: Compare's slot names and picker button are translated.
// Every test puts the saved language back to "system".

test.setTimeout(120_000);

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };
const LOCALES_DIR = path.resolve(__dirname, "..", "..", "src", "locales");

/** Names shown as they are in every language: files and packages. */
const VERBATIM = new Set([
  "cwd", "llama-server", "llama-server-turbo", "llama-cli",
  "torch", "diffusers", "accelerate", "transformers", "imageio", "imageio-ffmpeg",
  "sentencepiece", "tiktoken", "protobuf", "ftfy",
]);
/** Sections whose row labels come from the backend (probe keys, environment variable names). */
const DYNAMIC_SECTIONS = new Set(["torchRuntime", "environment"]);

type Tree = { [key: string]: string | null | Tree };

function catalogue(locale: string, namespace: string): Tree {
  return JSON.parse(readFileSync(path.join(LOCALES_DIR, locale, `${namespace}.json`), "utf8")) as Tree;
}

function leaves(node: unknown, prefix: string[] = []): Array<[string, string | null]> {
  if (node && typeof node === "object") {
    return Object.entries(node).flatMap(([key, value]) => leaves(value, [...prefix, key]));
  }
  return [[prefix.join("."), node as string | null]];
}

async function saveLocale(request: APIRequestContext, locale: string): Promise<void> {
  const response = await request.patch(`${API_BASE}/api/settings`, { headers, data: { locale } });
  expect(response.ok(), `PATCH locale ${locale} -> ${response.status()}`).toBeTruthy();
}

async function openInLocale(page: Page, request: APIRequestContext, locale: string): Promise<void> {
  // Accept the terms in English first; the button has another name once the language is saved.
  await openApp(page);
  await saveLocale(request, locale);
  await page.reload();
  await expect(page.locator("aside.sidebar")).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("lang", locale, { timeout: 15_000 });
  await expect(page.locator(".loading-state")).toHaveCount(0, { timeout: 60_000 });
}

async function openSettingsSection(page: Page, index: number): Promise<void> {
  await openMenuIfCompact(page);
  // Settings is the last standalone entry in every language.
  await page.locator("aside.sidebar nav .nav-button").last().click();
  const tab = page.locator(".settings-subtab-bar [role=tab]").nth(index);
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

test.afterEach(async ({ request }) => {
  await saveLocale(request, "system");
});

for (const locale of ["de", "ja"]) {
  test(`${locale}: Settings > Diagnostics shows no English label`, async ({ page, request }) => {
    const local = new Map(leaves(catalogue(locale, "diagnostics")));
    const english = new Map(leaves(catalogue("en", "diagnostics")));
    const panelKeys = [...english.keys()].filter((key) => key === "title" || key === "subtitleSnapshot" || key.startsWith("panel."));
    expect(panelKeys.length).toBeGreaterThanOrEqual(80);
    // The key list: every key has a string in this language.
    expect(panelKeys.filter((key) => typeof local.get(key) !== "string")).toEqual([]);
    const say = (key: string) => local.get(key) as string;

    await openInLocale(page, request, locale);
    await openSettingsSection(page, 5);
    const panel = page.locator("section.panel", { has: page.locator(".diagnostics-body") });
    await expect(panel.locator(".diagnostics-body")).toBeVisible({ timeout: 60_000 });

    // Header: title, subtitle, both buttons.
    await expect(panel.locator(".panel-header h2")).toHaveText(say("title"));
    await expect(panel.locator(".panel-header p")).toHaveText(say("subtitleSnapshot"));
    await expect(panel.locator(".panel-actions button")).toHaveText([say("panel.refresh"), say("panel.copy")]);

    // Section titles and row labels, read with the section they sit in.
    const seen = await panel.locator("details.diagnostics-section").evaluateAll((sections) =>
      sections.map((section) => ({
        id: section.getAttribute("data-section") ?? "",
        title: section.querySelector("summary")?.textContent?.trim() ?? "",
        labels: [...section.querySelectorAll("td.diagnostics-key")].map((cell) => cell.textContent?.trim() ?? ""),
        values: [...section.querySelectorAll("td.diagnostics-value")].map((cell) => cell.textContent?.trim() ?? ""),
      })),
    );
    expect(seen.map((section) => section.id)).toEqual(expect.arrayContaining(["os", "hardware", "python", "runtime", "gpuPackages", "extras", "environment"]));
    const rowLabels = new Set(panelKeys.filter((key) => key.startsWith("panel.rows.")).map(say));
    for (const section of seen) {
      expect.soft(section.title, `title of ${section.id}`).toBe(say(`panel.sections.${section.id}`));
      if (DYNAMIC_SECTIONS.has(section.id)) continue;
      const stray = section.labels.filter((label) => !rowLabels.has(label) && !VERBATIM.has(label));
      expect.soft(stray, `${locale} ${section.id}: labels that are not catalogue strings`).toEqual([]);
    }
    // Values the app words itself.
    const packages = seen.find((section) => section.id === "gpuPackages")!;
    expect(packages.values.length).toBe(10);
    for (const value of packages.values) expect([say("panel.values.installed"), say("panel.values.missing")]).toContain(value);
    const extras = seen.find((section) => section.id === "extras")!;
    expect([say("panel.values.yes"), say("panel.values.no")]).toContain(extras.values[1]);

    // The log tail line and the recovery actions.
    await expect(panel.locator(".install-log-summary")).toContainText(say("panel.logTail").split("(")[0].split("（")[0].trim());
    const actionTexts = await panel.locator(".diagnostics-actions").evaluateAll((blocks) =>
      blocks.flatMap((block) => [...block.querySelectorAll("p, button")].map((el) => el.textContent?.trim() ?? "")),
    );
    const allowed = new Set(panelKeys.map(say));
    expect(actionTexts.length).toBeGreaterThanOrEqual(3);
    expect(actionTexts.filter((text) => !allowed.has(text)), `${locale}: recovery actions not from the catalogue`).toEqual([]);
    expect(actionTexts).toContain(say("panel.reextract.button"));
    expect(actionTexts).toContain(say("panel.restart"));

    // Last, the blunt check: no English string that this language words differently is on the panel
    // (data cells and the log are left out; a path or a variable may hold any word).
    const englishOnly = panelKeys.filter((key) => english.get(key) !== local.get(key)).map((key) => english.get(key) as string);
    const sameInBoth = new Set(panelKeys.filter((key) => english.get(key) === local.get(key)).map((key) => english.get(key) as string));
    const texts = await panel.evaluate((root) => {
      const found: string[] = [];
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const text = (walker.currentNode.nodeValue ?? "").trim();
        const parent = walker.currentNode.parentElement;
        if (!text || !parent || parent.closest("pre, code, td.diagnostics-value")) continue;
        found.push(text);
      }
      return found;
    });
    const leftInEnglish = texts.filter((text) => englishOnly.includes(text) && !sameInBoth.has(text));
    expect(leftInEnglish, `${locale}: English left on Diagnostics`).toEqual([]);
  });
}

test("de: the Language panel has its own title and labelled rows; Chat welcome and Compare are translated", async ({ page, request }) => {
  const common = catalogue("de", "common") as { languagePanel: Record<string, string>; settingsTab: { appearance: Record<string, string> } };
  const chat = catalogue("de", "chat") as { emptyBanner: Record<string, string>; compareView: Record<string, string> & { slot: Record<string, string> } };
  await openInLocale(page, request, "de");

  // L-7
  await openSettingsSection(page, 0);
  const language = page.locator("section.panel", { has: page.locator("#settings-language-select") });
  await expect(language.locator(".panel-header h2")).toHaveText("Sprache und Uhrzeit");
  await expect(language.locator(".panel-header h2")).toHaveText(common.languagePanel.title);
  const clock = page.getByRole("radiogroup", { name: common.languagePanel.clockFormatAria });
  await expect(clock).toBeVisible();
  await expect(page.locator("#settings-clock-format-label")).toBeVisible();
  await expect(page.locator("#settings-clock-format-label")).toHaveText(common.languagePanel.clockFormatAria);
  const scale = page.getByRole("radiogroup", { name: common.settingsTab.appearance.uiScaleAria });
  await expect(scale).toBeVisible();
  await expect(page.locator("#settings-ui-scale-label")).toBeVisible();

  // L-4: the welcome on a library with no chat model names no model family.
  expect(chat.emptyBanner.welcomeBody).not.toMatch(/Qwen|Llama/);
  expect(catalogue("en", "chat")).toMatchObject({ emptyBanner: { welcomeBody: expect.not.stringMatching(/Qwen|Llama/) } });

  // L-5: Compare (the second tab of the Chat group): slot names, the picker's title and its confirm button.
  const nav = page.locator("aside.sidebar nav");
  await openMenuIfCompact(page);
  const chatGroup = nav.locator(".nav-group-header").first();
  if ((await chatGroup.getAttribute("aria-expanded")) !== "true") await chatGroup.click();
  await nav.locator(".nav-group.open .nav-child-button").nth(1).click();
  await expect(page.getByText(chat.compareView.title, { exact: true })).toBeVisible();
  // The two result panels are titled with the slot name too.
  await expect(page.locator(".panel-header h2", { hasText: /^Modell [AB]$/ })).toHaveCount(2);
  await expect(page.locator(".eyebrow", { hasText: /^Modell A$/ })).toBeVisible();
  await expect(page.getByText("Model A", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: chat.compareView.select, exact: true }).first().click();
  const picker = page.getByRole("dialog");
  await expect(picker.locator(".modal-header h3")).toHaveText("Modell A auswählen");
  await expect(picker.getByRole("button", { name: "Für Modell A verwenden", exact: true })).toBeVisible();
  await expect(picker.getByText(/Use for Model A|Select Model A/)).toHaveCount(0);
  await page.keyboard.press("Escape");
});
