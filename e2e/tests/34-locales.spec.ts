import { readFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext, Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, openApp, openMenuIfCompact } from "./helpers";

// Review stream D: languages.
// 1. First launch on a system that is not English: the language banner must not break the shell. It is a child
//    of the two-column grid and used to take the sidebar's cell (the app was unusable until it was answered).
// 2. Every shipped locale boots with no raw key, no leftover {placeholder} and no sideways scroll.
// 3. A pseudo-locale (accented English, about 40 % longer) on the widest surfaces: Settings General, Storage,
//    Diagnostics and Accelerators (which hosts the Setup install strings), at the smallest window and at the
//    default one. The German catalogue requests are answered with the pseudo text, since the app has no
//    ?devLocale switch.
// Every test puts the saved language back to "system".

test.setTimeout(180_000);

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };
const LOCALES = ["zh-CN", "zh-TW", "ja", "de", "ru", "ko", "fr", "es", "pt-BR", "tlh"];
const EN_DIR = path.resolve(__dirname, "..", "..", "src", "locales", "en");

async function saveLocale(request: APIRequestContext, locale: string): Promise<void> {
  const response = await request.patch(`${API_BASE}/api/settings`, { headers, data: { locale } });
  expect(response.ok(), `PATCH locale ${locale} -> ${response.status()}`).toBeTruthy();
}

test.afterEach(async ({ request }) => {
  await saveLocale(request, "system");
});

/** Text on the page that looks like an i18n key or an unfilled placeholder. */
async function rawText(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const found: string[] = [];
    const walker = document.createTreeWalker(document.querySelector(".app-shell") ?? document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const text = (walker.currentNode.nodeValue ?? "").trim();
      const parent = walker.currentNode.parentElement;
      if (!text || !parent || parent.closest("pre, code, .mono-text, script, style")) continue;
      // A key: dotted camelCase words and nothing else, not a file name or a host.
      if (/^[a-zA-Z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$/.test(text) && !/\.(py|sh|json|md|gguf|safetensors|wav|mp4|png|app|cpp|com|ai|io)$/i.test(text)) found.push(`key: ${text}`);
      if (/\{\{?[a-zA-Z_]+\}?\}/.test(text)) found.push(`placeholder: ${text.slice(0, 100)}`);
    }
    return [...new Set(found)];
  });
}

async function sideways(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
}

async function openSettings(page: Page): Promise<void> {
  await openMenuIfCompact(page);
  // Settings is the last standalone entry in every language.
  await page.locator("aside.sidebar nav .nav-button").last().click();
  await expect(page.locator(".settings-subtab-bar")).toBeVisible();
}

async function settingsSection(page: Page, index: number): Promise<void> {
  const tab = page.locator(".settings-subtab-bar [role=tab]").nth(index);
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

test.describe("first launch on a German system", () => {
  test.use({ locale: "de-DE", viewport: { width: 1400, height: 920 } });

  test("the language banner sits above the app and does not take the sidebar's place", async ({ page, request }) => {
    await saveLocale(request, "system");
    await page.goto("/");
    await expect(page.locator("aside.sidebar")).toBeVisible();
    const banner = page.locator(".locale-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("Deutsch");
    // The app already starts in the detected language, so the banner says so in that language and does not
    // ask whether to switch to it (L-1: it read "App auf Deutsch umstellen?" on an app already in German).
    await expect(page.locator("html")).toHaveAttribute("lang", "de");
    await expect(banner.locator("strong")).toHaveText("Die App verwendet Deutsch, die Sprache des Systems.");
    await expect(banner).not.toContainText("?");
    await expect(banner).not.toContainText("umstellen");
    await expect(banner.getByRole("button")).toHaveText(["Deutsch beibehalten", "Use English"]);

    const bar = (await banner.boundingBox())!;
    const sidebar = (await page.locator("aside.sidebar").boundingBox())!;
    const main = (await page.locator("main.workspace").boundingBox())!;
    expect(bar.width).toBeGreaterThanOrEqual(1399);
    expect(bar.height).toBeLessThanOrEqual(90);
    expect(sidebar.x).toBe(0);
    expect(sidebar.width).toBeLessThanOrEqual(260);
    expect(sidebar.y).toBeGreaterThanOrEqual(bar.height - 1);
    expect(main.x).toBeGreaterThanOrEqual(sidebar.width - 1);
    expect(main.width).toBeGreaterThanOrEqual(1100);
    expect(main.y).toBeLessThanOrEqual(bar.height + 1);
    // Both buttons are styled buttons.
    await expect(banner.getByRole("button").first()).toHaveClass(/primary-button/);
    await expect(banner.getByRole("button").last()).toHaveClass(/secondary-button/);

    // "Use English" (always written in English) switches at once and is saved.
    const saved = page.waitForRequest((r) => r.method() === "PATCH" && r.url().endsWith("/api/settings"));
    await banner.getByRole("button").last().click();
    expect((await saved).postDataJSON()).toEqual({ locale: "en" });
    await expect(banner).toBeHidden();
    await expect(page.locator("html")).toHaveAttribute("lang", "en", { timeout: 2000 });
    await expect(page.locator("aside.sidebar nav .nav-button").first()).toContainText("Dashboard");
  });

  test("keeping the detected language saves it and the app stays in German", async ({ page, request }) => {
    await saveLocale(request, "system");
    await page.goto("/");
    const banner = page.locator(".locale-banner");
    await expect(banner).toBeVisible();
    const saved = page.waitForRequest((r) => r.method() === "PATCH" && r.url().endsWith("/api/settings"));
    await banner.getByRole("button", { name: "Deutsch beibehalten" }).click();
    expect((await saved).postDataJSON()).toEqual({ locale: "de" });
    await expect(banner).toBeHidden();
    await expect(page.locator("html")).toHaveAttribute("lang", "de");
    // Answered once: it does not come back on the next load.
    await page.reload();
    await expect(page.locator("aside.sidebar")).toBeVisible();
    await expect(page.locator(".locale-banner")).toHaveCount(0);
  });
});

for (const locale of LOCALES) {
  test(`${locale}: boots with no raw keys, no leftover placeholders and no sideways scroll`, async ({ page, request }) => {
    await saveLocale(request, locale);
    await openApp(page).catch(() => undefined); // the terms button is not "I agree" in this language; they are accepted already
    await expect(page.locator("aside.sidebar")).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("lang", locale, { timeout: 15_000 });
    await expect(page.locator(".error-boundary")).toHaveCount(0);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));

    const look = async (label: string) => {
      await page.waitForTimeout(400);
      await expect(page.locator(".error-boundary"), `${label} crashed`).toHaveCount(0);
      expect.soft(await rawText(page), `${locale} ${label}: raw keys or placeholders`).toEqual([]);
      expect.soft(await sideways(page), `${locale} ${label}: scrolls sideways`).toBeLessThanOrEqual(0);
    };
    await look("Dashboard");
    const nav = page.locator("aside.sidebar nav");
    const groups = nav.locator(".nav-group-header");
    for (let group = 0; group < 5; group++) { // Chat, Images, Video, Voice, Music and Audio
      const header = groups.nth(group);
      if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
      const children = nav.locator(".nav-group.open .nav-child-button");
      const count = await children.count();
      for (let child = 0; child < count; child++) {
        await children.nth(child).click();
        await look(`group ${group} tab ${child}`);
      }
    }
    await openSettings(page);
    for (let section = 0; section < 8; section++) {
      await settingsSection(page, section);
      await look(`Settings section ${section}`);
    }
    expect(errors, "uncaught errors").toEqual([]);
  });
}

// ── Pseudo-locale ────────────────────────────────────────────────────────────

const ACCENTS: Record<string, string> = { a: "å", e: "é", i: "î", o: "ö", u: "ü", A: "Å", E: "É", I: "Î", O: "Ö", U: "Ü", c: "ç", n: "ñ", s: "š", y: "ý" };

/** Accents outside {placeholders} and about 40 % padding, in brackets so a cut string is easy to spot. */
function pseudo(text: string): string {
  let depth = 0;
  let out = "";
  for (const ch of text) {
    if (ch === "{") depth += 1;
    out += depth === 0 ? (ACCENTS[ch] ?? ch) : ch;
    if (ch === "}") depth = Math.max(0, depth - 1);
  }
  return `[${out}${"~".repeat(Math.ceil(text.replace(/\{[^}]*\}/g, "").length * 0.4))}]`;
}

function pseudoTree(node: unknown): unknown {
  if (typeof node === "string") return pseudo(node);
  if (node && typeof node === "object") return Object.fromEntries(Object.entries(node).map(([key, value]) => [key, pseudoTree(value)]));
  return node;
}

async function servePseudoAsGerman(page: Page): Promise<void> {
  await page.route(/\/src\/locales\/de\/[a-z]+\.json/, (route) => {
    const namespace = /\/de\/([a-z]+)\.json/.exec(route.request().url())![1];
    const english = JSON.parse(readFileSync(path.join(EN_DIR, `${namespace}.json`), "utf8"));
    return route.fulfill({ contentType: "application/javascript", body: `export default ${JSON.stringify(pseudoTree(english))};` });
  });
}

/** Controls whose text is wider than their box with no ellipsis, and things past the right edge. */
async function clipped(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const width = window.innerWidth;
    const cut = [...document.querySelectorAll(".app-shell button, .app-shell .badge, .app-shell .subtab, .app-shell h2, .app-shell h3, .app-shell th")]
      .filter((el) => el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).textOverflow !== "ellipsis")
      .map((el) => `cut: ${el.tagName.toLowerCase()} "${(el.textContent ?? "").trim().slice(0, 30)}" ${el.scrollWidth}/${el.clientWidth}`);
    const wide = [...document.querySelectorAll("main *")]
      .filter((el) => {
        const box = el.getBoundingClientRect();
        if (!(box.width > 0 && box.right > width + 1)) return false;
        for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
          if (getComputedStyle(p).overflowX !== "visible") return p.getBoundingClientRect().right > width + 1;
        }
        return true;
      })
      .map((el) => `past the edge: ${el.tagName.toLowerCase()}.${String(el.className).slice(0, 30)}`);
    return [...new Set([...cut, ...wide])].slice(0, 8);
  });
}

for (const [width, height] of [[1100, 760], [1400, 920]] as const) {
  test.describe(`pseudo-locale at ${width} x ${height}`, () => {
    test.use({ viewport: { width, height } });

    test("Settings General, Storage, Diagnostics and Accelerators hold text 40 % longer", async ({ page }) => {
      await servePseudoAsGerman(page);
      await openApp(page);
      await openSettings(page);
      await page.locator("#settings-language-select").selectOption("de");
      // The shell re-renders in the pseudo text.
      await expect(page.locator(".settings-subtab-bar [role=tab]").first()).toHaveText(/^\[Géñérål~+\]$/);
      for (const [index, label] of [[0, "General"], [1, "Storage"], [5, "Diagnostics"], [6, "Accelerators"]] as const) {
        await settingsSection(page, index);
        await page.waitForTimeout(500);
        expect.soft(await sideways(page), `${label}: scrolls sideways`).toBeLessThanOrEqual(0);
        expect.soft(await clipped(page), `${label}: clipped text`).toEqual([]);
      }
      // Back to English for the specs that follow.
      await settingsSection(page, 0);
      await page.locator("#settings-language-select").selectOption("system");
      await expect(page.locator(".settings-subtab-bar [role=tab]").first()).toHaveText("General");
    });
  });
}
