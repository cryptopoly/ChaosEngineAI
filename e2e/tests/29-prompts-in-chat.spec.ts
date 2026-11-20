import { mkdirSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { API_BASE, TOKEN, goTo, openApp, primaryNav, SHOTS_DIR } from "./helpers";

// D-87: the prompt library lives in Chat (a Prompts panel beside the composer) and the Plugins feature is gone.
// Runs against the run's own backend (throwaway HOME), so the real prompt store answers; no model runs. Chat is
// faked as in 16-skills: a model looks loaded and the chat stream is a page-side fetch wrapper, so the test reads
// what the composer sends.
//
// The panel: the five built-ins, search focused on open, Up and Down between the search box and the rows, a save
// with no name says why, a template with a variable is created, filled in and previewed, "Use as system prompt"
// is sent as `systemPrompt` and shown in the panel with "Stop using it", "Insert into message" fills the composer,
// "Save message as prompt" starts from the typed message, edit, delete with a confirm, Escape steps back and then
// closes with focus on the button, 1280 and 375 px. The sidebar has no Prompts or Plugins entry, a remembered
// Prompts id opens Chat and a request for Plugins opens the Dashboard.

const AUTH = { Authorization: `Bearer ${TOKEN}` };
const NAME = "E2E chat template";

async function removeTemplates(page: Page) {
  const response = await page.request.get(`${API_BASE}/api/prompts`, { headers: AUTH });
  const { templates } = (await response.json()) as { templates: Array<{ id: string; name: string }> };
  for (const template of templates.filter((t) => t.name.startsWith(NAME))) {
    await page.request.delete(`${API_BASE}/api/prompts/${template.id}`, { headers: AUTH });
  }
}

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__promptPayload = JSON.parse(String(init?.body ?? "{}"));
      const body = new ReadableStream<Uint8Array>({ start: (controller) => controller.close() });
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function mockLoadedModel(page: Page) {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as { runtime: Record<string, unknown> };
    body.runtime = {
      ...body.runtime,
      state: "loaded",
      loadedModel: {
        ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "mlx", source: "library",
        engine: "mlx", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
        contextTokens: 8192, loadedAt: "2026-11-20 08:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

const trigger = (page: Page) => page.locator(".composer .chat-prompts__trigger");
const panel = (page: Page) => page.getByRole("dialog", { name: "Prompts" });
const rows = (page: Page) => panel(page).locator(".chat-prompts__row");
const composerBox = (page: Page) => page.locator(".composer textarea");

async function sentPayload(page: Page, text: string): Promise<Record<string, unknown>> {
  await page.evaluate(() => { (window as unknown as { __promptPayload?: unknown }).__promptPayload = undefined; });
  await composerBox(page).fill(text);
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __promptPayload?: unknown }).__promptPayload));
  return page.evaluate(() => (window as unknown as { __promptPayload: Record<string, unknown> }).__promptPayload);
}

test.afterEach(async ({ page }) => {
  await removeTemplates(page);
});

test("the sidebar has no Prompts or Plugins entry", async ({ page }) => {
  await openApp(page);
  const nav = await primaryNav(page);
  for (const header of await nav.locator(".nav-group-header").all()) {
    if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
  }
  await expect(nav.getByText(/^(Prompts|Plugins)$/)).toHaveCount(0);
  expect((await page.request.get(`${API_BASE}/api/plugins`, { headers: AUTH })).status()).toBe(404);
});

test("Prompts in Chat: browse, search, keyboard, create, use as system prompt, insert, save message, edit, delete", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  await scriptChatStream(page);
  await mockLoadedModel(page);
  await goTo(page, "Chat", "Chat");
  await removeTemplates(page);

  await expect(trigger(page)).toHaveAttribute("aria-expanded", "false");
  await trigger(page).click();
  await expect(trigger(page)).toHaveAttribute("aria-expanded", "true");
  await expect(rows(page)).toHaveCount(5);
  const search = panel(page).getByRole("searchbox", { name: "Search templates..." });
  await expect(search).toBeFocused();

  // Up and Down move between the search box and the rows; Enter opens a row; Escape steps back, then closes.
  await page.keyboard.press("ArrowDown");
  await expect(rows(page).first()).toBeFocused();
  await page.keyboard.press("ArrowUp");
  await expect(search).toBeFocused();
  await page.keyboard.press("ArrowUp");
  await expect(rows(page).last()).toBeFocused();
  await rows(page).filter({ hasText: "Translator" }).focus();
  await page.keyboard.press("Enter");
  await expect(panel(page).getByTestId("chat-prompts-preview")).toContainText("You are a professional translator.");
  await expect(panel(page).getByText(/^Created: /)).toHaveText(/Created: .*20\d\d.*\| Updated: .*20\d\d/);
  await page.keyboard.press("Escape");
  await expect(rows(page)).toHaveCount(5);
  await page.keyboard.press("Escape");
  await expect(panel(page)).toHaveCount(0);
  await expect(trigger(page)).toBeFocused();

  // Nothing typed: "Save message as prompt" is off. New with no name is refused with a sentence; bad variables too.
  await trigger(page).click();
  await expect(panel(page).getByRole("button", { name: "Save message as prompt" })).toBeDisabled();
  await panel(page).getByRole("button", { name: "New", exact: true }).click();
  await panel(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(panel(page).getByRole("alert")).toHaveText("Give the template a name before saving.");
  await panel(page).getByRole("textbox", { name: "Name", exact: true }).fill(NAME);
  await panel(page).getByRole("textbox", { name: "Variables (JSON array)" }).fill('[{"name": "language"');
  await panel(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(panel(page).getByRole("alert")).toContainText("Variables must be a JSON array");

  // A template with one variable.
  await panel(page).getByRole("textbox", { name: "Category", exact: true }).fill("Review");
  await panel(page).getByRole("textbox", { name: "Tags (comma-separated)" }).fill("e2e, review");
  await panel(page).getByRole("textbox", { name: "Prompt", exact: true }).fill("Answer only in {{language}}. Keep quokkas in mind.");
  await panel(page).getByRole("textbox", { name: "Variables (JSON array)" })
    .fill('[{"name": "language", "type": "string", "default": "French"}]');
  await panel(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(panel(page)).toContainText("1 variable");
  await expect(panel(page).getByTestId("chat-prompts-preview")).toHaveText("Answer only in French. Keep quokkas in mind.");
  await panel(page).getByRole("textbox", { name: "language" }).fill("Welsh");
  await expect(panel(page).getByTestId("chat-prompts-preview")).toHaveText("Answer only in Welsh. Keep quokkas in mind.");
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-prompts-chat-1280-detail.png") });

  // Use as system prompt: the panel and the button show it, and the next message carries it.
  await panel(page).getByRole("button", { name: "Use as system prompt" }).click();
  const inUse = panel(page).locator(".chat-prompts__in-use");
  await expect(inUse).toContainText("Chat is using this system prompt in every thread");
  await expect(inUse).toContainText("Answer only in Welsh. Keep quokkas in mind.");
  await expect(trigger(page)).toHaveClass(/chat-prompts__trigger--active/);
  await panel(page).getByRole("button", { name: "Close" }).click();
  expect((await sentPayload(page, "hello")).systemPrompt).toBe("Answer only in Welsh. Keep quokkas in mind.");

  // Stop using it: the next message has none.
  await trigger(page).click();
  await panel(page).getByRole("button", { name: "Stop using it" }).click();
  await expect(panel(page).locator(".chat-prompts__in-use")).toHaveCount(0);
  await panel(page).getByRole("button", { name: "Close" }).click();
  expect((await sentPayload(page, "hello again")).systemPrompt).toBeUndefined();

  // Search reads the prompt text; Insert puts the filled text into the composer after what is typed.
  await composerBox(page).fill("Please help.");
  await trigger(page).click();
  await search.fill("quokka");
  await expect(rows(page)).toHaveCount(1);
  await search.fill("no-such-template-text");
  await expect(rows(page)).toHaveCount(0);
  await expect(panel(page)).toContainText("No template matches the search.");
  await search.fill("quokka");
  await rows(page).first().click();
  await panel(page).getByRole("textbox", { name: "language" }).fill("Basque");
  await panel(page).getByRole("button", { name: "Insert into message" }).click();
  await expect(panel(page)).toHaveCount(0);
  await expect(composerBox(page)).toHaveValue("Please help.\n\nAnswer only in Basque. Keep quokkas in mind.");
  await expect(composerBox(page)).toBeFocused();

  // Save message as prompt starts from the typed message.
  await composerBox(page).fill(`${NAME} from message\nSecond line of the message.`);
  await trigger(page).click();
  await panel(page).getByRole("button", { name: "Save message as prompt" }).click();
  await expect(panel(page).getByRole("textbox", { name: "Name", exact: true })).toHaveValue(`${NAME} from message`);
  await expect(panel(page).getByRole("textbox", { name: "Prompt", exact: true })).toHaveValue(`${NAME} from message\nSecond line of the message.`);
  await panel(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(panel(page).getByTestId("chat-prompts-preview")).toContainText("Second line of the message.");

  // Edit, then delete with a confirm; the backend agrees.
  await panel(page).getByRole("button", { name: "Edit", exact: true }).click();
  await panel(page).getByRole("textbox", { name: "Name", exact: true }).fill(`${NAME} v2`);
  await panel(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(panel(page).locator(".chat-prompts__title")).toHaveText(`${NAME} v2`);
  const asked: string[] = [];
  page.once("dialog", (dialog) => { asked.push(dialog.message()); void dialog.dismiss(); });
  await panel(page).getByRole("button", { name: "Delete", exact: true }).click();
  expect(asked[0]).toBe(`Delete the template "${NAME} v2"? This cannot be undone.`);
  await expect(panel(page).locator(".chat-prompts__title")).toHaveText(`${NAME} v2`);
  page.once("dialog", (dialog) => void dialog.accept());
  await panel(page).getByRole("button", { name: "Delete", exact: true }).click();
  await expect(rows(page)).toHaveCount(1);
  await search.fill("");
  await expect(rows(page)).toHaveCount(6);
  const after = (await (await page.request.get(`${API_BASE}/api/prompts`, { headers: AUTH })).json()) as { templates: Array<{ name: string }> };
  expect(after.templates.some((t) => t.name === `${NAME} v2`)).toBe(false);
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-prompts-chat-1280-list.png") });
});

test("Prompts in Chat fits a 375 px window", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await mockLoadedModel(page);
  await goTo(page, "Chat", "Chat");
  await trigger(page).scrollIntoViewIfNeeded();
  await trigger(page).click();
  await expect(rows(page)).toHaveCount(5);
  const box = (await panel(page).boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(375);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.y + box.height).toBeLessThanOrEqual(812);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-prompts-chat-375-list.png") });
  await rows(page).filter({ hasText: "Summarizer" }).click();
  await expect(panel(page).getByTestId("chat-prompts-preview")).toContainText("concise summarization assistant");
  await panel(page).getByRole("button", { name: "Insert into message" }).scrollIntoViewIfNeeded();
  await expect(panel(page).getByRole("button", { name: "Insert into message" })).toBeInViewport();
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-prompts-chat-375-detail.png") });
});

test("a failed load says so instead of looking empty", async ({ page }) => {
  await page.route("**/api/prompts", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await goTo(page, "Chat", "Chat");
  await trigger(page).click();
  await expect(panel(page).getByRole("alert")).toHaveText("The templates could not be loaded. Check that the backend is running.");
  await expect(panel(page)).not.toContainText("No templates yet.");
});

test("old ids: a remembered Prompts page opens Chat, a request for Plugins opens the Dashboard", async ({ page }) => {
  await page.addInitScript(() => {
    if (window.sessionStorage.getItem("e2e-prefs-set")) return;
    window.sessionStorage.setItem("e2e-prefs-set", "1");
    window.localStorage.setItem(
      "chaosengine.sidebar.prefs.v1",
      JSON.stringify({ mode: "tabs", lastChildByGroup: { chat: "prompt-library", tools: "plugins" } }),
    );
  });
  await openApp(page);
  await (await primaryNav(page)).locator(".nav-button", { hasText: /^Chat/ }).first().click();
  await expect(trigger(page)).toBeVisible();
  // A studio card's "Open in Library" asks for a tab by id; an old id still lands somewhere real.
  await page.evaluate(() => window.dispatchEvent(new CustomEvent("chaosengine:open-tab", { detail: "plugins" })));
  await expect(page.locator(".composer")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Dashboard", level: 2 })).toBeVisible();
});
