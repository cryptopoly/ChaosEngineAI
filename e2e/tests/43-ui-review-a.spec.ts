import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab } from "./helpers";

// UI/UX review of 2026-11-20 (docs/review/ui-ux-v0100-a, FU-450, FU-453, FU-455): small defects found on the Chat
// and Chat Models pages. No model runs: where one is needed the workspace answer says it is loaded.
//
// 1. First run (nothing loaded, no thread): no green "Ready" badge; the model picker says no model is downloaded
//    instead of "No models match your search"; the Models header says "0 GB on disk", not "Unknown on disk".
// 2. A long unbroken word in a reply wraps inside its bubble.
// 3. The Keep going panel opens inside the window and does not scroll the thread sideways.

const LONG = "an_unbroken_word_".repeat(14);
const LOADED = {
  state: "loaded",
  loadedModel: {
    ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "llama.cpp", source: "library",
    engine: "llama.cpp", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
    contextTokens: 8192, loadedAt: "2026-11-20 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
    speculativeDecoding: false, treeBudget: 0,
    capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
  },
};
const SESSION = {
  id: "session-review-a", title: "Long word", updatedAt: "2026-11-20 10:00:00", pinned: false, model: "Mock Chat 8B",
  modelRef: "mock/chat-8b", thinkingMode: "off", cacheLabel: "Native f16",
  messages: [{ role: "user", text: "Say a long word", metrics: null }, { role: "assistant", text: LONG, metrics: null }],
};

type Workspace = { runtime: Record<string, unknown>; chatSessions: Record<string, unknown>[] };

async function mockLoaded(page: Page, sessions: Record<string, unknown>[] = []) {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as Workspace;
    body.runtime = { ...body.runtime, ...LOADED };
    body.chatSessions = [...sessions, ...body.chatSessions];
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

test("first run: no Ready badge, the picker names the empty library, the Models header says 0 GB", async ({ page }) => {
  // The run's library is empty; pin the two answers this test reads so a later spec's leftovers cannot change it.
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as Workspace & { library: unknown[] };
    body.library = [];
    body.chatSessions = [];
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Chat", "Chat");
  await expect(page.getByRole("button", { name: "Select Model" })).toBeVisible();
  await expect(page.locator(".badge.success", { hasText: "Ready" })).toHaveCount(0);
  await page.getByRole("button", { name: "Select Model" }).click();
  const picker = page.getByRole("dialog", { name: "Select Model" });
  await expect(picker.locator(".model-select-empty")).toContainText("No chat models downloaded yet.");
  await expect(picker.locator(".model-select-empty")).not.toContainText("match your search");
  await page.keyboard.press("Escape");
  await openTab(page, "Chat", "Models");
  const header = page.locator(".chat-models-page").first();
  await expect(header).toContainText("0 GB on disk");
  await expect(header).not.toContainText("Unknown on disk");
});

test("a long unbroken word wraps inside the reply", async ({ page }) => {
  await mockLoaded(page, [SESSION]);
  await goTo(page, "Chat", "Chat");
  await page.locator(".session-button", { hasText: "Long word" }).click();
  const bubble = page.locator(".message-bubble.assistant").last();
  await expect(bubble).toContainText("an_unbroken_word_");
  const overflow = await bubble.evaluate((el) => {
    const text = el.querySelector(".markdown-content") as HTMLElement;
    return text.scrollWidth - text.clientWidth;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

for (const width of [1280, 1100]) {
  test(`the Keep going panel opens inside a ${width} px window`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await mockLoaded(page);
    await goTo(page, "Chat", "Chat");
    await page.getByRole("switch", { name: "Studio" }).click();
    await page.locator(".composer .keep-going__toggle").click();
    const panel = page.locator(".keep-going-panel");
    await expect(panel).toBeVisible();
    const box = (await panel.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(width + 1);
    const scrolled = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>("main, main *")].filter((el) => el.scrollLeft > 0).length);
    expect(scrolled).toBe(0);
    await page.keyboard.press("Escape");
    // Leave the remembered Studio switch as the other specs expect it.
    await page.getByRole("switch", { name: "Studio ON" }).click();
  });
}
