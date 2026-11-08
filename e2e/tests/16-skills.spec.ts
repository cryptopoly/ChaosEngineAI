import { mkdirSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { API_BASE, TOKEN, goTo, openApp, openTab, SHOTS_DIR } from "./helpers";

// D-62: skills import. The import runs against the run's own backend (throwaway HOME) with the sample
// skill in docs/examples/skills, so the real validation and store are exercised; no model runs. Chat is
// faked the same way as 15-chat-studio-tools: a tool-capable model looks loaded and the chat stream is a
// page-side fetch wrapper, so the test only checks what the composer sends. Covers: import shows the
// description and file list, the skill starts off, switching it on shows the per-chat Skills switch,
// the switch sends `skills: true`, and Settings > Skills fits 375 px.

const SAMPLE = path.resolve(__dirname, "..", "..", "docs", "examples", "skills", "storyboard-house-style");
const SKILL_ID = "storyboard-house-style";

async function removeSkill(page: Page) {
  await page.request.delete(`${API_BASE}/api/skills/${SKILL_ID}`, { headers: { Authorization: `Bearer ${TOKEN}` } });
}

async function openSkillsSettings(page: Page) {
  await page.getByRole("navigation", { name: "Primary" }).getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("tablist", { name: "Settings sections" }).getByRole("tab", { name: "Skills" }).click();
  await expect(page.getByRole("heading", { name: "Skills", exact: true })).toBeVisible();
}

async function importSample(page: Page) {
  await page.getByRole("textbox", { name: "Skill folder or .zip path" }).fill(SAMPLE);
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText(
    "Imported storyboard-house-style: 2 files kept, nothing skipped. It is off until you switch it on.",
  );
  return page.getByRole("listitem", { name: "storyboard-house-style" });
}

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__skillsPayload = JSON.parse(String(init?.body ?? "{}"));
      const body = new ReadableStream<Uint8Array>({ start: () => undefined });
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
        contextTokens: 8192, loadedAt: "2026-11-08 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

test.afterEach(async ({ page }) => {
  await removeSkill(page);
});

test("import a skill, read its files, switch it on, and the chat sends skills: true", async ({ page }) => {
  await scriptChatStream(page);
  await mockLoadedModel(page);
  await openApp(page);
  await removeSkill(page);
  await openSkillsSettings(page);
  await expect(page.getByRole("note")).toContainText("Scripts in a skill are never run.");

  const card = await importSample(page);
  await expect(card).toContainText("House style for storyboard shots");
  const enabled = card.getByRole("checkbox", { name: "Available in Chat" });
  await expect(enabled).not.toBeChecked();
  await card.getByText("2 files").click();
  await expect(card).toContainText("palette.md");
  await expect(card).toContainText("SKILL.md");

  // A second import of the same skill is refused with the backend's sentence.
  await page.getByRole("textbox", { name: "Skill folder or .zip path" }).fill(SAMPLE);
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("already imported");

  // While the skill is off, Chat has no Skills switch.
  await openTab(page, "Chat", "Chat");
  await page.locator(".composer textarea").waitFor();
  await expect(page.getByRole("switch", { name: /^Skills/ })).toHaveCount(0);

  await openSkillsSettings(page);
  await enabled.click();  // the box follows the saved state, so it ticks once the backend answers
  await expect(enabled).toBeChecked();

  await openTab(page, "Chat", "Chat");
  // A new chat gets its id with the first message; the switch is per chat, so send once to make the thread.
  const skills = page.getByRole("switch", { name: "Skills" });
  if (!(await skills.isVisible())) {
    await page.locator(".composer textarea").fill("hello");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await page.waitForFunction(() => Boolean((window as unknown as { __skillsPayload?: unknown }).__skillsPayload));
    const first = await page.evaluate(() => (window as unknown as { __skillsPayload: Record<string, unknown> }).__skillsPayload);
    expect(first.skills).toBeUndefined();
    await page.reload();
    await page.locator("aside.sidebar").waitFor();
    await openTab(page, "Chat", "Chat");
  }
  await expect(skills).toHaveAttribute("aria-checked", "false");
  await skills.click();
  await expect(page.getByRole("switch", { name: "Skills ON" })).toHaveAttribute("aria-checked", "true");
  await page.evaluate(() => { (window as unknown as { __skillsPayload?: unknown }).__skillsPayload = undefined; });
  await page.locator(".composer textarea").fill("Plan three shots for a coffee advert");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __skillsPayload?: unknown }).__skillsPayload));
  const payload = await page.evaluate(() => (window as unknown as { __skillsPayload: Record<string, unknown> }).__skillsPayload);
  expect(payload.skills).toBe(true);
});

test("Settings > Skills fits a 375 px window", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await goTo(page, "Chat", "Chat");
  await removeSkill(page);
  await openSkillsSettings(page);
  const card = await importSample(page);
  await card.getByText("2 files").click();
  await page.waitForTimeout(300);
  mkdirSync(SHOTS_DIR, { recursive: true });
  // A tall window for the picture so the whole panel shows; the width checks below are what count.
  await page.setViewportSize({ width: 375, height: 2000 });
  await page.waitForTimeout(200);
  await page.locator(".skills-panel").screenshot({ path: path.join(SHOTS_DIR, "skills-settings-375.png") });
  const fit = await page.evaluate(() => {
    const width = window.innerWidth;
    const panel = document.querySelector(".skills-panel");
    const wide = panel ? [...panel.querySelectorAll("*")].filter((el) => el.getBoundingClientRect().right > width + 1).length : -1;
    return { sideways: document.documentElement.scrollWidth - width, wide };
  });
  expect(fit.sideways).toBeLessThanOrEqual(0);
  expect(fit.wide).toBe(0);
  const importButton = page.getByRole("button", { name: "Import", exact: true });
  await importButton.scrollIntoViewIfNeeded();
  await expect(importButton).toBeInViewport();
});
