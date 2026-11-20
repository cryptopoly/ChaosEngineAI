// UI/UX review, checks: the Keep going panel stays inside the window at several widths, and a count of controls
// with no accessible name and fields with no label on each page in my areas. Prints JSON; not a behaviour test.
import { writeFileSync } from "node:fs";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openApp, openTab } from "./helpers";
import { shoot, top } from "./uxaShared";

const LOADED = {
  state: "loaded",
  loadedModel: {
    ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "llama.cpp", source: "library",
    engine: "llama.cpp", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
    contextTokens: 8192, loadedAt: "2026-11-20 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
    speculativeDecoding: false, treeBudget: 0, capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
  },
};

async function mockWorkspace(page: Page) {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as { runtime: Record<string, unknown> };
    body.runtime = { ...body.runtime, ...LOADED };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

for (const width of [1280, 1100, 900, 700]) {
  test(`keep going panel stays inside a ${width} px window`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await mockWorkspace(page);
    await goTo(page, "Chat", "Chat");
    await page.getByRole("switch", { name: "Studio" }).click();
    await page.locator(".composer button", { hasText: "Keep going" }).first().click();
    const panel = page.locator(".keep-going-panel");
    await expect(panel).toBeVisible();
    await page.waitForTimeout(400);
    const box = await panel.boundingBox();
    const scrolled = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>("main, main *")].filter((el) => el.scrollLeft > 0).map((el) => el.className));
    if (width === 1280) await shoot(page, "30-keep-going-panel-after-fix", false, false);
    console.log(`KG ${width}: x=${box?.x.toFixed(0)} right=${((box?.x ?? 0) + (box?.width ?? 0)).toFixed(0)} scrolled=${JSON.stringify(scrolled)}`);
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width + 1);
    expect(scrolled).toEqual([]);
  });
}

async function probe(page: Page) {
  return page.evaluate(() => {
    const name = (el: Element) =>
      (el.getAttribute("aria-label") || el.getAttribute("title") || (el as HTMLElement).innerText || "").trim() ||
      (el.getAttribute("aria-labelledby") ? "x" : "");
    const visible = (el: Element) => (el as HTMLElement).offsetParent !== null;
    const buttons = [...document.querySelectorAll("main button, main [role=button]")].filter(visible).filter((el) => !name(el));
    const fields = [...document.querySelectorAll<HTMLInputElement>("main input, main select, main textarea")].filter(visible).filter((el) => {
      if (el.type === "hidden") return false;
      if (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby") || el.closest("label")) return false;
      if (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)) return false;
      return true;
    });
    const clickDivs = [...document.querySelectorAll<HTMLElement>("main div[onclick], main span[onclick]")].length;
    return {
      unnamedButtons: buttons.map((el) => el.className).slice(0, 8),
      unlabelledFields: fields.map((el) => `${el.tagName.toLowerCase()}.${el.className}[${el.getAttribute("placeholder") ?? ""}]`).slice(0, 12),
      clickDivs,
    };
  });
}

test("accessible names and labels on each page", async ({ page }) => {
  const out: Record<string, unknown> = {};
  await openApp(page);
  await top(page, "Dashboard");
  out.dashboard = await probe(page);
  for (const child of ["Chat", "Compare", "HTML Challenge", "Models"]) {
    await openTab(page, "Chat", child);
    await page.waitForTimeout(500);
    out[`chat-${child}`] = await probe(page);
  }
  await top(page, "Server");
  out.server = await probe(page);
  await page.getByRole("tab", { name: "Activity" }).click();
  out["server-activity"] = await probe(page);
  await top(page, "Settings");
  for (const s of ["General", "Storage", "Providers", "Integrations", "Skills", "Diagnostics", "Accelerators", "Disk usage"]) {
    await page.getByRole("tablist", { name: "Settings sections" }).getByRole("tab", { name: s, exact: true }).click();
    await page.waitForTimeout(800);
    out[`settings-${s}`] = await probe(page);
  }
  // Keyboard: the first Tab stops from the top of the page, by accessible name.
  await openTab(page, "Chat", "Chat");
  await page.locator("body").click({ position: { x: 5, y: 5 } });
  const stops: string[] = [];
  for (let i = 0; i < 14; i++) {
    await page.keyboard.press("Tab");
    stops.push(await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      return el ? `${el.tagName.toLowerCase()}:${(el.getAttribute("aria-label") || el.innerText || el.getAttribute("placeholder") || "").trim().slice(0, 30)}` : "none";
    }));
  }
  out.chatTabOrder = stops;
  writeFileSync("E:/tmp/uxa/a11y.json", JSON.stringify(out, null, 1));
});
