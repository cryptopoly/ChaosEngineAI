import type { APIRequestContext, Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, goTo, openProject, openProjectTab } from "./helpers";

// S-P: what an iPad in the simulator showed. A touch device is emulated (coarse pointer, touch events): two fingers
// moved together pan the Board without zooming, a long press selects no text, the tap targets of a project are
// 44 px, fields are 16 px (iOS zooms the page into smaller ones) and iOS text inflation is off. Also the Chat
// thread name field, which a model's capability badges squeezed to 0 px.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

async function sample(request: APIRequestContext, name: string): Promise<void> {
  const response = await request.post(`${API_BASE}/api/storyboard/sample`, { headers, data: { name } });
  expect(response.ok()).toBeTruthy();
}

/** Two touch points down, moved together in `steps`, up: the browser turns them into touch pointer events. */
async function twoFingers(page: Page, from: [number, number][], to: [number, number][], steps = 6): Promise<void> {
  const cdp = await page.context().newCDPSession(page);
  const at = (k: number) => from.map(([x, y], i) => ({ x: x + ((to[i][0] - x) * k) / steps, y: y + ((to[i][1] - y) * k) / steps, id: i }));
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: at(0) });
  for (let k = 1; k <= steps; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: at(k) });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}

async function planeOffset(page: Page): Promise<{ x: number; y: number }> {
  const transform = await page.locator(".board-plane").evaluate((el) => (el as HTMLElement).style.transform);
  const [, x, y] = /translate\(([-\d.]+)px, ([-\d.]+)px\)/.exec(transform) ?? [];
  return { x: Number(x), y: Number(y) };
}

test.describe("touch screen", () => {
  test.use({ hasTouch: true, isMobile: true, viewport: { width: 1280, height: 900 } });

  test("Board: two fingers pan without zooming, a pinch zooms, a long press selects no text", async ({ page, request }) => {
    await sample(request, "Touch film");
    await openProject(page, "Touch film", "Board");
    const zoom = page.locator(".board-toolbar__zoom");
    await expect(zoom).toHaveText(/\d+%/);
    // The board fits its cards once laid out: wait for the zoom to settle before measuring from it.
    let start = 0;
    await expect.poll(async () => {
      const now = parseInt(await zoom.innerText(), 10);
      const settled = now === start;
      start = now;
      return settled;
    }, { intervals: [300] }).toBe(true);
    const box = (await page.locator(".board-viewport").boundingBox())!;
    const y = box.y + box.height - 40; // below the cards: the empty board
    const before = await planeOffset(page);
    await twoFingers(page, [[box.x + 300, y], [box.x + 400, y]], [[box.x + 380, y - 30], [box.x + 480, y - 30]]);
    await expect(zoom).toHaveText(`${start}%`);
    const after = await planeOffset(page);
    expect(Math.round(after.x - before.x)).toBe(80);
    expect(Math.round(after.y - before.y)).toBe(-30);
    await twoFingers(page, [[box.x + 300, y], [box.x + 400, y]], [[box.x + 250, y], [box.x + 450, y]]);
    await expect(zoom).toHaveText(`${Math.round(start * 2)}%`);
    const style = await page.locator(".board-viewport").evaluate((el) => {
      const s = getComputedStyle(el) as CSSStyleDeclaration & { webkitUserSelect: string };
      return { select: s.webkitUserSelect || s.userSelect, touch: s.touchAction };
    });
    expect(style).toEqual({ select: "none", touch: "none" });
  });

  test("a project's tap targets are 44 px and fields 16 px; no iOS text inflation", async ({ page, request }) => {
    await sample(request, "Touch targets");
    await openProject(page, "Touch targets", "Storyboard");
    expect(await page.evaluate(() => matchMedia("(pointer: coarse)").matches)).toBe(true);
    for (const tab of ["Storyboard", "Board", "Plan"] as const) {
      await openProjectTab(page, tab);
      const small = await page.evaluate(() =>
        [...document.querySelectorAll(".project-shell :is(.primary-button, .secondary-button, .icon-button, .project-shell__tab)")]
          .map((el) => ({ el, r: el.getBoundingClientRect() }))
          .filter(({ r }) => r.width > 0 && r.height > 0 && (r.height < 43.5 || r.width < 43.5))
          .map(({ el, r }) => `${(el.textContent || el.getAttribute("aria-label") || "").trim()} ${Math.round(r.width)}x${Math.round(r.height)}`),
      );
      expect(small, tab).toEqual([]);
    }
    const brief = page.getByLabel("Brief");
    expect(parseFloat(await brief.evaluate((el) => getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(16);
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("-webkit-text-size-adjust"))).toBe("100%");
  });
});

test("Chat: the thread name field keeps its width beside a model's capability badges", async ({ page }) => {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as { runtime: Record<string, unknown> };
    body.runtime = {
      ...body.runtime,
      state: "loaded",
      loadedModel: {
        ref: "mock/chat-27b", name: "Mock Chat 27B", canonicalRepo: "mock/chat-27b", backend: "mlx", source: "library",
        engine: "mlx", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
        contextTokens: 8192, loadedAt: "2026-11-16 10:00:00", path: "/mock/chat-27b", runtimeTarget: "/mock/chat-27b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: true, supportsReasoning: true, supportsCoding: true, supportsAgents: true },
      },
    };
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Chat", "Chat");
  const field = page.locator(".thread-title-field");
  await expect(field).toBeVisible();
  expect((await field.boundingBox())!.width).toBeGreaterThanOrEqual(160);
  const input = (await field.locator("input").boundingBox())!;
  const actions = (await page.locator(".thread-toolbar-actions").boundingBox())!;
  const overlap = input.y < actions.y + actions.height && actions.y < input.y + input.height
    && input.x < actions.x + actions.width && actions.x < input.x + input.width;
  expect(overlap).toBe(false);
});
