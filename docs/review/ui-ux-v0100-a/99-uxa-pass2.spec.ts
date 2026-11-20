// UI/UX review, pass 2: Chat with a model that looks loaded, the composer's panels, the Studio plan card and job
// cards, Keep going, and Chat > Models rows in their states. Everything is mocked; no model runs or downloads.
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab } from "./helpers";
import { DESKTOP, PHONE, dumpTexts, shoot } from "./uxaShared";

const LOADED = {
  state: "loaded",
  loadedModel: {
    ref: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", name: "Nemotron 3.5 Lightning GGUF Q4_0 (24 GB card)",
    canonicalRepo: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", backend: "llama.cpp", source: "library",
    engine: "llama.cpp", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
    contextTokens: 16384, loadedAt: "2026-11-20 10:00:00", path: "E:\\AI Models\\nemotron.gguf", runtimeTarget: "E:\\AI Models\\nemotron.gguf",
    speculativeDecoding: false, treeBudget: 0,
    runtimeNote: "Context raised from 8,192 to 16,384 tokens: this model's template turns the Studio tool list into about 9,200 tokens.",
    capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: true },
  },
};
const LONG = "Pneumonoultramicroscopicsilicovolcanoconiosis_and_a_very_long_unbroken_token_that_should_wrap_or_scroll_without_pushing_the_layout_sideways_0123456789";
const SESSION = {
  id: "session-uxa", title: "Flower shop launch plan with a very long thread title that has to be cut somewhere sensible", updatedAt: "2026-11-20 10:00:00",
  pinned: false, model: LOADED.loadedModel.name, modelRef: LOADED.loadedModel.ref, thinkingMode: "off", cacheLabel: "Native f16",
  messages: [
    { role: "user", text: "Write a launch plan for a flower shop. " + LONG, metrics: null },
    {
      role: "assistant",
      text: "## Launch plan\n\n1. **Week 1**: window display and a poster.\n2. **Week 2**: opening offer.\n\n| Item | Cost |\n|---|---|\n| Poster | 40 |\n| Flyers | 25 |\n\n```python\nprint('hello flower shop')\n```\n\n" + LONG,
      metrics: { tokS: 233.1, completionTokens: 180, promptTokens: 9349, responseSeconds: 0.8, ttftSeconds: 0.2, finishReason: "stop", model: LOADED.loadedModel.name, backend: "llama.cpp", cacheLabel: "Native f16", contextTokens: 16384 },
    },
  ],
};
const PLAN = {
  planId: "plan-uxa", sessionId: "session-x",
  items: [
    {
      callId: "c1", tool: "generate_image", kind: "image", prompt: "Rose shop poster, warm morning light, hand-painted lettering", model: "FLUX.2 Klein 4B", modelId: "klein",
      size: "1024x1024", steps: 4, estimatedSeconds: 6, peakGb: 16, tier: "standard",
      tierOptions: { draft: { size: "512x512", estimatedSeconds: 2 }, standard: { size: "1024x1024", estimatedSeconds: 6 }, final: { size: "1536x1536", estimatedSeconds: 19 } },
      modelOptions: [{ id: "klein", name: "FLUX.2 Klein 4B" }, { id: "schnell", name: "FLUX.1-schnell" }],
      qualityNotes: ["Final was brought down to 1536x1536: the largest size measured on this GPU."],
    },
    {
      callId: "c2", tool: "generate_video", kind: "video", prompt: "Slow push-in on the shop window at dawn", model: "LTX-2.5 (4-bit)", size: "768x512", frames: 121, fps: 24, steps: 8,
      estimatedSeconds: 55, peakGb: 20, tier: "standard",
      tierOptions: { draft: { size: "384x256", estimatedSeconds: 12 }, standard: { size: "768x512", estimatedSeconds: 55 }, final: { size: "1152x768", estimatedSeconds: 179 } },
    },
    { callId: "c3", tool: "speak", kind: "speech", prompt: "Welcome to Rosa's, open from Monday.", model: "Kokoro", estimatedSeconds: 3 },
  ],
  memory: { chatModel: "Nemotron 3.5 Lightning", freeGb: 4.1, totalGb: 24, level: "risky", advice: "The clip needs about 20 GB and 4 GB is free: unload the chat model first, or it will run from system memory and take far longer." },
  status: "pending",
};
const PNG_1X1 = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64");

type Workspace = { runtime: Record<string, unknown>; chatSessions: Record<string, unknown>[] };

async function mockWorkspace(page: Page, sessions: Record<string, unknown>[] = []) {
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

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
      const body = new ReadableStream<Uint8Array>({ start: (c) => { controller = c; } });
      w.__studioPush = (event: unknown) => controller?.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}
const push = (page: Page, event: unknown) =>
  page.evaluate((e) => (window as unknown as { __studioPush: (x: unknown) => void }).__studioPush(e), event);

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });
    test.afterAll(() => dumpTexts(`p2-${phone ? "m" : "d"}`));

    test("10 chat thread with a loaded model, long text", async ({ page }) => {
      await mockWorkspace(page, [SESSION]);
      await goTo(page, "Chat", "Chat");
      await page.locator(".session-button").first().click();
      await shoot(page, "10-chat-thread", phone);
      await page.locator(".composer textarea").fill("/");
      await shoot(page, "10-chat-slash-menu", phone, false);
      await page.locator(".composer textarea").fill("");
    });

    for (const panel of ["prompts", "voice", "Samplers", "Temp", "KV:"]) {
      test(`11 composer panel ${panel}`, async ({ page }) => {
        await mockWorkspace(page, [SESSION]);
        await goTo(page, "Chat", "Chat");
        await page.locator(".session-button").first().click();
        if (panel === "prompts") {
          await page.locator(".composer .chat-prompts__trigger").click();
          await shoot(page, "11-chat-prompts-panel", phone, false);
          await page.getByRole("dialog", { name: "Prompts" }).locator(".chat-prompts__row").first().click();
          await shoot(page, "11-chat-prompts-detail", phone, false);
          await page.getByRole("dialog", { name: "Prompts" }).getByRole("button", { name: "Edit" }).click();
          await shoot(page, "11-chat-prompts-edit", phone, false);
        } else if (panel === "voice") {
          await page.locator(".voice-popover__button").click();
          await shoot(page, "11-chat-voice-popover", phone, false);
        } else {
          await page.locator(".composer button", { hasText: panel }).first().click();
          await shoot(page, `11-chat-${panel.toLowerCase().replace(":", "")}-popover`, phone, false);
        }
      });
    }

    test("12 select model dialog", async ({ page }) => {
      await goTo(page, "Chat", "Chat");
      await page.getByRole("button", { name: "Select Model" }).click();
      await shoot(page, "12-select-model-dialog-empty", phone, false);
    });

    test("13 studio: keep going panel, plan card, job cards", async ({ page }) => {
      await scriptChatStream(page);
      await mockWorkspace(page);
      await page.route("**/api/chat/studio/plans/*/decision", (route) => route.fulfill({ json: { planId: PLAN.planId } }));
      await page.route("**/api/chat/studio/price", (route) => route.fulfill({ json: { estimatedSeconds: 31, size: "1280x1280", steps: 8 } }));
      await page.route(/\/api\/chat\/studio\/jobs\/job-\d+$/, (route) => {
        const id = new URL(route.request().url()).pathname.split("/").pop();
        const job = id === "job-1"
          ? { jobId: id, kind: "image", label: "FLUX.2 Klein 4B", state: "done", output: { path: `/mock/${id}.png`, width: 1024, height: 1024 } }
          : id === "job-2"
            ? { jobId: id, kind: "video", label: "LTX-2.5 (4-bit)", state: "running", progress: 0.37, step: 3, totalSteps: 8, etaSeconds: 34 }
            : { jobId: id, kind: "speech", label: "Kokoro", state: "failed", error: "The voice runtime is not installed. Install it under Voice > Models." };
        return route.fulfill({ json: { job } });
      });
      await page.route(/\/api\/chat\/studio\/jobs\/job-\d+\/media/, (route) => route.fulfill({ body: PNG_1X1, contentType: "image/png" }));
      await goTo(page, "Chat", "Chat");
      await page.getByRole("switch", { name: "Studio" }).click();
      await shoot(page, "13-chat-studio-on", phone, false);
      const keep = page.locator(".composer button", { hasText: "Keep going" }).first();
      if (await keep.isVisible()) {
        await keep.click();
        await shoot(page, "13-chat-keep-going-panel", phone, false);
        await page.keyboard.press("Escape");
      }
      await page.locator(".composer textarea").fill("Make a poster, a short clip and a voice line for a flower shop");
      await page.getByRole("button", { name: "Send", exact: true }).click();
      await page.waitForFunction(() => Boolean((window as unknown as { __studioPush?: unknown }).__studioPush));
      await push(page, { phase: "generating", ttftSeconds: 1.2 });
      await push(page, { studioPlan: PLAN });
      const card = page.getByRole("region", { name: "Plan to approve" });
      await expect(card).toBeVisible();
      await shoot(page, "13-chat-plan-card", phone);
      await card.locator("summary", { hasText: "Expert settings" }).first().click();
      await card.locator(".studio-plan-card__expert input").first().fill("1280");
      await page.waitForTimeout(800);
      await shoot(page, "13-chat-plan-card-expert", phone);
      await card.getByRole("button", { name: "Approve all" }).click();
      const resolved = { ...PLAN, status: "decided", decisions: { c1: "approve", c2: "approve", c3: "approve" }, alwaysAllow: false };
      await push(page, { studioPlanResolved: resolved });
      const results = PLAN.items.map((item, i) => ({
        id: item.callId, name: item.tool, arguments: { prompt: item.prompt }, elapsed: 0.1, renderAs: "studio",
        result: JSON.stringify({ jobId: `job-${i + 1}`, state: "queued" }), data: { tool: item.tool, jobId: `job-${i + 1}`, label: item.model },
      }));
      for (const r of results) {
        await push(page, { toolCallStart: { id: r.id, name: r.name } });
        await push(page, { toolCallResult: r });
      }
      await push(page, { toolCallStart: { id: "c4", name: "web_search", arguments: { query: "flower shop opening offers" } } });
      await push(page, { toolCallResult: { id: "c4", name: "web_search", arguments: { query: "flower shop opening offers" }, elapsed: 1.4, result: "1. Ten opening offers for florists...\n2. How to price a bouquet..." } });
      await page.waitForTimeout(1500);
      await shoot(page, "13-chat-job-cards", phone);
    });

    test("14 chat models: rows and states", async ({ page }) => {
      await page.route("**/api/models/download/status", (route) => route.fulfill({
        json: {
          downloads: [
            { repo: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", state: "downloading", progress: 0.37, downloadedGb: 7.5, totalGb: 20.3, error: null },
            { repo: "prism-ml/Ternary-Bonsai-2-27B-gguf", state: "failed", progress: 0.1, downloadedGb: 0.8, totalGb: 8.5, error: "401 Client Error: this repository is gated. Add a Hugging Face token in Settings > Providers and accept the licence on the model page." },
            { repo: "ggml-org/Qwen3.6-35B-A3B-GGUF", state: "cancelled", progress: 0.55, downloadedGb: 11.3, totalGb: 20.5, error: null },
          ],
        },
      }));
      await goTo(page, "Chat", "Models");
      await page.waitForTimeout(1500);
      await shoot(page, "14-models-top", phone, false);
      await page.getByPlaceholder(/Search name/).fill("Nemotron 3.5 Lightning");
      await page.waitForTimeout(1200);
      await page.locator(".media-model-row, .models-card").filter({ hasText: "24 GB card" }).first().click();
      await shoot(page, "14-models-nemotron-search-expanded", phone);
      await page.getByPlaceholder(/Search name/).fill("Qwen3.6 35B");
      await page.waitForTimeout(1200);
      await shoot(page, "14-models-qwen36", phone);
      await page.getByPlaceholder(/Search name/).fill("Bonsai");
      await page.waitForTimeout(1200);
      await shoot(page, "14-models-bonsai-failed", phone);
      await page.getByPlaceholder(/Search name/).fill("Gemma 4");
      await page.waitForTimeout(1200);
      await shoot(page, "14-models-gemma", phone);
      await page.getByPlaceholder(/Search name/).fill("zzzz-nothing-matches");
      await page.waitForTimeout(2500);
      await shoot(page, "14-models-no-results", phone, false);
    });
  });
}
