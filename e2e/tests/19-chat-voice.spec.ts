import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { goTo, SHOTS_DIR } from "./helpers";
import { ADA, mockVoiceApi, silentWavBase64 } from "./voiceMocks";

// D-67 W6 / D-72: any voice in Chat read-aloud and calls. No model and no speech runs: the voice API is mocked
// (voiceMocks.ts), the chat reply is a scripted stream (as in spec 18) and /api/voice/speak/stream is answered here.
// Covers: Use in Chat and Use for calls on a natural voice with its measured cost, a cloned voice refused for calls,
// the read-aloud request body carrying the preset voice, the fallback note shown in the speech bar, and the Call info
// naming the call voice and pointing at Voices.

const BOTH_CUSTOM = ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b"];

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__payload = JSON.parse(String(init?.body ?? "{}"));
      let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
      const body = new ReadableStream<Uint8Array>({ start: (c) => { controller = c; } });
      w.__push = (event: unknown) => controller?.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      w.__close = () => controller?.close();
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function push(page: Page, event: unknown) {
  await page.evaluate((e) => (window as unknown as { __push: (x: unknown) => void }).__push(e), event);
}

async function mockLoadedModel(page: Page): Promise<() => Record<string, unknown>> {
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
        contextTokens: 8192, loadedAt: "2026-11-09 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: false, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
  return () => runtime;
}

test("a natural voice can be the chat voice and the call voice, with its measured cost; a cloned voice is refused for calls", async ({ page }) => {
  const seen = await mockVoiceApi(page, { clones: [ADA], installed: BOTH_CUSTOM });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  const list = page.locator(".voice-list");
  const ryan = list.locator("li").filter({ has: page.getByText("Ryan", { exact: true }) });
  const ada = list.locator("li").filter({ has: page.getByText("Ada", { exact: true }) });
  await expect(ryan.getByText("first audio 1.3 s (4.1 s cold), 3.51 GB")).toBeVisible();
  await ryan.getByRole("button", { name: "Use in Chat" }).click();
  await expect.poll(() => seen.ttsVoices).toEqual(["qwen3:Ryan"]);
  await expect(ryan.getByText("Chat voice", { exact: true })).toBeVisible();
  await expect(ada.getByRole("button", { name: "Use for calls" })).toBeDisabled();
  await expect(ada.getByRole("button", { name: "Use in Chat" })).toBeEnabled();
  const heart = list.locator("li", { hasText: "Heart" }).first();
  await heart.getByRole("button", { name: "Use for calls" }).click();
  await expect.poll(() => seen.callVoices).toEqual(["af_heart"]);
  await expect(heart.getByText("Call voice", { exact: true })).toBeVisible();

  // The call voice select in Chat and calls offers natural and quick voices with their time, never a cloned one.
  const panel = page.locator(".voice-chat-calls");
  const callSelect = panel.getByLabel("Call voice");
  await expect(callSelect).toHaveValue("af_heart");
  await expect(callSelect.locator("option", { hasText: "Ryan, 1.3 s" })).toHaveCount(1);
  await expect(callSelect.locator('option[value="clone:voice-1"]')).toHaveCount(0);
  await expect(panel.getByLabel("Chat voice (read-aloud)").locator('option[value="clone:voice-1"]')).toHaveCount(1);
  mkdirSync(SHOTS_DIR, { recursive: true });
  await panel.screenshot({ path: path.join(SHOTS_DIR, "voice-chat-calls.png") });
});

test("read-aloud sends the preset chat voice and shows the fallback note; Call info names the call voice", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM, ttsVoice: "qwen3:Ryan", callVoice: "", ready: true });
  const bodies: Array<Record<string, unknown>> = [];
  await page.route("**/api/voice/speak/stream", (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    const lines = [
      { note: "Ryan could not speak here (Qwen3-TTS 1.7B CustomVoice is not downloaded), so the Kokoro voice reads instead.", voice: "af_heart" },
      { index: 0, total: 1, text: "Here is the answer.", audio: silentWavBase64(2), durationS: 2, voice: "af_heart" },
      { done: true },
    ];
    return route.fulfill({ status: 200, contentType: "application/x-ndjson", body: lines.map((l) => JSON.stringify(l)).join("\n") + "\n" });
  });
  await scriptChatStream(page);
  const runtime = await mockLoadedModel(page);
  await goTo(page, "Chat", "Chat");

  // Call info: the voice by name, and where to change it.
  await page.locator(".info-tooltip").filter({ has: page.locator(".info-tooltip-icon") }).last().hover();
  await expect(page.getByRole("tooltip")).toContainText("Voice: Ryan. Change it in Voice Studio > Voices.");

  await page.locator(".composer textarea").fill("Tell me something");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __push?: unknown }).__push));
  const payload = await page.evaluate(() => (window as unknown as { __payload: Record<string, unknown> }).__payload);
  const assistant = { role: "assistant", text: "Here is the answer.", createdAt: Date.now() };
  await push(page, {
    done: true,
    session: {
      id: String(payload.sessionId), title: "Voice", updatedAt: "2026-11-09 10:01:00", model: "Mock Chat 8B", cacheLabel: "Native f16",
      messages: [{ role: "user", text: "Tell me something", createdAt: Date.now() }, assistant],
    },
    assistant,
    runtime: runtime(),
  });
  await page.evaluate(() => (window as unknown as { __close: () => void }).__close());
  await expect(page.getByText("Here is the answer.")).toBeVisible();
  await page.locator(".message-bubble.assistant").last().hover();
  await page.getByRole("button", { name: "Read aloud" }).last().click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ text: "Here is the answer.", voice: "qwen3:Ryan" });
  expect(bodies[0].use).toBeUndefined(); // read-aloud is the default use
  await expect(page.locator(".speech-bar__note")).toContainText("the Kokoro voice reads instead");
});
