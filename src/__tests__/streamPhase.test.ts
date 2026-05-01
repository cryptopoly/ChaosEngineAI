import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import { generateChatStream } from "../api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/**
 * Build a fetch-like response whose body emits the given SSE chunks one at a
 * time. Each chunk is encoded as `data: <json>\n` so the api.ts parser sees
 * realistic line boundaries.
 */
function makeStreamResponse(events: object[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

/**
 * Build a fetch mock that routes auth/session requests to a benign token
 * payload and chat-stream requests to the configured SSE response. Without
 * this, the chat stream call is preceded by an auth fetch that would otherwise
 * consume the same mocked response and break the test.
 */
function makeFetchMock(streamEvents: object[]): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation((url: string) => {
    if (url.includes("/api/auth/session")) {
      return Promise.resolve(
        new Response(JSON.stringify({ apiToken: null }), { status: 200, headers: { "Content-Type": "application/json" } }),
      );
    }
    return Promise.resolve(makeStreamResponse(streamEvents));
  });
}

describe("generateChatStream phase events (Phase 2.0)", () => {
  it("invokes onPhase('prompt_eval') as soon as the backend emits it", async () => {
    const fetchMock = makeFetchMock(
      [
        { phase: "prompt_eval" },
        {
          done: true,
          session: { id: "s1", title: "x", updatedAt: "now", model: "m", cacheLabel: "f16", messages: [] },
          assistant: { role: "assistant", text: "" },
          runtime: {},
        },
      ],
    );
    vi.stubGlobal("fetch", fetchMock);

    const phaseCalls: Array<[string, number | undefined]> = [];
    await generateChatStream(
      { prompt: "hi" },
      {
        onToken: () => {},
        onPhase: (phase, ttft) => phaseCalls.push([phase, ttft]),
        onDone: () => {},
        onError: () => {},
      },
    );

    expect(phaseCalls).toEqual([["prompt_eval", undefined]]);
  });

  it("invokes onPhase('generating', ttftSeconds) on phase transition", async () => {
    const fetchMock = makeFetchMock(
      [
        { phase: "prompt_eval" },
        { phase: "generating", ttftSeconds: 0.42 },
        { token: "hi" },
        {
          done: true,
          session: { id: "s1", title: "x", updatedAt: "now", model: "m", cacheLabel: "f16", messages: [] },
          assistant: { role: "assistant", text: "hi" },
          runtime: {},
        },
      ],
    );
    vi.stubGlobal("fetch", fetchMock);

    const phaseCalls: Array<[string, number | undefined]> = [];
    await generateChatStream(
      { prompt: "hi" },
      {
        onToken: () => {},
        onPhase: (phase, ttft) => phaseCalls.push([phase, ttft]),
        onDone: () => {},
        onError: () => {},
      },
    );

    expect(phaseCalls).toEqual([
      ["prompt_eval", undefined],
      ["generating", 0.42],
    ]);
  });

  it("does not invoke onPhase when callback omitted", async () => {
    const fetchMock = makeFetchMock(
      [
        { phase: "prompt_eval" },
        {
          done: true,
          session: { id: "s1", title: "x", updatedAt: "now", model: "m", cacheLabel: "f16", messages: [] },
          assistant: { role: "assistant", text: "" },
          runtime: {},
        },
      ],
    );
    vi.stubGlobal("fetch", fetchMock);

    let errored = false;
    await generateChatStream(
      { prompt: "hi" },
      {
        onToken: () => {},
        onDone: () => {},
        onError: () => { errored = true; },
      },
    );

    expect(errored).toBe(false);
  });

  it("ignores unknown phase values", async () => {
    const fetchMock = makeFetchMock(
      [
        { phase: "weird_phase" },
        {
          done: true,
          session: { id: "s1", title: "x", updatedAt: "now", model: "m", cacheLabel: "f16", messages: [] },
          assistant: { role: "assistant", text: "" },
          runtime: {},
        },
      ],
    );
    vi.stubGlobal("fetch", fetchMock);

    const phaseCalls: Array<[string, number | undefined]> = [];
    await generateChatStream(
      { prompt: "hi" },
      {
        onToken: () => {},
        onPhase: (phase, ttft) => phaseCalls.push([phase, ttft]),
        onDone: () => {},
        onError: () => {},
      },
    );

    expect(phaseCalls).toEqual([]);
  });
});
