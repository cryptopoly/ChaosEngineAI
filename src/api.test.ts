import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: () => false,
}));

import { convertModel, generateChat, getWorkspace, loadModel } from "./api";
import { mockWorkspace } from "./mockData";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("desktop api helpers", () => {
  it("falls back to mock workspace when the sidecar is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    const result = await getWorkspace();

    expect(result.runtime.engine).toBe(mockWorkspace.runtime.engine);
    expect(result.featuredModels.length).toBeGreaterThan(0);
  });

  it("posts model load payloads to the sidecar", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ runtime: mockWorkspace.runtime }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const runtime = await loadModel({ modelRef: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF" });

    expect(runtime.loadedModel?.name).toBe("Nemotron 3 Nano 4B GGUF");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8876/api/models/load",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts chat generation payloads to the sidecar", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session: mockWorkspace.chatSessions[0],
        assistant: mockWorkspace.chatSessions[0].messages[1],
        runtime: mockWorkspace.runtime,
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await generateChat({ prompt: "Hello there" });

    expect(result.assistant.role).toBe("assistant");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8876/api/chat/generate",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts conversion payloads to the sidecar", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        conversion: {
          sourceLabel: "model.gguf",
          hfRepo: "google/gemma-4-E4B-it",
          outputPath: "/Users/dan/Models/gemma-4-e4b-it-mlx",
          quantize: true,
          qBits: 4,
          dtype: "float16",
          log: "converted",
        },
        library: mockWorkspace.library,
        runtime: mockWorkspace.runtime,
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await convertModel({ path: "/tmp/model.gguf", hfRepo: "google/gemma-4-E4B-it" });

    expect(result.conversion.outputPath).toContain("gemma-4-e4b-it-mlx");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8876/api/models/convert",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
