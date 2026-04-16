import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import { convertModel, generateChat, getWorkspace, loadModel, searchHubModels } from "./api";
import { mockWorkspace } from "./mockData";

const stubSession = {
  id: "session-1",
  title: "New chat",
  updatedAt: "2026-04-13 12:00:00",
  model: "Preview model",
  modelRef: null,
  modelSource: null,
  modelPath: null,
  modelBackend: null,
  cacheLabel: "Native f16",
  messages: [
    { role: "user" as const, text: "Hello there" },
    { role: "assistant" as const, text: "Hi.", metrics: null },
  ],
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("desktop api helpers", () => {
  it("throws when the sidecar is unavailable (no mock fallback)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    await expect(getWorkspace()).rejects.toThrow("offline");
  });

  it("posts model load payloads to the sidecar", async () => {
    const mockRuntime = {
      ...mockWorkspace.runtime,
      loadedModel: {
        ref: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        name: "Nemotron 3 Nano 4B GGUF",
        source: "catalog",
        backend: "llama.cpp",
        path: null,
        cacheBits: 0,
        fp16Layers: 0,
        contextTokens: 8192,
        runtimeTarget: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
      },
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ runtime: mockRuntime }),
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
        session: stubSession,
        assistant: stubSession.messages[1],
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
          outputPath: "/tmp/gemma-4-e4b-it-mlx",
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

  it("fetches live hub results from the dedicated hub search endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        results: [
          {
            id: "zai-org/GLM-4.7-Flash",
            repo: "zai-org/GLM-4.7-Flash",
            name: "GLM-4.7-Flash",
            provider: "zai-org",
            link: "https://huggingface.co/zai-org/GLM-4.7-Flash",
            format: "Transformers",
            tags: ["text-generation"],
            downloads: 1,
            likes: 1,
            downloadsLabel: "1 download",
            likesLabel: "1 like",
            availableLocally: false,
            launchMode: "convert",
            backend: "mlx",
          },
        ],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await searchHubModels("glm");

    expect(result).toHaveLength(1);
    expect(result[0]?.repo).toBe("zai-org/GLM-4.7-Flash");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8876/api/models/hub-search?q=glm",
      expect.any(Object),
    );
  });
});
