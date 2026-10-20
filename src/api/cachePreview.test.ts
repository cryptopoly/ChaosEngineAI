import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import { getCachePreview } from "./index";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch() {
  const fetchMock = vi.fn(async (url: string) => {
    if (url.endsWith("/api/auth/session")) {
      return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    }
    return { ok: true, status: 200, json: async () => ({ archSource: "config" }) } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const base = {
  bits: 0, fp16Layers: 0, numLayers: 28, numHeads: 16, numKvHeads: 8,
  hiddenSize: 2048, contextTokens: 32768, paramsB: 1.7, strategy: "native",
};

function previewUrl(fetchMock: ReturnType<typeof stubFetch>): URL {
  const call = fetchMock.mock.calls.find(([url]) => String(url).includes("/api/cache/preview"));
  return new URL(String(call?.[0]));
}

describe("getCachePreview", () => {
  it("sends the model path and repo so the backend can read the real layout", async () => {
    const fetchMock = stubFetch();
    await getCachePreview({ ...base, modelPath: "/models/bonsai", modelRepo: "prism-ml/Bonsai-1.7B-gguf" });
    const url = previewUrl(fetchMock);
    expect(url.searchParams.get("model_path")).toBe("/models/bonsai");
    expect(url.searchParams.get("model_repo")).toBe("prism-ml/Bonsai-1.7B-gguf");
  });

  it("leaves them out when unknown", async () => {
    const fetchMock = stubFetch();
    await getCachePreview({ ...base, modelPath: null });
    const url = previewUrl(fetchMock);
    expect(url.searchParams.has("model_path")).toBe(false);
    expect(url.searchParams.has("model_repo")).toBe(false);
  });
});
