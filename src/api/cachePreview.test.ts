import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import { checkModelMemory, getCachePreview } from "./index";

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
    await getCachePreview({
      ...base, modelPath: "/models/bonsai", modelRepo: "prism-ml/Bonsai-1.7B-gguf", weightsGb: 0.25,
    });
    const url = previewUrl(fetchMock);
    expect(url.searchParams.get("weights_gb")).toBe("0.25");
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

describe("checkModelMemory", () => {
  it("sends the load inputs and returns null when the check fails", async () => {
    const fetchMock = stubFetch();
    await checkModelMemory({
      weightsGb: 33, contextTokens: 32768, bits: 0, fp16Layers: 0, strategy: "native", paramsB: 78,
      numLayers: 80, numHeads: 64, numKvHeads: 8, hiddenSize: 8192, modelRepo: "velaia/Kolibri-1-MLX-3bit",
    });
    const call = fetchMock.mock.calls.find(([url]) => String(url).includes("/api/models/memory-check"));
    const url = new URL(String(call?.[0]));
    expect(url.searchParams.get("weights_gb")).toBe("33");
    expect(url.searchParams.get("model_repo")).toBe("velaia/Kolibri-1-MLX-3bit");

    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    expect(await checkModelMemory({
      weightsGb: 1, contextTokens: 1024, bits: 0, fp16Layers: 0, strategy: "native", paramsB: 1,
      numLayers: 1, numHeads: 1, numKvHeads: 1, hiddenSize: 256,
    })).toBeNull();
  });
});
