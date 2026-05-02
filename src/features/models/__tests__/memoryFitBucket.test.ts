import { describe, expect, it } from "vitest";
import type { ModelVariant } from "../../../types";
import { memoryFitBucket } from "../OnlineModelsTab";

function makeVariant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return {
    id: "test/model",
    familyId: "fam",
    name: "Test",
    repo: "test/model",
    link: "https://huggingface.co/test/model",
    paramsB: 7,
    sizeGb: 4,
    format: "GGUF",
    quantization: "Q4_K_M",
    capabilities: [],
    note: "",
    contextWindow: "8K",
    estimatedMemoryGb: 5,
    estimatedCompressedMemoryGb: 3,
    availableLocally: false,
    launchMode: "direct",
    backend: "llama.cpp",
    ...overrides,
  };
}

describe("memoryFitBucket", () => {
  it("returns unknown when availableMemoryGb is null", () => {
    expect(memoryFitBucket(makeVariant(), null)).toEqual({ kind: "unknown", label: "" });
  });

  it("returns unknown when availableMemoryGb is zero", () => {
    expect(memoryFitBucket(makeVariant(), 0)).toEqual({ kind: "unknown", label: "" });
  });

  it("returns unknown when neither size nor estimate is known", () => {
    expect(
      memoryFitBucket(
        makeVariant({ sizeGb: 0, estimatedMemoryGb: null }),
        16,
      ),
    ).toEqual({ kind: "unknown", label: "" });
  });

  it("returns comfortable when estimate is well under available", () => {
    // 5 GB estimate vs 16 GB available → estimate is 31% → comfortable
    expect(memoryFitBucket(makeVariant({ estimatedMemoryGb: 5 }), 16)).toEqual({
      kind: "comfortable",
      label: "Fits",
    });
  });

  it("returns tight when estimate is close to available", () => {
    // 14 GB estimate vs 16 GB available → 87% → tight
    expect(memoryFitBucket(makeVariant({ estimatedMemoryGb: 14 }), 16)).toEqual({
      kind: "tight",
      label: "Tight",
    });
  });

  it("returns over when estimate exceeds available", () => {
    // 20 GB estimate vs 16 GB available → over
    expect(memoryFitBucket(makeVariant({ estimatedMemoryGb: 20 }), 16)).toEqual({
      kind: "over",
      label: "Too big",
    });
  });

  it("falls back to sizeGb when estimatedMemoryGb is missing", () => {
    expect(
      memoryFitBucket(makeVariant({ estimatedMemoryGb: null, sizeGb: 4 }), 16),
    ).toEqual({ kind: "comfortable", label: "Fits" });
  });
});
