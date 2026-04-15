import { describe, expect, it } from "vitest";

import { findCatalogVariantForLibraryItem } from "../library";
import type { LibraryItem, ModelFamily, ModelVariant } from "../../types";

function makeVariant(overrides: Partial<ModelVariant> & { id: string; name: string; repo: string }): ModelVariant {
  const { id, name, repo, ...rest } = overrides;
  return {
    id,
    familyId: "qwen",
    name,
    repo,
    link: `https://example.com/${id}`,
    paramsB: 0,
    sizeGb: 0,
    format: "GGUF",
    quantization: "Q4_K_M",
    capabilities: [],
    note: "",
    contextWindow: "128K",
    estimatedMemoryGb: null,
    estimatedCompressedMemoryGb: null,
    availableLocally: false,
    launchMode: "direct",
    backend: "llama.cpp",
    ...rest,
  };
}

function makeFamily(variants: ModelVariant[]): ModelFamily {
  return {
    id: "qwen-family",
    name: "Qwen",
    provider: "Qwen",
    headline: "",
    summary: "",
    description: "",
    updatedLabel: "",
    popularityLabel: "",
    likesLabel: "",
    badges: [],
    capabilities: [],
    defaultVariantId: variants[0]?.id ?? "",
    variants,
    readme: [],
  };
}

function makeItem(overrides: Partial<LibraryItem> & { name: string; path: string }): LibraryItem {
  const { name, path, ...rest } = overrides;
  return {
    name,
    path,
    format: "GGUF",
    quantization: "Q4_K_M",
    backend: "llama.cpp",
    modelType: "text",
    sizeGb: 20.6,
    lastModified: "2026-04-15T10:00:00Z",
    actions: [],
    ...rest,
  };
}

describe("findCatalogVariantForLibraryItem()", () => {
  it("prefers matching parameter scale over matching quantization alone", () => {
    const qwen9 = makeVariant({
      id: "lmstudio-community/Qwen3.5-9B-GGUF",
      name: "Qwen3.5 9B GGUF",
      repo: "lmstudio-community/Qwen3.5-9B-GGUF",
      paramsB: 9,
      sizeGb: 5.8,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const qwen35 = makeVariant({
      id: "Qwen/Qwen3.5-35B-A3B-FP8",
      name: "Qwen3.5 35B A3B",
      repo: "Qwen/Qwen3.5-35B-A3B-FP8",
      paramsB: 35,
      sizeGb: 22.8,
      format: "Transformers",
      quantization: "FP8",
      launchMode: "convert",
      backend: "auto",
    });

    const item = makeItem({
      name: "Qwen3.5-35B-A3B-GGUF",
      path: "/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
    });

    const matched = findCatalogVariantForLibraryItem([makeFamily([qwen9, qwen35])], item);
    expect(matched?.id).toBe(qwen35.id);
  });

  it("still matches the 9B GGUF variant for a real 9B GGUF file", () => {
    const qwen9 = makeVariant({
      id: "lmstudio-community/Qwen3.5-9B-GGUF",
      name: "Qwen3.5 9B GGUF",
      repo: "lmstudio-community/Qwen3.5-9B-GGUF",
      paramsB: 9,
      sizeGb: 5.8,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const qwen35 = makeVariant({
      id: "Qwen/Qwen3.5-35B-A3B-FP8",
      name: "Qwen3.5 35B A3B",
      repo: "Qwen/Qwen3.5-35B-A3B-FP8",
      paramsB: 35,
      sizeGb: 22.8,
      format: "Transformers",
      quantization: "FP8",
      launchMode: "convert",
      backend: "auto",
    });

    const item = makeItem({
      name: "Qwen3.5-9B-GGUF",
      path: "/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
      sizeGb: 5.8,
    });

    const matched = findCatalogVariantForLibraryItem([makeFamily([qwen9, qwen35])], item);
    expect(matched?.id).toBe(qwen9.id);
  });
});
