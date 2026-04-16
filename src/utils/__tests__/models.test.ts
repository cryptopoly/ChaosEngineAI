import { describe, expect, it } from "vitest";

import {
  discoverSearchTokens,
  extractHuggingFaceRepoIdFromQuery,
  capabilityMeta,
  defaultVariantForFamily,
  findVariantById,
  firstDirectVariant,
  flattenVariants,
  modelFamilyMatchesDiscoverQuery,
  normalizeDiscoverSearchText,
  normalizeCapability,
} from "../models";
import type { ModelFamily, ModelVariant } from "../../types";

function makeVariant(overrides: Partial<ModelVariant> & { id: string }): ModelVariant {
  return {
    name: overrides.id,
    repo: `org/${overrides.id}`,
    link: `https://hf.co/${overrides.id}`,
    sizeGb: 4,
    paramsB: 7,
    launchMode: "direct" as const,
    capabilities: [],
    quantizations: [],
    ...overrides,
  } as ModelVariant;
}

function makeFamily(overrides: Partial<ModelFamily> & { id: string }): ModelFamily {
  return {
    name: overrides.id,
    variants: [],
    defaultVariantId: "",
    ...overrides,
  } as ModelFamily;
}

describe("normalizeCapability()", () => {
  it("lowercases and replaces spaces with hyphens", () => {
    expect(normalizeCapability("Tool Use")).toBe("tool-use");
  });

  it("trims whitespace", () => {
    expect(normalizeCapability("  chat  ")).toBe("chat");
  });

  it("collapses multiple spaces", () => {
    expect(normalizeCapability("multi   lingual")).toBe("multi-lingual");
  });
});

describe("Discover search helpers", () => {
  it("normalizes punctuation and alpha-numeric joins", () => {
    expect(normalizeDiscoverSearchText(" Qwen3-Coder.Next 32B ")).toBe("qwen 3 coder next 32 b");
  });

  it("splits normalized tokens for token-based matching", () => {
    expect(discoverSearchTokens("qwen next 32b")).toEqual(["qwen", "next", "32", "b"]);
  });

  it("extracts repo ids from Hugging Face model URLs", () => {
    expect(extractHuggingFaceRepoIdFromQuery("https://huggingface.co/Qwen/Qwen3.6-35B-A3B")).toBe("Qwen/Qwen3.6-35B-A3B");
    expect(extractHuggingFaceRepoIdFromQuery("https://hf.co/Qwen/Qwen3.6-35B-A3B")).toBe("Qwen/Qwen3.6-35B-A3B");
  });

  it("normalizes Hugging Face URLs into search tokens", () => {
    expect(discoverSearchTokens("https://huggingface.co/Qwen/Qwen3.6-35B-A3B")).toEqual(["qwen", "qwen", "3", "6", "35", "b", "a", "3b"]);
  });

  it("matches families across name, variant, and repo text", () => {
    const family = makeFamily({
      id: "qwen3-coder",
      name: "Qwen3 Coder",
      provider: "Qwen",
      headline: "Code-specialised Qwen3 family",
      summary: "Purpose-built coding model.",
      description: "Strong for tool use.",
      capabilities: ["coding", "tool-use"],
      readme: ["Official Next repos replace older provisional 8B and 32B placeholders."],
      variants: [
        makeVariant({
          id: "Qwen/Qwen3-Coder-Next-FP8",
          name: "Qwen3 Coder Next FP8",
          repo: "Qwen/Qwen3-Coder-Next-FP8",
          format: "Transformers",
          quantization: "FP8",
          note: "Official FP8 repo.",
          contextWindow: "256K",
          capabilities: ["coding", "agents", "tool-use"],
        }),
      ],
    });

    expect(modelFamilyMatchesDiscoverQuery(family, "qwen coder")).toBe(true);
    expect(modelFamilyMatchesDiscoverQuery(family, "coder qwen")).toBe(true);
    expect(modelFamilyMatchesDiscoverQuery(family, "qwen next 32b")).toBe(true);
    expect(modelFamilyMatchesDiscoverQuery(family, "qwen next fp8")).toBe(true);
  });

  it("does not false-positive dotted version queries against nearby numeric families", () => {
    const family = makeFamily({
      id: "qwen3-5",
      name: "Qwen 3.5",
      provider: "Qwen",
      headline: "Hybrid reasoning family",
      summary: "Long context with strong agent support.",
      description: "Includes 35B A3B and 262K context variants.",
      capabilities: ["reasoning", "coding", "vision"],
      readme: ["Useful when you want modern reasoning and coding performance."],
      variants: [
        makeVariant({
          id: "Qwen/Qwen3.5-35B-A3B",
          name: "Qwen3.5 35B A3B",
          repo: "Qwen/Qwen3.5-35B-A3B",
          format: "Transformers",
          quantization: "FP8",
          note: "262K context variant.",
          contextWindow: "262K",
          capabilities: ["reasoning", "coding", "vision", "agents"],
        }),
      ],
    });

    expect(modelFamilyMatchesDiscoverQuery(family, "Qwen3.6")).toBe(false);
    expect(modelFamilyMatchesDiscoverQuery(family, "Qwen3.6-35B-A3B")).toBe(false);
  });
});

describe("capabilityMeta()", () => {
  it("returns known metadata for a recognized capability", () => {
    const meta = capabilityMeta("coding");
    expect(meta.shortLabel).toBe("Code");
    expect(meta.title).toBe("Coding support");
  });

  it("returns a fallback for unknown capabilities", () => {
    const meta = capabilityMeta("quantum-teleportation");
    expect(meta.shortLabel).toBe("QUAN");
    expect(meta.title).toBe("quantum-teleportation");
  });
});

describe("flattenVariants()", () => {
  it("flattens variants from multiple families", () => {
    const families = [
      makeFamily({ id: "f1", variants: [makeVariant({ id: "v1" }), makeVariant({ id: "v2" })] }),
      makeFamily({ id: "f2", variants: [makeVariant({ id: "v3" })] }),
    ];
    expect(flattenVariants(families)).toHaveLength(3);
  });

  it("returns empty for no families", () => {
    expect(flattenVariants([])).toEqual([]);
  });
});

describe("defaultVariantForFamily()", () => {
  it("returns the variant matching defaultVariantId", () => {
    const family = makeFamily({
      id: "f1",
      defaultVariantId: "v2",
      variants: [makeVariant({ id: "v1" }), makeVariant({ id: "v2" })],
    });
    expect(defaultVariantForFamily(family)?.id).toBe("v2");
  });

  it("falls back to first variant if defaultVariantId does not match", () => {
    const family = makeFamily({
      id: "f1",
      defaultVariantId: "missing",
      variants: [makeVariant({ id: "v1" })],
    });
    expect(defaultVariantForFamily(family)?.id).toBe("v1");
  });

  it("returns null for null/undefined family", () => {
    expect(defaultVariantForFamily(null)).toBeNull();
    expect(defaultVariantForFamily(undefined)).toBeNull();
  });
});

describe("findVariantById()", () => {
  const families = [
    makeFamily({ id: "f1", variants: [makeVariant({ id: "v1" })] }),
    makeFamily({ id: "f2", variants: [makeVariant({ id: "v2" }), makeVariant({ id: "v3" })] }),
  ];

  it("finds a variant across families", () => {
    expect(findVariantById(families, "v3")?.id).toBe("v3");
  });

  it("returns null for unknown id", () => {
    expect(findVariantById(families, "nope")).toBeNull();
  });

  it("returns null for null/undefined id", () => {
    expect(findVariantById(families, null)).toBeNull();
    expect(findVariantById(families, undefined)).toBeNull();
  });
});

describe("firstDirectVariant()", () => {
  it("returns the smallest direct variant sorted by paramsB then sizeGb", () => {
    const families = [
      makeFamily({
        id: "f1",
        variants: [
          makeVariant({ id: "big", paramsB: 70, sizeGb: 40, launchMode: "direct" }),
          makeVariant({ id: "small", paramsB: 3, sizeGb: 2, launchMode: "direct" }),
        ],
      }),
    ];
    expect(firstDirectVariant(families)?.id).toBe("small");
  });

  it("falls back to any variant when no direct variants exist", () => {
    const families = [
      makeFamily({
        id: "f1",
        variants: [makeVariant({ id: "api-only", launchMode: "api" as any })],
      }),
    ];
    expect(firstDirectVariant(families)?.id).toBe("api-only");
  });

  it("returns null for empty families", () => {
    expect(firstDirectVariant([])).toBeNull();
  });
});
