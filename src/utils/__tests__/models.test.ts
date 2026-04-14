import { describe, expect, it } from "vitest";

import {
  capabilityMeta,
  defaultVariantForFamily,
  findVariantById,
  firstDirectVariant,
  flattenVariants,
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
