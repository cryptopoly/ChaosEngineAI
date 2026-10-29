import type { ImageModelFamily, ImageModelVariant } from "../../../types";

export function makeImageVariant(overrides: Partial<ImageModelVariant> = {}): ImageModelVariant {
  return {
    id: "black-forest-labs/FLUX.1-schnell",
    familyId: "flux-fast",
    name: "FLUX.1 Schnell",
    provider: "Black Forest Labs",
    repo: "black-forest-labs/FLUX.1-schnell",
    link: "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
    runtime: "diffusers",
    styleTags: ["fast", "general"],
    taskSupport: ["txt2img"],
    sizeGb: 23.8,
    recommendedResolution: "1024x1024",
    note: "Four-step distilled FLUX.",
    availableLocally: false,
    hasLocalData: false,
    estimatedGenerationSeconds: 8,
    releaseDate: "2024-08",
    ...overrides,
  };
}

export function makeImageFamily(overrides: Partial<ImageModelFamily> = {}, variants?: ImageModelVariant[]): ImageModelFamily {
  return {
    id: "flux-fast",
    name: "FLUX Fast",
    provider: "Black Forest Labs",
    headline: "Fast FLUX",
    summary: "Distilled FLUX models",
    updatedLabel: "Updated",
    badges: [],
    defaultVariantId: "black-forest-labs/FLUX.1-schnell",
    variants: variants ?? [makeImageVariant()],
    ...overrides,
  };
}
