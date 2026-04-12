import type { ModelFamily, ModelVariant } from "../types";
import { CAPABILITY_META } from "../constants/capabilities";

export function flattenVariants(families: ModelFamily[]): ModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function normalizeCapability(capability: string): string {
  return capability.trim().toLowerCase().replace(/\s+/g, "-");
}

export function capabilityMeta(capability: string) {
  const normalized = normalizeCapability(capability);
  return (
    CAPABILITY_META[normalized] ?? {
      shortLabel: normalized.slice(0, 4).toUpperCase(),
      title: capability,
    }
  );
}

export function defaultVariantForFamily(family: ModelFamily | null | undefined): ModelVariant | null {
  if (!family) {
    return null;
  }
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

export function findVariantById(families: ModelFamily[], variantId: string | null | undefined): ModelVariant | null {
  if (!variantId) {
    return null;
  }
  for (const family of families) {
    const variant = family.variants.find((item) => item.id === variantId);
    if (variant) {
      return variant;
    }
  }
  return null;
}

export function firstDirectVariant(families: ModelFamily[]): ModelVariant | null {
  const directVariants = flattenVariants(families)
    .filter((variant) => variant.launchMode === "direct")
    .sort((left, right) => left.paramsB - right.paramsB || left.sizeGb - right.sizeGb);
  return directVariants[0] ?? flattenVariants(families)[0] ?? null;
}

export function findVariantForReference(
  families: ModelFamily[],
  modelRef: string | null | undefined,
  modelName?: string | null,
): ModelVariant | null {
  if (!modelRef && !modelName) {
    return null;
  }
  const loweredRef = modelRef?.toLowerCase();
  const loweredName = modelName?.toLowerCase();
  for (const variant of flattenVariants(families)) {
    if (
      loweredRef &&
      [variant.id, variant.repo, variant.name, variant.link].some((candidate) => candidate.toLowerCase() === loweredRef)
    ) {
      return variant;
    }
    if (loweredName && variant.name.toLowerCase() === loweredName) {
      return variant;
    }
  }
  return null;
}
