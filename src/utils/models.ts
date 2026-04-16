import type { ModelFamily, ModelVariant } from "../types";
import { CAPABILITY_META } from "../constants/capabilities";

const DISCOVER_SEARCH_PUNCT_RE = /[^a-z0-9]+/g;
const DISCOVER_SEARCH_ALPHA_NUM_RE = /([a-z])(\d)|(\d)([a-z])/g;

export function flattenVariants(families: ModelFamily[]): ModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function normalizeDiscoverSearchText(value: string): string {
  const lowered = String(value ?? "").trim().toLowerCase();
  if (!lowered) return "";
  const normalized = lowered.replace(
    DISCOVER_SEARCH_ALPHA_NUM_RE,
    (_match, leftAlpha?: string, leftDigit?: string, rightDigit?: string, rightAlpha?: string) =>
      `${leftAlpha ?? rightDigit} ${leftDigit ?? rightAlpha}`,
  );
  return normalized.replace(DISCOVER_SEARCH_PUNCT_RE, " ").trim().replace(/\s+/g, " ");
}

export function discoverSearchTokens(query: string): string[] {
  const normalized = normalizeDiscoverSearchText(query);
  return normalized ? normalized.split(" ") : [];
}

export function modelFamilyDiscoverHaystack(family: ModelFamily): string {
  const fragments: string[] = [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.description,
    ...family.capabilities,
    ...family.readme,
  ];
  for (const variant of family.variants) {
    fragments.push(
      variant.name,
      variant.repo,
      variant.format,
      variant.quantization,
      variant.note,
      variant.contextWindow,
      ...variant.capabilities,
    );
  }
  return normalizeDiscoverSearchText(fragments.filter(Boolean).join(" "));
}

export function modelFamilyMatchesDiscoverQuery(family: ModelFamily, query: string): boolean {
  const tokens = discoverSearchTokens(query);
  if (!tokens.length) return true;
  const haystack = modelFamilyDiscoverHaystack(family);
  return tokens.every((token) => haystack.includes(token));
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
