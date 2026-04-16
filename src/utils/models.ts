import type { ModelFamily, ModelVariant } from "../types";
import { CAPABILITY_META } from "../constants/capabilities";

const DISCOVER_SEARCH_PUNCT_RE = /[^a-z0-9]+/g;
const DISCOVER_SEARCH_ALPHA_NUM_RE = /([a-z])(\d)|(\d)([a-z])/g;
const HF_QUERY_URL_HOSTS = new Set(["huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"]);

export function flattenVariants(families: ModelFamily[]): ModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function extractHuggingFaceRepoIdFromQuery(value: string): string | null {
  const text = String(value ?? "").trim();
  if (!text) return null;
  try {
    const parsed = new URL(text);
    if (!HF_QUERY_URL_HOSTS.has(parsed.hostname.toLowerCase())) return null;
    const segments = parsed.pathname.split("/").filter(Boolean);
    if (segments.length < 2) return null;
    if (["models", "spaces", "datasets"].includes(segments[0]!.toLowerCase()) && segments.length >= 3) {
      return `${segments[1]}/${segments[2]}`;
    }
    return `${segments[0]}/${segments[1]}`;
  } catch {
    return null;
  }
}

export function normalizeDiscoverSearchText(value: string): string {
  const lowered = (extractHuggingFaceRepoIdFromQuery(value) ?? String(value ?? "").trim()).toLowerCase();
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
  const haystackTokens = new Set(discoverSearchTokens(modelFamilyDiscoverHaystack(family)));
  return tokens.every((token) => haystackTokens.has(token));
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
