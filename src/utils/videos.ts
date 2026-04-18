import type { VideoModelFamily, VideoModelVariant, VideoRuntimeStatus } from "../types";
import type { VideoDiscoverTaskFilter } from "../types/video";

export function flattenVideoVariants(families: VideoModelFamily[]): VideoModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function defaultVideoVariantForFamily(family?: VideoModelFamily | null): VideoModelVariant | null {
  if (!family) return null;
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

export function findVideoVariantById(families: VideoModelFamily[], variantId: string): VideoModelVariant | null {
  for (const family of families) {
    const match = family.variants.find((variant) => variant.id === variantId);
    if (match) return match;
  }
  return null;
}

export function findVideoVariantByRepo(
  families: VideoModelFamily[],
  repo: string | null | undefined,
): VideoModelVariant | null {
  if (!repo) return null;
  for (const family of families) {
    const match = family.variants.find((variant) => variant.repo === repo);
    if (match) return match;
  }
  return null;
}

export function videoVariantMatchesDiscoverFilters(
  variant: VideoModelVariant,
  taskFilter: VideoDiscoverTaskFilter,
): boolean {
  if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) {
    return false;
  }
  return true;
}

export function videoDiscoverVariantHaystack(variant: VideoModelVariant): string {
  return [
    variant.name,
    variant.familyName ?? "",
    variant.provider,
    variant.repo,
    variant.runtime,
    variant.recommendedResolution,
    variant.note,
    String(variant.sizeGb),
    String(variant.defaultDurationSeconds),
    variant.taskSupport.join(" "),
    variant.styleTags.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

export function videoDiscoverFamilyHaystack(family: VideoModelFamily): string {
  return [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.updatedLabel,
    family.badges.join(" "),
    ...family.variants.map((variant) => videoDiscoverVariantHaystack(variant)),
  ]
    .join(" ")
    .toLowerCase();
}

export function videoDiscoverVariantMatchesQuery(variant: VideoModelVariant, query: string): boolean {
  if (!query) return true;
  return videoDiscoverVariantHaystack(variant).includes(query);
}

export function videoDiscoverFamilyMatchesQuery(family: VideoModelFamily, query: string): boolean {
  if (!query) return true;
  return videoDiscoverFamilyHaystack(family).includes(query);
}

export function videoRuntimeErrorStatus(error: unknown): VideoRuntimeStatus {
  return {
    activeEngine: "unavailable",
    realGenerationAvailable: false,
    message: error instanceof Error ? error.message : "Video runtime unavailable.",
    missingDependencies: [],
  };
}
