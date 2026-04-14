import type { ImageModelFamily, ImageModelVariant, ImageOutputArtifact } from "../types";
import type { ImageDiscoverTaskFilter, ImageDiscoverAccessFilter, ImageGalleryOrientationFilter } from "../types/image";

export function flattenImageVariants(families: ImageModelFamily[]): ImageModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function defaultImageVariantForFamily(family?: ImageModelFamily | null): ImageModelVariant | null {
  if (!family) return null;
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

export function findImageVariantById(families: ImageModelFamily[], variantId: string): ImageModelVariant | null {
  for (const family of families) {
    const match = family.variants.find((variant) => variant.id === variantId);
    if (match) return match;
  }
  return null;
}

export function findImageVariantByRepo(families: ImageModelFamily[], repo: string | null | undefined): ImageModelVariant | null {
  if (!repo) return null;
  for (const family of families) {
    const match = family.variants.find((variant) => variant.repo === repo);
    if (match) return match;
  }
  return null;
}

export function imageRuntimeKind(label?: string | null) {
  const lowered = (label ?? "").toLowerCase();
  if (lowered.includes("placeholder")) return "placeholder";
  if (lowered.includes("diffusers")) return "diffusers";
  return "other";
}

export function imageOrientation(width: number, height: number): Exclude<ImageGalleryOrientationFilter, "all"> {
  if (width === height) return "square";
  return width > height ? "landscape" : "portrait";
}

export function imageArtifactTimestamp(artifact: ImageOutputArtifact) {
  const timestamp = Date.parse(artifact.createdAt);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

export function imageVariantMatchesDiscoverFilters(
  variant: ImageModelVariant,
  taskFilter: ImageDiscoverTaskFilter,
  accessFilter: ImageDiscoverAccessFilter,
) {
  if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) {
    return false;
  }
  if (accessFilter === "open" && variant.gated === true) {
    return false;
  }
  if (accessFilter === "gated" && variant.gated !== true) {
    return false;
  }
  return true;
}

export function imageDiscoverVariantHaystack(variant: ImageModelVariant) {
  return [
    variant.name,
    variant.familyName ?? "",
    variant.provider,
    variant.repo,
    variant.runtime,
    variant.recommendedResolution,
    variant.note,
    variant.updatedLabel ?? "",
    variant.license ?? "",
    variant.pipelineTag ?? "",
    variant.downloadsLabel ?? "",
    variant.likesLabel ?? "",
    String(variant.sizeGb),
    String(variant.repoSizeGb ?? ""),
    String(variant.coreWeightsGb ?? ""),
    variant.gated ? "gated access" : "open access",
    variant.taskSupport.join(" "),
    variant.styleTags.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

export function imageDiscoverFamilyHaystack(family: ImageModelFamily) {
  return [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.updatedLabel,
    family.badges.join(" "),
    ...family.variants.map((variant) => imageDiscoverVariantHaystack(variant)),
  ]
    .join(" ")
    .toLowerCase();
}

export function imageDiscoverVariantMatchesQuery(variant: ImageModelVariant, query: string) {
  if (!query) return true;
  return imageDiscoverVariantHaystack(variant).includes(query);
}

export function imageDiscoverFamilyMatchesQuery(family: ImageModelFamily, query: string) {
  if (!query) return true;
  return imageDiscoverFamilyHaystack(family).includes(query);
}
