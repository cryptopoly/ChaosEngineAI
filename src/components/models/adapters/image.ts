/**
 * Image catalog → Models page entries.
 *
 * ``collectImageVariants`` gives every curated variant its own row plus
 * the tracked "latest" rows the curated catalog does not already list
 * (R26: a tracked entry for a curated repo would show twice).
 * ``imageVariantToEntry`` maps one variant; ``buildImageEntries`` does the
 * whole catalog with the lookups the mapping needs.
 */

import type { ImageModelFamily, ImageModelVariant, SystemStats } from "../../../types";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  formatReleaseLabel,
  imageDiscoverFamilyMatchesQuery,
  imageDiscoverMemoryEstimate,
  imageDiscoverVariantMatchesQuery,
  imageOrVideoVariantPlatformGate,
  imagePrimarySizeLabel,
  imageSecondarySizeLabel,
  isVariantCompatibleWithHost,
} from "../../../utils";
import { acceleratorsForHost } from "../AcceleratorPills";
import { readBestFitVariantId, readVariantFit } from "../fit";
import type { ModelEntry } from "../modelEntry";

export type ImageModelEntry = ModelEntry<ImageModelVariant>;

export type HostLike = Pick<SystemStats, "platform" | "arch" | "gpuVramTotalGb"> | null | undefined;

export interface ImageEntryContext {
  family?: ImageModelFamily | null;
  host?: HostLike;
  /** Repo the image runtime has in memory. */
  loadedRepo?: string | null;
  /** Variant id → display name, for shared-download siblings. */
  nameById?: Map<string, string>;
}

/** Curated variants (each tagged with its family name) followed by the
 * tracked rows whose repo the curated catalog does not list. */
export function collectImageVariants(
  catalog: ImageModelFamily[],
  latest: ImageModelVariant[],
): ImageModelVariant[] {
  const curated = catalog.flatMap((family) =>
    family.variants.map((variant) => ({ ...variant, familyName: variant.familyName ?? family.name })),
  );
  const curatedRepos = new Set(curated.map((variant) => variant.repo.toLowerCase()));
  const seenTracked = new Set<string>();
  const tracked = latest.filter((variant) => {
    const repo = variant.repo.toLowerCase();
    if (curatedRepos.has(repo) || seenTracked.has(repo)) return false;
    seenTracked.add(repo);
    return true;
  });
  return [...curated, ...tracked];
}

function firstPositive(values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

export function imageSizeSortKey(variant: ImageModelVariant): number | null {
  return firstPositive([variant.onDiskGb, variant.coreWeightsGb, variant.repoSizeGb, variant.sizeGb]);
}

export function imageVariantToEntry(variant: ImageModelVariant, ctx: ImageEntryContext = {}): ImageModelEntry {
  const family = ctx.family ?? null;
  const memory = imageDiscoverMemoryEstimate(variant);
  const primarySize = imagePrimarySizeLabel(variant);
  const secondarySize = imageSecondarySizeLabel(variant);
  const siblings = variant.sharedRepoSiblings ?? 0;
  const siblingIds = variant.sharedRepoSiblingIds ?? [];
  return {
    id: variant.id,
    area: "image",
    name: variant.name,
    provider: variant.provider,
    familyId: variant.familyId,
    familyName: variant.familyName ?? family?.name ?? null,
    note: variant.note || null,
    repo: variant.repo,
    link: variant.link,
    downloadKey: variant.repo,
    deleteTarget: variant.repo,
    availableLocally: Boolean(variant.availableLocally),
    hasLocalData: Boolean(variant.hasLocalData),
    loaded: Boolean(ctx.loadedRepo && ctx.loadedRepo === variant.repo),
    localPath: variant.localPath ?? null,
    hostCompatible: isVariantCompatibleWithHost(imageOrVideoVariantPlatformGate(variant), ctx.host),
    trackedOnly: Boolean(variant.trackedOnly),
    gated: typeof variant.gated === "boolean" ? variant.gated : null,
    license: variant.license ?? null,
    runtime: variant.runtime,
    tasks: [...variant.taskSupport],
    styleTags: [...variant.styleTags],
    accelerators: acceleratorsForHost(variant, ctx.host),
    sizeGb: imageSizeSortKey(variant),
    sizeLabel: compactModelSizeLabel(primarySize),
    sizeTitle: [primarySize, secondarySize].filter(Boolean).join(" / ") || null,
    memory: memory ? { label: memory.label, title: memory.title, peakGb: memory.estimatedPeakGb } : null,
    releaseSortKey: variant.releaseDate ?? variant.createdAt ?? variant.lastModified ?? "",
    releaseLabel: compactReleaseLabel(formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt)),
    likes: variant.likes ?? null,
    downloads: variant.downloads ?? null,
    likesLabel: variant.likesLabel ?? null,
    downloadsLabel: variant.downloadsLabel ?? null,
    fit: readVariantFit(variant),
    bestFit: Boolean(family && readBestFitVariantId(family) === variant.id),
    shared:
      siblings > 0
        ? {
            count: siblings,
            key: variant.sharedRepoKey ?? variant.repo,
            siblingNames: siblingIds.map((id) => ctx.nameById?.get(id) ?? id),
          }
        : null,
    source: variant,
  };
}

export function buildImageEntries(
  catalog: ImageModelFamily[],
  latest: ImageModelVariant[],
  ctx: Omit<ImageEntryContext, "family" | "nameById"> = {},
): ImageModelEntry[] {
  const familyByVariantId = new Map<string, ImageModelFamily>();
  const nameById = new Map<string, string>();
  for (const family of catalog) {
    for (const variant of family.variants) {
      familyByVariantId.set(variant.id, family);
      nameById.set(variant.id, variant.name);
    }
  }
  for (const variant of latest) {
    if (!nameById.has(variant.id)) nameById.set(variant.id, variant.name);
  }
  return collectImageVariants(catalog, latest).map((variant) =>
    imageVariantToEntry(variant, { ...ctx, family: familyByVariantId.get(variant.id) ?? null, nameById }),
  );
}

/** Search used by the old Image Discover tab: a row matches when its own
 * text or its family's text contains the query. */
export function imageEntryMatchesQuery(
  entry: ImageModelEntry,
  query: string,
  familyById: Map<string, ImageModelFamily>,
): boolean {
  if (!query) return true;
  if (imageDiscoverVariantMatchesQuery(entry.source, query)) return true;
  const family = entry.familyId ? familyById.get(entry.familyId) : undefined;
  return Boolean(family && family.variants.some((variant) => variant.id === entry.id) && imageDiscoverFamilyMatchesQuery(family, query));
}
