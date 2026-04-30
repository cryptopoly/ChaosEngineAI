import type { DiscoverSort } from "../types/image";

// Minimum shape needed to sort a Discover variant — lets the same
// comparator serve ImageModelVariant and VideoModelVariant without
// either of them having to conform to a common interface at the type
// level. Every field is optional; missing fields sink to the bottom
// of the sorted list.
interface DiscoverSortable {
  name?: string | null;
  provider?: string | null;
  taskSupport?: string[] | null;
  releaseDate?: string | null;
  createdAt?: string | null;
  lastModified?: string | null;
  likes?: number | null;
  downloads?: number | null;
  sizeGb?: number | null;
  repoSizeGb?: number | null;
  coreWeightsGb?: number | null;
  onDiskGb?: number | null;
  runtimeFootprintGb?: number | null;
  runtimeFootprintMpsGb?: number | null;
  runtimeFootprintCudaGb?: number | null;
  runtimeFootprintCpuGb?: number | null;
}

function releaseSortKey(variant: DiscoverSortable): string {
  // Curated catalogs set ``releaseDate`` (YYYY-MM); HF live metadata
  // sets ``createdAt`` (ISO datetime). Both sort lexicographically in
  // descending order, so we can use string comparison directly. Fall
  // back to ``lastModified`` so variants with no release date still
  // get an order instead of bunching at the bottom.
  return (
    (variant.releaseDate ?? "")
    || (variant.createdAt ?? "")
    || (variant.lastModified ?? "")
  );
}

function firstPositiveNumber(values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  return null;
}

function sizeSortKey(variant: DiscoverSortable): number | null {
  return firstPositiveNumber([
    variant.onDiskGb,
    variant.coreWeightsGb,
    variant.repoSizeGb,
    variant.sizeGb,
  ]);
}

function ramSortKey(variant: DiscoverSortable): number | null {
  const runtimeValues = [
    variant.runtimeFootprintGb,
    variant.runtimeFootprintMpsGb,
    variant.runtimeFootprintCudaGb,
    variant.runtimeFootprintCpuGb,
  ].filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);
  const runtimeMax = runtimeValues.length > 0 ? Math.max(...runtimeValues) : null;
  return firstPositiveNumber([
    runtimeMax,
    variant.runtimeFootprintGb,
    variant.coreWeightsGb,
    variant.sizeGb,
    variant.onDiskGb,
    variant.repoSizeGb,
  ]);
}

function compareNullableNumberDesc(left: number | null, right: number | null): number {
  const leftKnown = typeof left === "number" && Number.isFinite(left);
  const rightKnown = typeof right === "number" && Number.isFinite(right);
  if (leftKnown && rightKnown) return (right as number) - (left as number);
  if (leftKnown) return -1;
  if (rightKnown) return 1;
  return 0;
}

function taskSortKey(variant: DiscoverSortable): string {
  return (variant.taskSupport ?? []).join(" ");
}

export function compareDiscoverVariants(
  sort: DiscoverSort,
  a: DiscoverSortable,
  b: DiscoverSortable,
): number {
  if (sort === "name") {
    return (a.name ?? "").localeCompare(b.name ?? "");
  }
  if (sort === "provider") {
    const diff = (a.provider ?? "").localeCompare(b.provider ?? "");
    if (diff !== 0) return diff;
    return (a.name ?? "").localeCompare(b.name ?? "");
  }
  if (sort === "tasks") {
    const diff = taskSortKey(a).localeCompare(taskSortKey(b));
    if (diff !== 0) return diff;
    return (a.name ?? "").localeCompare(b.name ?? "");
  }
  if (sort === "size") {
    const diff = compareNullableNumberDesc(sizeSortKey(a), sizeSortKey(b));
    if (diff !== 0) return diff;
    return releaseSortKey(b).localeCompare(releaseSortKey(a));
  }
  if (sort === "ram") {
    const diff = compareNullableNumberDesc(ramSortKey(a), ramSortKey(b));
    if (diff !== 0) return diff;
    return releaseSortKey(b).localeCompare(releaseSortKey(a));
  }
  if (sort === "likes") {
    const diff = (b.likes ?? -1) - (a.likes ?? -1);
    if (diff !== 0) return diff;
    // Stable tiebreak by recency so equal-likes variants stay in a
    // predictable order rather than flipping on every catalog refresh.
    return releaseSortKey(b).localeCompare(releaseSortKey(a));
  }
  if (sort === "downloads") {
    const diff = (b.downloads ?? -1) - (a.downloads ?? -1);
    if (diff !== 0) return diff;
    return releaseSortKey(b).localeCompare(releaseSortKey(a));
  }
  // Default: most recently released first.
  const keyA = releaseSortKey(a);
  const keyB = releaseSortKey(b);
  if (keyA && keyB) return keyB.localeCompare(keyA);
  if (keyB) return 1;
  if (keyA) return -1;
  return 0;
}
