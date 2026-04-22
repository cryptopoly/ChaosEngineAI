import type { DiscoverSort } from "../types/image";

// Minimum shape needed to sort a Discover variant — lets the same
// comparator serve ImageModelVariant and VideoModelVariant without
// either of them having to conform to a common interface at the type
// level. Every field is optional; missing fields sink to the bottom
// of the sorted list.
interface DiscoverSortable {
  releaseDate?: string | null;
  createdAt?: string | null;
  lastModified?: string | null;
  likes?: number | null;
  downloads?: number | null;
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

export function compareDiscoverVariants(
  sort: DiscoverSort,
  a: DiscoverSortable,
  b: DiscoverSortable,
): number {
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
