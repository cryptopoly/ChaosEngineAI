/**
 * Filtering and sorting for a Models page, kept pure so the rules are
 * testable.
 *
 * Order of operations (``buildModelRows``):
 *   1. resolve each entry's status;
 *   2. drop rows the host cannot run, unless something of them is on disk
 *      or in flight (``isVisibleOnHost``);
 *   3. apply the search matcher and the area's filter groups;
 *   4. count the buckets (so the availability control shows how many rows
 *      each choice would give under the other filters);
 *   5. apply the availability filter;
 *   6. sort.
 */

import type { ModelAvailabilityFilter, ModelEntry } from "./modelEntry";
import {
  countBuckets,
  isVisibleOnHost,
  matchesAvailability,
  type ModelBucketCounts,
  type ResolvedModelStatus,
} from "./modelStatus";

export type SortDir = "asc" | "desc";

export interface ModelRow<E extends ModelEntry = ModelEntry> {
  entry: E;
  status: ResolvedModelStatus;
}

export interface ModelSortOption<E extends ModelEntry = ModelEntry> {
  id: string;
  /** Label in the sort select. */
  label: string;
  /** Text after "N models ·" in the summary line. */
  summary?: (dir: SortDir) => string;
  defaultDir: SortDir;
  /** Ascending comparison, for text and rank keys. */
  compare?: (left: ModelRow<E>, right: ModelRow<E>) => number;
  /** Numeric key (size, RAM, likes...). Rows without a number go last in
   * both directions, as on the old Discover tabs. */
  value?: (row: ModelRow<E>) => number | null | undefined;
}

/** A filter the area owns (task, access, capability, format...). When
 * ``predicate`` is given the shell applies it; otherwise the area has
 * already filtered ``entries`` and the group is only rendered. */
export interface ModelFilterGroup<E extends ModelEntry = ModelEntry> {
  id: string;
  label: string;
  kind: "select" | "chips";
  value: string;
  defaultValue: string;
  options: Array<{ value: string; label: string; count?: number; title?: string; color?: string }>;
  onChange: (value: string) => void;
  predicate?: (entry: E, value: string) => boolean;
  /** Hide the group when it has fewer than this many options. */
  hideBelow?: number;
}

export interface BuildModelRowsInput<E extends ModelEntry> {
  entries: E[];
  resolveStatus: (entry: E) => ResolvedModelStatus;
  query?: string;
  matches?: (entry: E, query: string) => boolean;
  groups?: ModelFilterGroup<E>[];
  availability: ModelAvailabilityFilter;
  sort?: ModelSortOption<E> | null;
  sortDir?: SortDir;
  /** Rows to keep above the others whatever the sort (chat favourites).
   * Order inside each band follows the sort. */
  pinned?: (entry: E) => boolean;
}

export interface BuildModelRowsResult<E extends ModelEntry> {
  rows: ModelRow<E>[];
  counts: ModelBucketCounts;
  /** Rows visible on this host before search, groups and availability. */
  hostVisibleCount: number;
}

function hasNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Tie-break: newest release first, then name. */
export function defaultTieBreak<E extends ModelEntry>(left: ModelRow<E>, right: ModelRow<E>): number {
  const dateDiff = right.entry.releaseSortKey.localeCompare(left.entry.releaseSortKey);
  if (dateDiff !== 0) return dateDiff;
  return left.entry.name.localeCompare(right.entry.name);
}

export function sortModelRows<E extends ModelEntry>(
  rows: ModelRow<E>[],
  sort: ModelSortOption<E> | null | undefined,
  dir: SortDir,
): ModelRow<E>[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((left, right) => {
    if (sort?.value) {
      const a = sort.value(left);
      const b = sort.value(right);
      const aKnown = hasNumber(a);
      const bKnown = hasNumber(b);
      if (aKnown && bKnown && a !== b) return sign * (a - b);
      if (aKnown !== bKnown) return aKnown ? -1 : 1;
    } else if (sort?.compare) {
      const diff = sort.compare(left, right);
      if (diff !== 0) return sign * diff;
    }
    return defaultTieBreak(left, right);
  });
}

export function buildModelRows<E extends ModelEntry>(input: BuildModelRowsInput<E>): BuildModelRowsResult<E> {
  const query = (input.query ?? "").trim().toLowerCase();
  const visible: ModelRow<E>[] = [];
  let hostVisibleCount = 0;
  for (const entry of input.entries) {
    const status = input.resolveStatus(entry);
    if (!isVisibleOnHost(entry, status)) continue;
    hostVisibleCount += 1;
    if (query && input.matches && !input.matches(entry, query)) continue;
    const groupsMatch = (input.groups ?? []).every(
      (group) => !group.predicate || group.value === group.defaultValue || group.predicate(entry, group.value),
    );
    if (!groupsMatch) continue;
    visible.push({ entry, status });
  }
  const counts = countBuckets(visible.map((row) => row.status));
  const filtered = visible.filter((row) => matchesAvailability(row.status.bucket, input.availability));
  const sort = input.sort ?? null;
  const dir = input.sortDir ?? sort?.defaultDir ?? "desc";
  const sorted = sortModelRows(filtered, sort, dir);
  const pinned = input.pinned;
  const rows = pinned
    ? [...sorted.filter((row) => pinned(row.entry)), ...sorted.filter((row) => !pinned(row.entry))]
    : sorted;
  return { rows, counts, hostVisibleCount };
}

export function hasActiveModelFilters<E extends ModelEntry>(input: {
  query?: string;
  groups?: ModelFilterGroup<E>[];
  availability: ModelAvailabilityFilter;
  sortId?: string;
  defaultSortId?: string;
  sortDir?: SortDir;
  defaultSortDir?: SortDir;
}): boolean {
  if ((input.query ?? "").trim().length > 0) return true;
  if (input.availability !== "all") return true;
  if ((input.groups ?? []).some((group) => group.value !== group.defaultValue)) return true;
  if (input.sortId !== undefined && input.defaultSortId !== undefined && input.sortId !== input.defaultSortId) return true;
  if (input.sortDir !== undefined && input.defaultSortDir !== undefined && input.sortDir !== input.defaultSortDir) return true;
  return false;
}
