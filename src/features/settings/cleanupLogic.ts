/**
 * Settings > Clean up: the pure rules for what is ticked and what a delete would take.
 * Nothing is ticked by default; the only way an id reaches the backend is a tick on a listed item.
 */

import type { CleanupDeleteBody, CleanupDeleteResult, CleanupItems, CleanupResultRow } from "../../api/cleanup";

export type GroupKey = "projects" | "threads" | "outputs";
export const GROUPS: readonly GroupKey[] = ["projects", "threads", "outputs"];

export type Selection = Record<GroupKey, ReadonlySet<string>>;
export type GroupState = "none" | "some" | "all";

export function emptySelection(): Selection {
  return { projects: new Set(), threads: new Set(), outputs: new Set() };
}

export function toggleId(selection: Selection, group: GroupKey, id: string): Selection {
  const next = new Set(selection[group]);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return { ...selection, [group]: next };
}

/** Tick or clear every listed item of one group. */
export function setGroup(selection: Selection, items: CleanupItems, group: GroupKey, on: boolean): Selection {
  return { ...selection, [group]: on ? new Set(items[group].map((item) => item.id)) : new Set<string>() };
}

export function groupState(items: CleanupItems, selection: Selection, group: GroupKey): GroupState {
  const total = items[group].length;
  const ticked = items[group].filter((item) => selection[group].has(item.id)).length;
  if (ticked === 0) return "none";
  return ticked === total ? "all" : "some";
}

/** Drop ticks for items that are no longer listed (after a delete or a refresh). */
export function pruneSelection(selection: Selection, items: CleanupItems): Selection {
  const next = emptySelection() as { [K in GroupKey]: Set<string> };
  for (const group of GROUPS) {
    const listed = new Set(items[group].map((item) => item.id));
    for (const id of selection[group]) if (listed.has(id)) next[group].add(id);
  }
  return next;
}

export interface SelectionTotals {
  count: number;
  bytes: number;
  byGroup: Record<GroupKey, number>;
}

export function selectionTotals(items: CleanupItems, selection: Selection): SelectionTotals {
  const byGroup: Record<GroupKey, number> = { projects: 0, threads: 0, outputs: 0 };
  let bytes = 0;
  for (const group of GROUPS) {
    for (const item of items[group]) {
      if (!selection[group].has(item.id)) continue;
      byGroup[group] += 1;
      bytes += item.bytes;
    }
  }
  return { count: byGroup.projects + byGroup.threads + byGroup.outputs, bytes, byGroup };
}

/** The request body: ticked ids of listed items only. */
export function deleteBody(items: CleanupItems, selection: Selection): CleanupDeleteBody {
  const pick = (group: GroupKey) => items[group].map((item) => item.id).filter((id) => selection[group].has(id));
  return { projects: pick("projects"), threads: pick("threads"), outputs: pick("outputs") };
}

export interface ResultSummary {
  deleted: number;
  freedBytes: number;
  refused: CleanupResultRow[];
}

export function summarizeResult(result: CleanupDeleteResult): ResultSummary {
  return {
    deleted: result.results.filter((row) => row.ok).length,
    freedBytes: result.results.reduce((sum, row) => sum + (row.ok ? row.bytes : 0), 0),
    refused: result.results.filter((row) => !row.ok),
  };
}

/** Epoch seconds to a Date, or null for a missing or bad value. */
export function dateFromSeconds(seconds: number | null | undefined): Date | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0) return null;
  return new Date(seconds * 1000);
}
