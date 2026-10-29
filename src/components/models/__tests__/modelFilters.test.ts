import { describe, expect, it } from "vitest";
import type { DownloadStatus } from "../../../api";
import type { ModelEntry } from "../modelEntry";
import { buildModelRows, hasActiveModelFilters, sortModelRows, type ModelFilterGroup, type ModelSortOption } from "../modelFilters";
import { resolveModelStatus } from "../modelStatus";

function entry(overrides: Partial<ModelEntry> & { id: string }): ModelEntry {
  return {
    area: "image",
    name: overrides.id,
    provider: "Org",
    repo: `org/${overrides.id}`,
    downloadKey: `org/${overrides.id}`,
    deleteTarget: `org/${overrides.id}`,
    availableLocally: false,
    hasLocalData: false,
    hostCompatible: true,
    releaseSortKey: "",
    source: null,
    ...overrides,
  };
}

const downloads: Record<string, DownloadStatus> = {
  "org/paused": { repo: "org/paused", state: "cancelled", progress: 0.2, downloadedGb: 1, totalGb: 5, error: null },
};
const resolve = (item: ModelEntry) => resolveModelStatus(item, { download: downloads[item.downloadKey] });

const entries = [
  entry({ id: "installed", availableLocally: true, hasLocalData: true, releaseSortKey: "2026-05", sizeGb: 10 }),
  entry({ id: "available", releaseSortKey: "2026-09", sizeGb: 30 }),
  entry({ id: "paused", releaseSortKey: "2026-01" }),
  entry({ id: "partial", hasLocalData: true, releaseSortKey: "2025-12", sizeGb: 4 }),
  entry({ id: "mac-only", hostCompatible: false, releaseSortKey: "2026-10" }),
  entry({ id: "mac-only-installed", hostCompatible: false, availableLocally: true, hasLocalData: true }),
];

const ids = (rows: Array<{ entry: ModelEntry }>) => rows.map((row) => row.entry.id);

describe("buildModelRows", () => {
  it("hides rows the host cannot run unless they are on disk", () => {
    const { rows, hostVisibleCount } = buildModelRows({ entries, resolveStatus: resolve, availability: "all" });
    expect(ids(rows)).not.toContain("mac-only");
    expect(ids(rows)).toContain("mac-only-installed");
    expect(hostVisibleCount).toBe(5);
  });

  it("filters by availability bucket", () => {
    const pick = (availability: "installed" | "available" | "downloading" | "attention") =>
      ids(buildModelRows({ entries, resolveStatus: resolve, availability }).rows).sort();
    expect(pick("installed")).toEqual(["installed", "mac-only-installed"]);
    expect(pick("available")).toEqual(["available"]);
    expect(pick("downloading")).toEqual(["paused"]);
    expect(pick("attention")).toEqual(["partial"]);
  });

  it("counts buckets before the availability filter, after search and groups", () => {
    const result = buildModelRows({
      entries,
      resolveStatus: resolve,
      availability: "installed",
      query: "a",
      matches: (item, query) => item.id.startsWith(query),
    });
    expect(ids(result.rows)).toEqual([]);
    expect(result.counts).toEqual({ all: 1, installed: 0, available: 1, downloading: 0, attention: 0 });
  });

  it("applies a group predicate only when the group is not at its default", () => {
    const group: ModelFilterGroup = {
      id: "size",
      label: "Size",
      kind: "select",
      value: "big",
      defaultValue: "all",
      options: [],
      onChange: () => {},
      predicate: (item, value) => (value === "big" ? (item.sizeGb ?? 0) > 5 : true),
    };
    expect(ids(buildModelRows({ entries, resolveStatus: resolve, availability: "all", groups: [group] }).rows).sort()).toEqual([
      "available",
      "installed",
    ]);
    expect(
      buildModelRows({ entries, resolveStatus: resolve, availability: "all", groups: [{ ...group, value: "all" }] }).rows,
    ).toHaveLength(5);
  });

  it("sorts newest release first by default, then by name", () => {
    expect(ids(buildModelRows({ entries, resolveStatus: resolve, availability: "all" }).rows)).toEqual([
      "available",
      "installed",
      "paused",
      "partial",
      "mac-only-installed",
    ]);
  });
});

describe("sortModelRows", () => {
  const rows = entries.slice(0, 4).map((item) => ({ entry: item, status: resolve(item) }));
  const size: ModelSortOption = { id: "size", label: "Size", defaultDir: "desc", value: (row) => row.entry.sizeGb };

  it("keeps rows without a number last in both directions", () => {
    expect(ids(sortModelRows(rows, size, "desc"))).toEqual(["available", "installed", "partial", "paused"]);
    expect(ids(sortModelRows(rows, size, "asc"))).toEqual(["partial", "installed", "available", "paused"]);
  });

  it("flips a text comparison with the direction", () => {
    const name: ModelSortOption = {
      id: "name",
      label: "Name",
      defaultDir: "asc",
      compare: (a, b) => a.entry.name.localeCompare(b.entry.name),
    };
    expect(ids(sortModelRows(rows, name, "asc"))).toEqual(["available", "installed", "partial", "paused"]);
    expect(ids(sortModelRows(rows, name, "desc"))).toEqual(["paused", "partial", "installed", "available"]);
  });
});

describe("hasActiveModelFilters", () => {
  it("is false at the defaults and true for any change", () => {
    const base = { availability: "all" as const, sortId: "release", defaultSortId: "release", sortDir: "desc" as const, defaultSortDir: "desc" as const };
    expect(hasActiveModelFilters(base)).toBe(false);
    expect(hasActiveModelFilters({ ...base, query: " flux " })).toBe(true);
    expect(hasActiveModelFilters({ ...base, availability: "installed" })).toBe(true);
    expect(hasActiveModelFilters({ ...base, sortId: "name" })).toBe(true);
    expect(hasActiveModelFilters({ ...base, sortDir: "asc" })).toBe(true);
  });
});
