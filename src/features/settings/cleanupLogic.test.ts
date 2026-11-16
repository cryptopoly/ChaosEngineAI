import { describe, expect, it } from "vitest";
import type { CleanupDeleteResult, CleanupItems } from "../../api/cleanup";
import {
  dateFromSeconds,
  deleteBody,
  emptySelection,
  groupState,
  pruneSelection,
  selectionTotals,
  setGroup,
  summarizeResult,
  toggleId,
} from "./cleanupLogic";

const items: CleanupItems = {
  projects: [
    { id: "p1", name: "One", shots: 2, bytes: 1000, lastEdited: 1_790_000_000 },
    { id: "p2", name: "Two", shots: 0, bytes: 2000, lastEdited: 1_790_000_100 },
  ],
  threads: [{ id: "t1", title: "Plans", messages: 3, bytes: 300, lastUsed: null }],
  outputs: [
    { id: "image:a.json", kind: "image", name: "a", bytes: 50, createdAt: "2026-11-01T10:00:00Z" },
    { id: "video:b.json", kind: "video", name: "b", bytes: 5000, createdAt: "" },
  ],
};

describe("selection", () => {
  it("starts with nothing ticked", () => {
    const sel = emptySelection();
    expect(selectionTotals(items, sel)).toEqual({ count: 0, bytes: 0, byGroup: { projects: 0, threads: 0, outputs: 0 } });
    expect(deleteBody(items, sel)).toEqual({ projects: [], threads: [], outputs: [] });
  });

  it("toggles one id and totals the count and size", () => {
    let sel = toggleId(emptySelection(), "projects", "p2");
    sel = toggleId(sel, "outputs", "video:b.json");
    expect(selectionTotals(items, sel)).toEqual({ count: 2, bytes: 7000, byGroup: { projects: 1, threads: 0, outputs: 1 } });
    sel = toggleId(sel, "projects", "p2");
    expect(selectionTotals(items, sel).count).toBe(1);
  });

  it("select all per group ticks and clears only that group", () => {
    let sel = setGroup(emptySelection(), items, "projects", true);
    expect(groupState(items, sel, "projects")).toBe("all");
    expect(groupState(items, sel, "threads")).toBe("none");
    sel = toggleId(sel, "projects", "p1");
    expect(groupState(items, sel, "projects")).toBe("some");
    sel = setGroup(sel, items, "projects", false);
    expect(groupState(items, sel, "projects")).toBe("none");
  });

  it("sends only ticked ids of listed items", () => {
    // An id that is ticked but no longer listed never reaches the request.
    const sel = toggleId(toggleId(emptySelection(), "threads", "t1"), "threads", "gone");
    expect(deleteBody(items, sel)).toEqual({ projects: [], threads: ["t1"], outputs: [] });
  });

  it("prunes ticks for items that are no longer listed", () => {
    const sel = setGroup(emptySelection(), items, "outputs", true);
    const after = pruneSelection(sel, { ...items, outputs: [items.outputs[0]] });
    expect([...after.outputs]).toEqual(["image:a.json"]);
  });
});

describe("result", () => {
  it("counts deleted items, freed bytes and the refused ones with their reasons", () => {
    const result: CleanupDeleteResult = {
      deleted: 1,
      refused: 1,
      freedBytes: 1000,
      results: [
        { kind: "project", id: "p1", name: "One", ok: true, bytes: 1000, reason: null },
        { kind: "project", id: "p2", name: "Two", ok: false, bytes: 0, reason: "A batch is running for this project." },
      ],
    };
    const summary = summarizeResult(result);
    expect(summary.deleted).toBe(1);
    expect(summary.freedBytes).toBe(1000);
    expect(summary.refused.map((r) => r.reason)).toEqual(["A batch is running for this project."]);
  });

  it("turns epoch seconds into a date and refuses bad values", () => {
    expect(dateFromSeconds(1_790_000_000)?.getTime()).toBe(1_790_000_000_000);
    expect(dateFromSeconds(null)).toBeNull();
    expect(dateFromSeconds(0)).toBeNull();
    expect(dateFromSeconds(Number.NaN)).toBeNull();
  });
});
